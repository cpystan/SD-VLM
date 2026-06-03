import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
import argparse
import numpy as np
import openai
from tqdm import tqdm
from openai import OpenAI
import openai

def gpt(prompt,key):
    client = OpenAI(api_key=key)

    completion = client.chat.completions.create(
        model='gpt-4-turbo',
        messages=[
            {"role": "user", "content": prompt},
        ],
    )
    return completion.choices[0].message.content



class RunningAverage:
    def __init__(self):
        self.avg = 0
        self.count = 0

    def append(self, value):
        self.avg = (value + self.count * self.avg) / (self.count + 1)
        self.count += 1

    def get_value(self):
        return self.avg


class RunningAverageDict:
    """A dictionary of running averages."""

    def __init__(self):
        self._dict = dict(
            a1=RunningAverage(),
            a2=RunningAverage(),
            a3=RunningAverage(),
            abs_rel=RunningAverage(),
            rmse=RunningAverage(),
            log_10=RunningAverage(),
            rmse_log=RunningAverage(),
            silog=RunningAverage(),
            sq_rel=RunningAverage(),
        )

    def update(self, new_dict):
        if new_dict is None:
            return

        if self._dict is None:
            self._dict = dict()
            for key, value in new_dict.items():
                self._dict[key] = RunningAverage()

        for key, value in new_dict.items():
            self._dict[key].append(value)

    def get_value(self):
        if self._dict is None:
            return None
        return {key: value.get_value() for key, value in self._dict.items()}


def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25**2).mean()
    a3 = (thresh < 1.25**3).mean()

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err**2) - np.mean(err) ** 2) * 100

    log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()
    return dict(
        a1=a1, a2=a2, a3=a3, abs_rel=abs_rel, rmse=rmse, log_10=log_10, rmse_log=rmse_log, silog=silog, sq_rel=sq_rel
    )




def evaluate_quan_dist_question(question, answer, pred):

    prompt = f"""
You should help me to evaluate the response given the question and the correct answer.
You need to convert the measurement of the correct answer and response to meters. The conversion factors are as follows: 1 inch = 0.0254 meters. 1 foot = 0.3048 meters. 1 centimeter (cm) = 0.01 meters.
You should output two floats in meters, one for the answer, and one for the response. If the answer or reponse contains more than one number for prediction, you should output the List contains the number.
The output should be in JSON format.


Example 1:
Question: How tall is the long brown table opposite the crossed table?
Answer: The height of the long brown table opposite the crossed table is 1.02 m.
Response: It is 2.17 meters wide.
"answer_in_meters": 1.02, "response_in_meters": 2.17

Example 2:
Question: what's the total number of chairs in the image?
Answer: 2.
Response: There are 2 chairs.
"answer_in_meters": 2, "response_in_meters": 2

Example 3:
Question: What is the size of the dark pillow?
Answer: The dark pillow is with the size of 0.8 m x 0.63 m x 0.55 m
Response: It is 35.9 inches wide.
"answer_in_meters": [0.78,0.63,0.55], "response_in_meters": 0.91

Example 4:
Question: The height of the bed is 0.81 m, what is the height of the table and the nightstand ?
Answer: Since the height of the bed is 0.81 m, i think the height of the table is 1.02 meters and the height of the nightstand is 0.93 meters.
Response: Since the height of the bed is 0.81 m, i think the height of the table is 1.36 meters and the height of the nightstand is 0.77 meters
"answer_in_meters": [1.02,0.93], "response_in_meters":[1.36,0.77]

Your Turn:
Question: {question}
Answer: {answer}
Response: {pred}

"""
    return prompt.replace('{question}',question).replace('{answer}',answer).replace('{pred}',pred)




def evaluate_qual_question(question, answer, pred):
    prompt = f"""
You should help me to evaluate the response given the question and the correct answer.
To mark a response, you should output a single integer between 0 and 1.
1 means that the response perfectly matches the answer.
0 means that the response is completely different from the answer.
The output should be in JSON format.

Example 1:
Question: Is the blue bed to the left of the curtain from the viewer's perspective ?
Answer: Indeed, the bed is to the left of the curtain.
Response: Yes, the blue bed is positioned on the left side of the curtain.
"your_mark": 1

Example 2:
Question: Between the wooden table and the black chair, which on is taller?
Answer: The wooden table is taller.
Response: The chair.
"your_mark": 0

Example 3:
Question: What is the tallest among the table, the chaird, and the curtain?
Answer: The tallest is the curtain.
Response: The curtain.
"your_mark": 1


"""

    post_fix = f"""

Your Turn:
Question: {question}
Answer: {answer}
Response: {pred}
    """

    prompt = prompt + post_fix
    return prompt.replace('{question}',question).replace('{answer}',answer).replace('{pred}',pred)


def main(args):
    quantitative_types = ['scale_estimation','absolute_distance','count','grounding','refer_obj_estimation']
    qualitative_types = ['relative_position','scale_compare','existence']

    #gpt = args.gpt
    openai.api_key = args.key
    
    paths = [args.path]
    for path in paths:
        #data_path = sys.argv[1]
        data_path = path
        with open(data_path) as f:
            lines = f.readlines()

        total = len(lines)

        qualitative_dict = defaultdict(list)
        quantitative_success_dict = defaultdict(list)

        raw_list = []

        match_fail_count = 0


        middle=[]
        middle.append(lines[0])
        count=0
        for line in tqdm(lines[:):
            count+=1

            match_success = False
            data = json.loads(line)

            data["llm_match_info"] = {}

            try:
                if data["type"] in quantitative_types:
                    llama_evaluation = evaluate_quan_dist_question(
                        question=data["question"], answer=data["gt"], pred=data["pred"]
                    )
                elif data["type"] in qualitative_types:
                    llama_evaluation = evaluate_qual_question(
                            question=data["question"], answer=data["gt"], pred=data["pred"]
                        )
            except:
                print(count)
                print(line)

            middle.append({'input':llama_evaluation,'answer':None,'type':data['type']})

        for line in tqdm(middle[1:]):
            try:
                prompt = line['input']
                response = gpt(prompt,args.key)
                line['answer'] =response
            except:
                print('error')
                continue
        save_path = data_path.replace('.json','_gpt.json')
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(middle, f)





if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--path", type=str, default="./preds.json")

    #api setting:
    parser.add_argument("--gpt", type=str, default="gpt-4-turbo")
    parser.add_argument("--key", type=str, default="key")
    args = parser.parse_args()
    main(args)
