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
import copy

import numpy as np


from sentence_transformers import SentenceTransformer, util

# 加载预训练模型
model = SentenceTransformer('all-MiniLM-L6-v2')  # 轻量级模型

  

class Counter_quantity():
    def __init__(self,):
        self.counter = {'total':{'count': 0, 'rate': [0,0,0],'ratio':[0,0,0]}}
        self.error = {'total':{'count': 0, 'rate': 0,'ratio':0}}
    def add(self,type_name,rate):
        name = 'total'
        self.counter[name]['count']+=1
        for i in range(len(rate)):
            s = rate[i]
    
            self.counter[name]['rate'][i]+=s
            self.counter[name]['ratio'][i]= self.counter[name]['rate'][i] /self.counter[name]['count']

        if not type_name in self.counter.keys():
            self.counter[type_name] ={'count': 0, 'rate': [0,0,0],'ratio':[0,0,0]}

        self.counter[type_name]['count'] +=1
        for i in range(len(rate)):
            s = rate[i]
    
            self.counter[type_name]['rate'][i]+=s
            self.counter[type_name]['ratio'][i]= self.counter[type_name]['rate'][i] /self.counter[type_name]['count']


    def error_add(self,type_name,rate):
        name = 'total'
        self.error[name]['count']+=1

        s = rate
    
        self.error[name]['rate']+=s
        self.error[name]['ratio']= self.error[name]['rate'] /self.error[name]['count']

        if not type_name in self.error.keys():
            self.error[type_name] ={'count': 0, 'rate': 0,'ratio':0}
            self.error[type_name]['count'] +=1
            self.error[type_name]['rate']+=s
            self.error[type_name]['ratio']= self.error[type_name]['rate']/self.error[type_name]['count']
        else:
            self.error[type_name]['count'] +=1
            self.error[type_name]['rate']+=s
            self.error[type_name]['ratio']= self.error[type_name]['rate']/self.error[type_name]['count']

class Counter_quality():
    def __init__(self):
        self.counter = {'total':{'count': 0, 'rate': 0,'ratio':0}}
    def add(self,type_name,rate):
        name = 'total'
        self.counter[name]['count']+=1

        s = rate
    
        self.counter[name]['rate']+=s
        self.counter[name]['ratio']= self.counter[name]['rate'] /self.counter[name]['count']

        if not type_name in self.counter.keys():
            self.counter[type_name] ={'count': 0, 'rate': 0,'ratio':0}

        self.counter[type_name]['count'] +=1
        
        s = rate

        self.counter[type_name]['rate']+=s
        self.counter[type_name]['ratio']= self.counter[type_name]['rate'] /self.counter[type_name]['count']



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





def main(args):

    quantitative_types = ['scale_estimation','absolute_distance','count','grounding','refer_obj_estimation']
    qualitative_types = ['relative_position','scale_compare','existence']



    #data_path = sys.argv[1]
    data_path = args.path
  

    with open(data_path) as f:
        t_lines = json.load(f)
    lines = copy.deepcopy(t_lines)
    total = len(lines)

    qualitative_dict = defaultdict(list)
    quantitative_success_dict = defaultdict(list)
    quantitative_error_dict = defaultdict(list)

    raw_list = []

    match_fail_count = 0
    counter = Counter_quantity()
    counter2 = Counter_quality()

    for line in tqdm(lines[1:]):


        match_success = False
        
        data=line
        data["llm_match_info"] ={}


        if data["type"] in quantitative_types:


                if data["type"]  == "position":

                    try:
                        llama_evaluation = eval(data['answer'])
                        answer_in_pos, response_in_pos = (llama_evaluation["answer_in_meters"]), (
                            llama_evaluation["response_in_meters"]
                        )

                        if isinstance(answer_in_pos,str):
                            embeddings = model.encode([answer_in_pos,response_in_pos])

                            # 计算余弦相似度
                            similarity = util.cos_sim(embeddings[0], embeddings[1])
                            
                            success = [float(similarity) >=0.5]
                            match_success = True
                            error_rate=0
                        else:

                            ratio = []
                            error_rate =[]
                            for j in range(len(answer_in_pos)):
                                ratio.append( abs(response_in_pos[j]-answer_in_pos[j]))
                                error_rate.append((np.abs(response_in_pos[j] - answer_in_pos[j])) / (answer_in_pos[j] + 1e-4))
                            ratio = sum(ratio) /len(ratio)
                            error_rate = sum(error_rate)/len(error_rate)

                            success = [ratio <= 0.1]
                            match_success = True
                    except:
                        answer_in_clock = response_in_clock = "N/A"
                        match_success = False
                        match_fail_count += 1
                        success = 0



                else:
                    try:

                        llama_evaluation = eval(data['answer'])
                        answer_in_meters, response_in_meters = llama_evaluation["answer_in_meters"], llama_evaluation["response_in_meters"]

                        if isinstance(answer_in_meters,list):
                            for item in answer_in_meters:
                                item = float(item)
                            for item in response_in_meters:
                                item = float(item)
                            ratio = []
                            error_rate =[]
                            for j in range(len(answer_in_meters)):
                                ratio.append( max(response_in_meters[j]/answer_in_meters[j], answer_in_meters[j]/response_in_meters[j]))
                                error_rate.append((np.abs(response_in_meters[j] - answer_in_meters[j])) / (answer_in_meters[j] + 1e-4))
                            ratio = sum(ratio) /len(ratio)
                            error_rate = sum(error_rate)/len(error_rate)
                        else:
                            answer_in_meters, response_in_meters = float(llama_evaluation["answer_in_meters"]), float(
                            llama_evaluation["response_in_meters"]
                        )
                            ratio = max(response_in_meters/answer_in_meters, answer_in_meters/response_in_meters)
                            error_rate = (np.abs(response_in_meters - answer_in_meters)) / (answer_in_meters + 1e-4)
                        success = [ratio<1.25,ratio<(1.25**2),ratio<(1.25**3)]
                        
                        match_success = True

                    except:
                        answer_in_meters = response_in_meters = "N/A"
                        match_success = False
                        match_fail_count += 1
                        success = 0

                    data["llm_match_info"]["answer"] = answer_in_meters
                    data["llm_match_info"]["response"] = response_in_meters

                if match_success:
                    counter.add(data['type'],success)
                    counter.error_add(data['type'],error_rate)
                else:
                    counter.add(data['type'],[int(success),int(success),int(success)])


        elif data["type"] in qualitative_types:

            llama_evaluation = 0
            try:
                llama_evaluation = eval(data['answer'])['your_mark']

                if llama_evaluation is None:
                    print("Got None from evaluation")
                    success = 0
                    llama_evaluation = 0

                data["llm_match_info"]["evaluation"] = int(llama_evaluation)
                match_success = True
            except:
                data["llm_match_info"]["evaluation"] = "N/A"
                match_success = False
                match_fail_count += 1
                success = 0

            # if match_success:
            counter2.add(data['type'],int(llama_evaluation > 0.5))
        else:
            print(f"{data['type']} not found")
            exit()
        raw_list.append(data)

    average=0
    for k,v in counter.counter.items():
        if not k=='total':
            average += v['ratio'][0]
    for k,v in counter2.counter.items():
        if not k=='total':
            average += v['ratio']
    average = average/8

    print(f'\n msmu_val: \n {lines[0]} \n counter: {counter.counter} \n {counter2.counter} \n ave:{average} \n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--path", type=str, default="path-to-pred_gpt")

    parser.add_argument("--gt-depth", type=str, default="False")
    parser.add_argument("--vision-tower", type=str, default="path-to-vit")


    args = parser.parse_args()

    main(args)
