import os
import re
import base64
import numpy as np
from io import BytesIO
import base64
from PIL import Image
import sys
sys.path.append("./")
import argparse
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
import math
import torch
import json
import copy
from tqdm import tqdm
import pandas as pd
judge_prompt = '''You are a helpful assistant designed to output in the format of given examples.\
You should help me to evaluate the response given the question and the correct answer.\
 You need to convert the distance of the correct answer and response to meters.\
The conversion factors are as follows:\
1 inch = 0.0254 meters. 1 foot = 0.3048 meters. 1 centimeter (cm) = 0.01 meters.\
You should output two floats in meters, one for the answer, and one for the response. \
\n Example 1: \
\n Question: what is the size of the brown table in the image? \
\n Answer: the long brown table opposite the crossed table is with the length of 2.28 m, width of 0.75 m, and height of 0.87 m. \
\n Responsde: The size of the table is 100 centimeters.\
\n {"answer_in_meters": [2.28,0.75,0.87], "response_in_meters": [1]}  \
\n Example 2: \
\n Question: Measure the width of the chair. \
\n Answer: The width of the chair is 1.02 meters. \
\n Responsde: The chair is 2.17 meters wide.\
\n {"answer_in_meters": [1.02], "response_in_meters": [2.17]}  '''


def base64_to_pil(image_base64):
    try:
        # 解码base64字符串
        image_data = base64.b64decode(image_base64)
        # 创建字节流并打开为图像
        image = Image.open(BytesIO(image_data))
        return image
    except Exception as e:
        print(f"转换失败: {e}")
        return None



def eval_llava(args):
    answers_file='./preds.json'
    
    ans_file = open(answers_file, "w")


    ans_file.write(
        json.dumps(
            {
                "model": args.model_path,

            }
        )
        + "\n"
    )

    data_path = args.data
    model_path = args.model_path
    model_base=None
    temperature = 0.2
    top_p = None
    num_beams = 1


    disable_torch_init()
    model_path = os.path.expanduser(model_path)
    model_name = get_model_name_from_path(model_path)

    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name, mm_vision_tower=args.vision_tower)
    model.to(torch.float16)
    model.gt_depth = eval(args.gt_depth)
    model.use_depth = eval(args.use_depth)
    print(f'use gt depth: {model.gt_depth}')

    data = load_dataset(data_path,download_mode="reuse_dataset_if_exists")['test']

    



    count = 0

    for i in tqdm(range(len(data[:]))):
        sample = data[i]
        count+=1
        prompt = sample['conversations'][0]['value']
        image = sample['image']

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        if model.gt_depth:
            ori_img = Image.open(os.path.join(image_folder, depth_path))
        else:
            ori_img = copy.deepcopy(image)


 
        image_tensor = process_images([image], image_processor, model.config)[0]
        


        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().to(input_ids.device),
                image_sizes=[image.size],
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                top_p=top_p,
                num_beams=num_beams,
                ori_imgs = [ori_img],
                # no_repeat_ngram_size=3,
                max_new_tokens=1024,
                use_cache=True,)


        response= tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

       
        gt = sample['conversations'][1]['value']
        print(f'pred:{response}')

        ans_file.write(
                json.dumps(
                    {
                        "question": prompt,
                        "pred": response,
                        "gt": gt,
                        "type": sample['type'],
                    }
                )
                + "\n"
            )

    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="path-to-model")

    parser.add_argument("--data", type=str, default="path-to-benchmark")
    parser.add_argument("--gt-depth", type=str, default="False")
    parser.add_argument("--use-depth", type=str, default="True")
    parser.add_argument("--vision-tower", type=str, default="path-to-vit")
    parser.add_argument("--mode", type=str, default='llava')
    args = parser.parse_args()

    eval_llava(args)