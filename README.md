# 👷SD-VLM: Spatial Measuring and Understanding with Depth-Encoded Vision-Language Models


[📢 [[Project Page](https://cpystan.github.io/SD_VLM_pages/)] [[Arxiv](https://arxiv.org/abs/2509.17664)]  [[Data](https://huggingface.co/datasets/cpystan/MSMU)] [[Model Zoo](https://huggingface.co/cpystan/SD-VLM-7B)] 



> **We are excited to announce that our paper is accepted by NeurIPS 2025!**

---

## Install

1. Clone this repository
```bash
git clone https://github.com/cpystan/SD-VLM.git
cd SD-VLM
```

2. Install Package
```Shell
conda create -n sdvlm python=3.10 -y
conda activate sdvlm
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

3. Install additional packages for training cases
```
pip install -e ".[train]"
```

### Quick Start With HuggingFace


```Python
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
import copy

model_path = "cpystan/SD-VLM-7B"

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path)
)

input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
ori_img = copy.deepcopy(image)
image_tensor = process_images([image], image_processor, model.config)[0]

with torch.inference_mode():
    output_ids = model.generate(
        input_ids,
        images=image_tensor.unsqueeze(0).half().to(input_ids.device),
        image_sizes=[image.size],
        do_sample=True if temperature > 0 else False,
        temperature=0.2,
        top_p=None,
        num_beams=1,
        ori_imgs = [ori_img],
        max_new_tokens=1024,
        use_cache=True,)
response= tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

```


## MSMU (Massive Spatial Measuring and Understanding) Dataset
For instruction tuning, please download the train.parquet of MSMU Dataset from [[HuggingFace](https://huggingface.co/datasets/cpystan/MSMU)]. 

For evaluation on MSMU-Bench, please download the test.parquet of MSMU Dataset from [[HuggingFace](https://huggingface.co/datasets/cpystan/MSMU)]. 

## Training
SD-VLM inherits the instruction-tuning pipeline of LLaVA, based on the well-established checkpoint of [LLaVA-1.5-7B](https://github.com/haotian-liu/LLaVA/). It requires relatively low resources for GPUs since SD-VLM can be trained with LoRA on 8 V100 GPUs. 

1. LoRA Finetuning (official setting)
```Shell
sh scripts/v1_5/finetune_task_lora.sh
```

2. Non-LoRA Finetuning 
```Shell
sh scripts/v1_5/finetune_task.sh
```

Some arguments in the script need to be modified:

- `--model_name_or_path`: path to the checkpoint of LLaVA-1.5-7B
- `--data_path`: path to the train set of MSMU
- `--vision_tower`: path to clip-vit-large-patch14-336
- `--depth_path`: path to depth_anything_v2_vitl


## Evaluation on MSMU-Bench

Step 1: Load the checkpoint and conduct inference on MSMU-Bench.
```Shell
python evaluation/inference.py --model_path 'SD-VLM-7B' --data 'MSMU' --vision_tower 'clip-vit-large-patch14-336'
```

Step 2: Load the json file saved from step 1 and use GPT-4-Turbo to get the evaluation response.
```Shell
python evaluation/llm_generate.py --path './preds.json' --key 'gpt_key'
```

Step 3: Load the json file saved from step 2 and calculate the final result.
```Shell
python evaluation/evaluate_final.py --path './preds_gpt.json' 
```
## 🚧 Status: Coming Soon
More details are coming soon.

