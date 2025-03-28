import os
from util_func import *
import torch
import json
import argparse
import threading
import time
from accelerate import init_empty_weights, infer_auto_device_map
import transformers
from transformers import AutoConfig, pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from transformers import StoppingCriteria, StoppingCriteriaList
from loguru import logger
from typing import List, Union
from vllm import LLM

home_dir = os.path.expanduser('~')

parser = argparse.ArgumentParser(description="LLM Evaluation Script")

parser.add_argument("--model_id", type=str, required=True, help="Path to the LLM model")
parser.add_argument("--test_file", type=str, required=True, help="Path to the test file")
parser.add_argument("--system_content", type=str, required=True, help="System content for the LLM")
parser.add_argument("--LLM_evaluate", type=str, action="store_true")
parser.add_argument("--task_type", type=str,  default="entity prediction", choices=["relation prediction", "triple classification", "entity prediction"], help="System content for the LLM")

parser.add_argument("--top_p", type=float, required=True, help="try 0.75")
parser.add_argument("--top_k", type=int, required=True, help="try 40")
parser.add_argument("--num_beams", type=int, required=True, help="try 4")
parser.add_argument("--max_new_tokens", type=int, required=True, help="try 512")
parser.add_argument("--temperature", type=float, required=True, help="try 0.1")

args = parser.parse_args()

model_id = args.model_id
test_file = args.test_file
task_type = args.task_type
with open(args.system_content, 'r') as sys_content_file:
    system_content = sys_content_file.readline()

res_file = test_file.split('.')[0].replace("data", "output")+"-"+model_id.split('/')[-1]+"-" +str(time.strftime('%m%d_%H%M',time.localtime())) +".csv"
mode = 'head' if test_file.find('head') != -1 else 'tail'
system_content = system_content.format(mode)


pipe = pipeline(
    "text-generation",
    model=model_id,
    # model_kwargs={"torch_dtype": torch.bfloat16},
    # model_kwargs={"torch_dtype": torch.float64},
    device_map="auto",
    temperature=args.temperature, # consider turn it higher if "RuntimeError: probability tensor contains either inf, nan or element < 0"
    top_p=args.top_p,
    top_k=args.top_k,
    num_beams=args.num_beams,
    max_new_tokens=args.max_new_tokens
)

terminators = [
    pipe.tokenizer.eos_token_id,
    pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

lines_to_write = []
labels = []
count = 0
correct_count = 0
rotate_correct = 0
error_count = 0

with open(test_file, "r", encoding="utf-8") as f:
    lines = f.readlines()
    if 'prompt' in str(lines[0]):
        lines = lines[1:]

    for line in tqdm(lines):
        count+=1
        tmp = line.strip().split("\t")
        prompt = tmp[0]
        label = tmp[1].lower()

        if prompt.split(':')[2].lower().find(label) == -1:
            lines_to_write.append("\n")
            print('Model 1 failed')
            continue
        else:
            rotate_correct += 1

        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": str(prompt) }
        ]

        # labels.append(label)
        try:
            response = pipe(
                messages,
                # max_new_tokens=256,
                eos_token_id=terminators,
                do_sample=True,
                # do_sample=False,
                # top_p=0.75,
                # top_k=40,
                # num_beams=4,
            )
            ans = response[0]["generated_text"][-1]["content"].lower()
        except Exception as e:
            print(e)
            error_count += 1
            continue
        if args.LLM_evaluate:
            if llm_eval_ans_label(ans, label, task_type) == 1:
                ans = label
                correct_count += 1
        else:
            if ans.find(label) != -1:
                ans = label
                correct_count += 1 

        lines_to_write.append(prompt+"\t"+ans.replace("\n",".")+"\t"+label+"\n")

with open(res_file, "w", encoding="utf-8") as f:
    # f.write('prompt\tgenerated\n')
    f.writelines(lines_to_write)

print("# acc:", correct_count/count,"\n")