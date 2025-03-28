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
from tqdm import tqdm

parser = argparse.ArgumentParser(description="triple classification")
parser.add_argument("--model_id", type=str, help="Path to the LLM model")
parser.add_argument("--test_file", type=str, required=True, help="Path to the test file")
parser.add_argument("--do_int8", action="store_true", help="Enable int8 optimization")
parser.add_argument("--low_cpu_mem_usage", action="store_true", default=True, help="Enable low CPU memory usage")
parser.add_argument("--LLM_evaluate", type=str, action="store_true")
parser.add_argument("--task_type", type=str,  default="triple classification", choices=["relation prediction", "triple classification", "entity prediction"], help="System content for the LLM")

parser.add_argument("--system_content", type=str, help="System prompt file for the LLM")
parser.add_argument("--top_p", type=float, required=True, help="try 1")
parser.add_argument("--top_k", type=int, required=True, help="try 50")
parser.add_argument("--num_beams", type=int, required=True, help="try 4")
parser.add_argument("--max_new_tokens", type=int, required=True, help="try 512")
parser.add_argument("--temperature", type=float, required=True, help="try 0.1")

args = parser.parse_args()

home_dir = os.path.expanduser('~')
model_id = args.model_id
test_file = args.test_file
do_int8 = args.do_int8
task_type = args.task_type
low_cpu_mem_usage = args.low_cpu_mem_usage
with open(args.system_content, 'r') as sys_content_file:
    system_content = sys_content_file.readline()

res_file = test_file.split(".")[0].replace("data", "output")+"-"+model_id.split("/")[-1]+"-"+str(time.strftime('%m%d_%H%M',time.localtime())) +".csv"
pipe = pipeline(
    "text-generation",
    model=model_id,
    # model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
    do_sample=False,
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
tn = 0
tp = 0
fp = 0
fn = 0
   
with open(test_file, "r", encoding="utf-8") as f:
    lines = f.readlines()
    for line in tqdm(lines[1:]):
        tmp = line.strip().split("\t")
        prompt = tmp[0]
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": str(prompt) }
        ]
        response = pipe(
            messages,
            # max_new_tokens=256,
            eos_token_id=terminators,
            # do_sample=False,
            # temperature=0.6,
            # top_p=1,
            # top_k=40,
            # num_beams=4,
        )
        ans = response[0]["generated_text"][-1]["content"].lower()

        count += 1
        label = tmp[1]

        if args.LLM_evaluate:
            if llm_eval_ans_label(ans, "yes", task_type) == 1:
                if label == "1":
                    correct_count += 1
                    tp += 1
                else:
                    fp += 1
            elif llm_eval_ans_label(ans, "false", task_type) == 1 or llm_eval_ans_label(ans, "no", task_type) == 1:
                if label == "-1":
                    correct_count += 1
                    tn += 1
                else:
                    fn += 1
        else:
            if ans.find("yes") != -1:
                if label == "1":
                    correct_count += 1
                    tp += 1
                else:
                    fp += 1
            elif (ans.find("false") != -1 or ans.find("not") != -1 or ans.find("n't") != -1 or ans.find("no") != -1):
                if label == "-1":
                    correct_count += 1
                    tn += 1
                else:
                    fn += 1

        # print(prompt)
        # # print(triple_txt)
        # print(ans)
        # print(label)
        # print("Acc: ",correct_count/count)
        # print("==============================================")
        lines_to_write.append(prompt+"\t"+ans.replace("\n",".") + "\t" + label + "\n")        

with open(res_file, "w", encoding="utf-8") as f:
    f.writelines(lines_to_write)

acc = (tn+tp)/(tn+tp+fp+fn)
pre = tp/(tp+fp)
recall = tp/(tp+fn)
f1 = 2 * pre * recall / (pre+recall)
print('# TP: {}, TN: {}, FP: {}, FN: {}'.format(tp,tn,fp,fn))
print('# ACC: ', acc)
print('# Precision: ', pre)
print('# Recall: ', recall)
print("# F1: ", f1)