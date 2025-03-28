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
from sklearn.metrics import precision_score, recall_score, accuracy_score


home_dir = os.path.expanduser('~')

parser = argparse.ArgumentParser(description="LLM Evaluation Script")
parser.add_argument("--model_id", type=str, required=True, help="Path to the LLM model")
parser.add_argument("--test_file", type=str, required=True, help="Path to the test file")
parser.add_argument("--rel2txt_file", type=str, required=True, help="Path to the relation2text file")
parser.add_argument("--system_content", type=str, required=True, help="System content file for the LLM")
parser.add_argument("--LLM_evaluate", type=str, action="store_true")
parser.add_argument("--task_type", type=str,  default="relation prediction", choices=["relation prediction", "triple classification", "entity prediction"], help="System content for the LLM")


parser.add_argument("--top_p", type=float, required=True, help="try 1")
parser.add_argument("--top_k", type=int, required=True, help="try 50")
parser.add_argument("--num_beams", type=int, required=True, help="try 4")
parser.add_argument("--max_new_tokens", type=int, required=True, help="try 512")
parser.add_argument("--temperature", type=float, required=True, help="try 0.1")

args = parser.parse_args()

model_id = args.model_id
test_file = args.test_file
task_type = args.task_type
rel2txt_file = args.rel2txt_file
with open(args.system_content, 'r') as sys_content_file:
    system_content = sys_content_file.readline()

rel2txt = get_txt(rel2txt_file, 0, 1)

#=======================================
categories = list(rel2txt.values())
res_file = test_file.split('.')[0].replace("data", "output")+"-"+model_id.split('/')[-1]+"-" +str(time.strftime('%m%d_%H%M',time.localtime())) +".csv"

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
predictions = []

with open(test_file, "r", encoding="utf-8") as f:
    lines = f.readlines()
    if 'prompt' in str(lines[0]):
        lines = lines[1:]

    for line in tqdm(lines):
        tmp = line.strip().split("\t")

        prompt = tmp[0]
        label = tmp[1]
        # if label == "is affiliated to" or label == "plays for":
        #     continue

        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": str(prompt) }
        ]

        labels.append(label)
        try:
            response = pipe(
                messages,
                eos_token_id=terminators,
                do_sample=True
            )
            ans = response[0]["generated_text"][-1]["content"].lower()
        except Exception as e:
            print(e)
            continue

        count += 1
        
        # print(count)
        if args.LLM_evaluate:
            if llm_eval_ans_label(ans, label, task_type) == 1:
                correct_count += 1
                predictions.append(label)
            elif llm_eval_ans_label(ans, "plays for", task_type) and label.find("is affiliated to") != -1:
                correct_count += 1
                predictions.append(label)
            elif llm_eval_ans_label(ans, "is affiliated to", task_type) and label.find("plays for") != -1:
                correct_count += 1
                predictions.append(label)
            else:
                predictions.append(ans)
        else:
            if ans.find(label) != -1:
                correct_count += 1 
                predictions.append(label)
            elif ans.find("plays for") != -1 and label.find("is affiliated to") != -1:
                correct_count += 1
                predictions.append(label)
            elif label.find("plays for") != -1 and ans.find("is affiliated to") != -1:
                correct_count += 1
                predictions.append(label)
            else:
                predictions.append(ans)

        # print(prompt)
        # print("-------------")
        # print("\n",correct_count/count,"\n")
        # print("-------------")
        # print(ans)
        # print(label)
        # print("==============================================")
        lines_to_write.append(prompt+"\t"+ans.replace("\n",".")+"\t"+label+"\n")


with open(res_file, "w", encoding="utf-8") as f:
    # f.write('prompt\tgenerated\n')
    f.writelines(lines_to_write)



merged_predictions = ["plays for or is affiliated to" if x in ["plays for", "is affiliated to"] else x for x in predictions]
merged_ground_truth = ["plays for or is affiliated to" if x in ["plays for", "is affiliated to"] else x for x in labels]

try:
    categories.remove('plays for')
    categories.remove('is affiliated to')
    categories.append("plays for or is affiliated to")
except Exception as e:
    pass
merged_categories = categories



weighted_precision = precision_score(merged_ground_truth, merged_predictions, labels=merged_categories, average="weighted")
weighted_recall = recall_score(merged_ground_truth, merged_predictions, labels=merged_categories, average="weighted")

c = calculate_multiclass_confusion_counts(merged_predictions, merged_ground_truth, categories)
acc = accuracy_score(merged_ground_truth, merged_predictions)


print("# ===================================================")
print("# weighted Precision:", weighted_precision)
print("# weighted Recall:", weighted_recall)
print("# ACC:", acc)