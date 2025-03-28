import os
import openai
from openai import OpenAI
from request_retry import send_request_with_retries
import requests
import random
from util_func import *
import time
import datetime
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, accuracy_score
import argparse


parser = argparse.ArgumentParser(description="triple classification")
parser.add_argument("--model_id", type=str, help="model name")
parser.add_argument("--test_file", type=str, required=True, help="Path to the test file")
parser.add_argument("--entity2txt_file", type=str, required=True, help="Path to the entity to text file")
parser.add_argument("--relation2txt_file", type=str, required=True, help="Path to the relation to text file")
parser.add_argument("--api_file", type=str, required=True, help="Path to the api config file")
parser.add_argument("--ans_file", type=str, required=True, help="Path to the answer file")


parser.add_argument("--LLM_evaluate", type=str, action="store_true")
parser.add_argument("--task_type", type=str,  default="relation prediction", choices=["relation prediction", "triple classification", "entity prediction"], help="Task for the LLM")

parser.add_argument("--system_content", type=str, default="", help="System prompt file for the LLM")

args = parser.parse_args()

with open(args.system_content, 'r') as sys_content_file:
    system_content = sys_content_file.readline()
with open(args.api_file, 'r') as api_config_file:
    config = json.load(api_config_file)
api_key = config["api_key"]
base_url = config["base_url"]
client = OpenAI(
    api_key=api_key,
    base_url=base_url
)

model = args.model_id
test_file = args.test_file
entity2txt_file = args.entity2txt_file
relation2txt_file = args.relation2txt_file
ans_file = args.ans_file
task_type = args.task_type


start_line = 0 
ent2txt = get_txt(entity2txt_file, 0, 1)
rel2txt = get_txt(relation2txt_file, 0, 1)

# error_file = test_file.split(".")[0].replace("data", "result")+"-{}_error.csv".format(model)
res_file = test_file.split(".")[0].replace("data", "result")+model+"-gpt_test-"+str(time.strftime('%m%d_%H%M',time.localtime()))+".csv"

lines_to_write = []
labels = []
count = 0
correct_count = 0
predictions = []
categories = list(rel2txt.values())

with open(test_file, "r") as f:
    lines = f.readlines()[start_line:]
    # print("File opened")
    # test_count = 0
    for line in lines:
    # for line in tqdm(lines):
        # test_count += 1
        # if test_count > 1:
        #     break
        tmp = line.strip().split("\t")
        count += 1
        prompt = tmp[0]
        label = tmp[1].lower()
        labels.append(label)        
        try:
            response = client.chat.completions.create(
                        messages=[
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": prompt } ],
                        model=model)
            ans = response.choices[0].message.content.lower()
        except Exception as e:
            ans = "I am not sure"
            print(f"Request failed with error: {e}")
            with open("error.txt", "w", encoding="utf-8") as fff:
                fff.write(str(start_line)+"\t"+prompt+"\n")
        
        lines_to_write.append(prompt +"\t" + ans.replace("\n", ".") +"\t"+ tmp[1] + "\n")
        if ans.find(label) != -1:
            correct_count  += 1
            predictions.append(label)
        elif label == "is affiliated to" and ans.find("plays for") != -1:
            correct_count  += 1
            predictions.append(label)
        elif label.lower() == "plays for" and ans.lower().find("is affiliated to") != -1:
            correct_count  += 1
            predictions.append(label)
        else:
            predictions.append(ans)
        
        # print(str(start_line)+"\n"+prompt+"\n"+ans+"\n"+label+"\n-----------------------------------------\n")
        start_line += 1

with open(res_file, "w", encoding="utf-8") as f:
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


# print(f"Length of merged_ground_truth: {len(merged_ground_truth)}")
# print(f"Length of merged_predictions: {len(merged_predictions)}")


macro_precision = precision_score(merged_ground_truth, merged_predictions, labels=merged_categories, average="macro")
macro_recall = recall_score(merged_ground_truth, merged_predictions, labels=merged_categories, average="macro")

weighted_precision = precision_score(merged_ground_truth, merged_predictions, labels=merged_categories, average="weighted")
weighted_recall = recall_score(merged_ground_truth, merged_predictions, labels=merged_categories, average="weighted")

c = calculate_multiclass_confusion_counts(merged_predictions, merged_ground_truth, categories)
acc = accuracy_score(merged_ground_truth, merged_predictions)




with open(ans_file, "a+", encoding="utf-8") as f:
    f.write("# ===================================================\n")
    f.write(f"# Total:{len(lines)}, Try: {count}, Successful: {correct_count}\n")
    f.write(f"# test quick acc: {correct_count / count}\n\n")
    f.write(f"# Merged Predictions: {len(merged_predictions)}\n")
    f.write(f"# Merged Ground Truth: {len(merged_ground_truth)}\n")
    f.write(f"# weighted Precision: {weighted_precision}\n")
    f.write(f"# weighted Recall: {weighted_recall}\n")
    f.write(f"# ACC: {acc}\n")



# nohup python test_gpt_rel.py > result/logs/gpt_rel.txt 2>&1 &





