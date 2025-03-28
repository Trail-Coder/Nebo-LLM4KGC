import os
import openai
from openai import OpenAI
from request_retry import send_request_with_retries
import requests
import random
from util_func import *
import time
import datetime
import argparse



parser = argparse.ArgumentParser(description="triple classification")
parser.add_argument("--model_id", type=str, help="model name")
parser.add_argument("--test_file", type=str, required=True, help="Path to the test file")
parser.add_argument("--entity2txt_file", type=str, required=True, help="Path to the entity to text file")
parser.add_argument("--relation2txt_file", type=str, required=True, help="Path to the relation to text file")
parser.add_argument("--api_file", type=str, required=True, help="Path to the api config file")
parser.add_argument("--ans_file", type=str, required=True, help="Path to the answer file")


parser.add_argument("--LLM_evaluate", type=str, action="store_true")
parser.add_argument("--task_type", type=str,  default="triple classification", choices=["relation prediction", "triple classification", "entity prediction"], help="Task for the LLM")

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


start_line = 0 # 从0开始计数

ent2txt = get_txt(entity2txt_file, 0, 1)
rel2txt = get_txt(relation2txt_file, 0, 1)


res_file = "result/"+test_file.split("/")[1]+"/triple-"+model+"-gpt_test-"+str(time.strftime('%m%d_%H%M',time.localtime()))+".csv"



lines_to_write = []
labels = []
count = 0
correct_count = 0
tn = 0
tp = 0
fp = 0
fn = 0

with open(test_file, "r") as f:
    lines = f.readlines()[start_line:]
    for line in lines:
        tmp = line.strip().split("\t")
        try:
            prompt = "Is the following assert true?" + ent2txt[tmp[0]] + " " + rel2txt[tmp[1]] + " " + ent2txt[tmp[2]] + "?"
            response = client.chat.completions.create(
                        messages=[
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": "The first entity is:"+ent2txt[tmp[0]]+". the second entity is:"+ent2txt[tmp[2]]+". the relation is:"+rel2txt[tmp[1]]+". The question is:"+prompt }
                        ],
                        model=model)
            ans = response.choices[0].message.content.lower()
        except Exception as e:
            ans = "I am not sure"
            print(f"Request failed with error: {e}")
            # with open("error.txt", "w", encoding="utf-8") as fff:
            #     fff.write(str(start_line)+"\t"+prompt+"\n")
        
        count += 1
        label = tmp[3]
        
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
        
        
        triple_txt = ent2txt[tmp[0]] + " " + rel2txt[tmp[1]] + " " + ent2txt[tmp[2]]
        lines_to_write.append(triple_txt +"\t" + ans.replace("\n", ".") +"\t"+ tmp[3] + "\n")

        
        start_line += 1

with open(res_file, "w", encoding="utf-8") as f:
    f.writelines(lines_to_write)

acc = (tn+tp)/(tn+tp+fp+fn)
pre = tp/(tp+fp)
recall = tp/(tp+fn)
f1 = 2 * pre * recall / (pre+recall)


# print("# test quick acc:", correct_count/count)
print('# TP: {}, TN: {}, FP: {}, FN: {}'.format(tp,tn,fp,fn))
print('# ACC: ', acc)
print('# Precision: ', pre)
print('# Recall: ', recall)
print("# F1: ", f1)
test_result = "model:{}\nsystem content:{}\ntest file:{}\nresult file:{}\nTP: {}, TN: {}, FP: {}, FN: {}\nAcc:{}\nPrecision:{}\nRecall: {}\nF1: {}\n\n".format(model,system_content,test_file,res_file,tp,tn,fp,fn,acc,pre,recall,f1)

with open(ans_file, "a+", encoding="utf-8") as f:
    f.write(test_result)

# nohup python test_gpt_triple.py > result/logs/gpt_triple.txt 2>&1 &

# 3903285




