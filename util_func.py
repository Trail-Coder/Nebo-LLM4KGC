from tqdm import tqdm
import os
import openai
from openai import OpenAI
from datetime import datetime
import json
import csv
import random
from collections import defaultdict

current_date = datetime.now()
date_mmdd = current_date.strftime('%02m%02d')

def quick_eval(filename, labels):
    correct_count = 0
    with open(filename, "r") as f:
        lines = f.readlines()
        for (i, line) in enumerate(lines[1:]):
            line = line.strip()
            start_idx = line.find("\"")
            res = line[start_idx + 1: -1]
            
            label = labels[i]

            if (res.find("Yes") != -1 or res.find(" yes") != -1) and label == "1":
                correct_count += 1
            elif (res.find("No") != -1 or res.find("not") != -1 or res.find("n't") != -1 or res.find("no") != -1) and label == "-1":
                correct_count += 1

    return correct_count, 1.0 * correct_count/len(labels)

def eval_results_simple(filename, triple_txt_list, labels, with_tqdm=False) -> list:
    all_count = 0
    correct_count = 0
    lines_to_write = []
    with open(filename, "r", encoding="utf-8") as f:
        all_text = f.read()
        for (i, txt) in enumerate(triple_txt_list):   
            all_count += 1
            txt_idx = all_text.find(txt)
            label = labels[i]
            label_idx = all_text.find("\t" + label + "\n")
            ans = all_text[txt_idx + len(txt) + 1: label_idx]

            all_text = all_text[label_idx + len("\t" + label + "\n"):]

            print("-------------------------\n", i, "\n", ans.replace("\n",""), "\n",label)

            if label == "1" and (ans.find("Yes") != -1 or ans.find("yes")!= -1) :
                correct_count += 1
            if label == "-1" and ans.find("not sure") == -1 and (ans.find("not") != -1 or ans.find("No") != -1 or ans.find("no") != -1 or ans.find("n\'t") != -1):
                correct_count += 1

    acc = 1.0 * correct_count /all_count
    print("total num:", len(labels))
    return [all_count, correct_count, acc]

def llm_eval_ans_label(answer, label, task_type) -> int:
    client = OpenAI(
        api_key="",
        base_url=""
    )

    sys_content = "The following sentence is the answer to a {} question. You should decide whether the answer is the same with the label. You should answer only \"Yes\" or \"No\" or \"I am not sure\" according to the given message, no other words.".format_map(task_type)
    response = "The answer is: {}. The label is: {}".format(answer, label)
    
    try:
        gpt_res = client.chat.completions.create(
            messages = [
                {"role":"system", "content":sys_content},
                {"role":"user", "content":response}
            ],
            model = "claude-3-haiku-20240307"
        )
        ans = gpt_res.choices[0].message.content
    except Exception as e:
        ans = "not sure"
        # print(f"Request failed with error: {e}")
        
    print(ans)
    ans = ans.replace("\n", "")
    # ans = response
    if ans.find("Yes") != -1 or ans.find("yes")!= -1:
        return 1
    else:
        return -1

def is_json(myjson):
    try:
        json.loads(myjson)
    except ValueError:
        return False
    return True

def load_api_config(file_path):
    with open(file_path, 'r') as f:
        config = json.load(f)
    return config

def eval_results_ai(filename, triple_txt_list, labels, with_tqdm=False) -> list:
    all_count = 0
    correct_count = 0

    client = OpenAI(
        api_key="",
        base_url=""
    )


    lines_to_write = []
    with open(filename, "r", encoding="utf-8") as f:
        all_text = f.read()
        if not with_tqdm:
            for (i, txt) in enumerate(triple_txt_list):
                txt_idx = all_text.find(txt)
                label = labels[i]
                label_idx = all_text.find("\t" + label + "\n")
                response = all_text[txt_idx + len(txt) + 1: label_idx]

                all_text = all_text[label_idx + len("\t" + label + "\n"):]

                end_idx  = response.find(".")
                res = response[:end_idx + 1]

                all_count += 1
                prompt = ""
                try:
                    gpt_res = client.chat.completions.create(
                        messages = [
                            {"role":"system", "content":"The following sentence is the answer to a question. Summarize the answer. You should answer only \"Yes\" or \"No\" or \"I am not sure\" according to the given message, no other words. "},
                            {"role":"user", "content":response}
                        ],
                        model = "claude-3-haiku-20240307"
                    )
                    ans = gpt_res.choices[0].message.content
                except Exception as e:
                    ans = "not sure"
                    print(f"Request failed with error: {e}")
                    with open("data/FB13/eval_error"+str(date_mmdd)+".csv", "a", encoding="utf-8") as fff:
                        fff.write(str(i)+"\t"+res+"\n")
                print(ans)
                ans = ans.replace("\n", "")
                # ans = response
                if label == "1" and (ans.find("Yes") != -1 or ans.find("yes")!= -1) :
                    correct_count += 1
                if label == "-1" and ans.find("not sure") == -1 and (ans.find("not") != -1 or ans.find("No") != -1 or ans.find("no") != -1 or ans.find("n\'t") != -1):
                    correct_count += 1
        else:
            for (i, txt) in tqdm(enumerate(triple_txt_list)):   
                txt_idx = all_text.find(txt)
                label = labels[i]
                label_idx = all_text.find("\t" + label + "\n")
                response = all_text[txt_idx + len(txt) + 1: label_idx]

                all_text = all_text[label_idx + len("\t" + label + "\n"):]

                end_idx  = response.find(".")
                res = response[:end_idx + 1]

                # print(label_idx, label)
                # print("\n"+response+"\n")
                # print(len(response))
                # print(end_idx)
                # print(res)
                # print("-------------------------")
                all_count += 1
                prompt = ""
                try:
                    gpt_res = client.chat.completions.create(
                        messages = [
                            {"role":"system", "content":"The following sentence is the answer to a question. Summarize the answer. You should answer only \"Yes\" or \"No\" or \"I am not sure\" according to the given message, no other words. "},
                            {"role":"user", "content":response}
                        ],
                        model = "claude-3-haiku-20240307"
                    )
                    ans = gpt_res.choices[0].message.content
                except Exception as e:
                    ans = "not sure"
                    print(f"Request failed with error: {e}")
                    with open("data/FB13/eval_error"+str(date_mmdd)+".csv", "a", encoding="utf-8") as fff:
                        fff.write(str(i)+"\t"+res+"\n")
                print("-------\n", ans)
                ans = ans.replace("\n", "")
                # ans = response
                if label == "1" and (ans.find("Yes") != -1 or ans.find("yes")!= -1) :
                    correct_count += 1
                if label == "-1" and ans.find("not sure") == -1 and (ans.find("not") != -1 or ans.find("No") != -1 or ans.find("no") != -1 or ans.find("n\'t") != -1):
                    correct_count += 1

    acc = 1.0 * correct_count /all_count
    return [all_count, correct_count, acc]


def get_txt(filename, position_a, position_b):
    txt_list = {}
    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            tmp = line.strip().split("\t")
            txt_list[tmp[position_a]] = tmp[position_b]
    return txt_list


def get_dict_k(filename, position_a, k):
    txt_list = {}
    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            tmp = line.strip().split("\t")
            txt_list[tmp[position_a]] = k
    return txt_list






def random_choose(filename, num):
    if filename.split(".")[-1] == "json":
        json_file_path = filename
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        sample_lines_num = num
        random_items = random.sample(data, sample_lines_num)

        res = json_file_path.split(".")[0]+"_"+str(sample_lines_num)+".json"
        
        with open(res, 'w', encoding='utf-8') as file: # 将随机选中的项写回到新的JSON文件
            json.dump(random_items, file, ensure_ascii=False, indent=4)

        print(f"已成功抽取{num}项，并保存到{res}")
    elif filename.split(".")[-1] == "csv":
        n = num  

        # 定义文件路径
        input_file_path = filename  # 你的CSV文件路径
        output_file_path = input_file_path.split('.')[0]+ '_'+str(n)+'.csv'  # 输出文件路径

        # 读取CSV文件并随机抽取n行
        with open(input_file_path, mode='r', newline='', encoding='utf-8') as csvfile:
            csvreader = csv.reader(csvfile)
            first_row = next(csvreader) # 读取第一行并存储
            remaining_rows = list(csvreader) # 读取剩余的行并存储在一个列表中
            
            # 随机抽取n行
            selected_rows = random.sample(remaining_rows, n) if len(remaining_rows) >= n else random.sample(remaining_rows, len(remaining_rows))

        # 将第一行和随机抽取的行写入新文件
        with open(output_file_path, mode='w', newline='', encoding='utf-8') as csvfile:
            csvwriter = csv.writer(csvfile)
            
            # 写入第一行
            csvwriter.writerow(first_row)
            
            # 写入随机抽取的行
            csvwriter.writerows(selected_rows)

        print(f"已成功抽取{len(selected_rows)}行，并保存到{output_file_path}")
    else:
        print("文件不支持")


def calculate_confusion_counts(ans, label, pos_label=1, neg_label=-1):
    tp = tn = fp = fn = 0


    for pred, true in zip(ans, label):
        if pred == pos_label and true == pos_label:
            tp += 1  # True Positive
        elif pred == neg_label and true == neg_label:
            tn += 1  # True Negative
        elif pred == pos_label and true == neg_label:
            fp += 1  # False Positive
        elif pred == neg_label and true == pos_label:
            fn += 1  # False Negative

    return tp, tn, fp, fn




def calculate_multiclass_confusion_counts(predictions, labels, classes):
    counts = {cls: {"TP": 0, "TN": 0, "FP": 0, "FN": 0} for cls in classes}

    for pred, true in zip(predictions, labels):
        for cls in classes:
            if pred == cls and true == cls:
                counts[cls]["TP"] += 1  # True Positive
            elif pred == cls and true != cls:
                counts[cls]["FP"] += 1  # False Positive
            elif pred != cls and true == cls:
                counts[cls]["FN"] += 1  # False Negative
            elif pred != cls and true != cls:
                counts[cls]["TN"] += 1  # True Negative

    return counts