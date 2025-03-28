import csv
import random
import json
import safetensors
import safetensors.torch
import torch
# from util_func import *
import pandas as pd



# ======================================================================
# 随机抽取json项
# ======================================================================

# json_file_path_list = ['train_ent_head.json', 'train_ent_tail.json']
# json_file_path_list = ['train_ent_head_full_neighbor.json', 'train_ent_tail_full_neighbor.json']

# for json_file_path in json_file_path_list:
#     with open(json_file_path, 'r', encoding='utf-8') as file:
#         data = json.load(file)

#     # 随机选择n项
#     sample_lines_num = 20000
#     random_items = random.sample(data, sample_lines_num)

#     res = json_file_path.split(".")[0]+"_"+str(sample_lines_num)+".json"


#     # 将随机选中的项写回到新的JSON文件
#     with open(res, 'w', encoding='utf-8') as file:
#         json.dump(random_items, file, ensure_ascii=False, indent=4)

#     print(res, "\nDone")

# ======================================================================
# 随机抽取csv中的项,保留
# ======================================================================

n = 100  # 例如，抽取5行
# csv_file_path = ['test_ent_head.csv', 'test_ent_tail.csv']
csv_file_path = ['test_ent_head.csv', 'test_ent_tail.csv']


for input_file_path in csv_file_path:

    output_file_path = input_file_path.split('.')[0]+ '_'+str(n)+'.csv'  # 输出文件路径


    # 读取CSV文件并随机抽取n行
    with open(input_file_path, mode='r', newline='', encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile)
        
        # 读取第一行并存储
        first_row = next(csvreader)
        
        # 读取剩余的行并存储在一个列表中
        remaining_rows = list(csvreader)
        
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