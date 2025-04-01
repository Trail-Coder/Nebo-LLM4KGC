import sys
sys.path.append('..')
sys.path.append('../..')

import json
import numpy as np
from tqdm import tqdm
from util_func import get_txt, random_choose
import argparse

parser = argparse.ArgumentParser(description="instruction files")
parser.add_argument("--output_file", type=str, required=True, help="Path to the output file")
args = parser.parse_args()
output_file = args.output_file

ent2txt = get_txt("entity2text_capital.txt", 0, 1)
rel2txt = get_txt("relation2text.txt",0,1)

rel_lines_to_write_llama_lora = []

neighbor0_num = 0
neighbor1_num = 0

with open("test.tsv", "r", encoding = "utf-8") as f:
    lines = f.readlines()
    for line in tqdm(lines):
        tmp = line.strip().split("\t")
        ent_0 = ent2txt[tmp[0]]
        rel = rel2txt[tmp[1]]
        ent_2 = ent2txt[tmp[2]]
        label = tmp[3]

        if int(label) == -1:
            continue
      
        prompt = "What is the relation between " + ent_0 + " and " +  ent_2 + "?"
        options = "|".join([rel2txt[key] for key in rel2txt])
        easy_prompt = prompt + " Please choose your answer from: " + options + "."

        tmp_str = easy_prompt + "\t"+ rel + "\n"
        rel_lines_to_write_llama_lora.append(tmp_str)

with open(output_file, "w", encoding = "utf-8") as f:
    f.writelines(rel_lines_to_write_llama_lora)

