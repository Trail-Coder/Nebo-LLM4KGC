import sys
sys.path.append('..')
sys.path.append('../..')

import json
import numpy as np
from util_func import *
import argparse

parser = argparse.ArgumentParser(description="instruction files")
parser.add_argument("--output_file", type=str, required=True, help="Path to the output file")
args = parser.parse_args()
output_file = args.output_file

def is_json(myjson):
    try:
        json.loads(myjson)
    except ValueError:
        return False
    return True

ent2txt = get_txt("data/YAGO3-10/entity2text.txt", 0, 1)
rel2txt = get_txt("data/YAGO3-10/relation2text.txt",0,1)
ent_neighbor = get_txt("data/YAGO3-10/entities_neighbors_en.csv",0,1)

ent_list = []
for ent in ent2txt:
    ent_list.append(ent)

rel_lines_to_write_llama_lora_with_neighbor = []

with open("data/YAGO3-10/train.tsv", "r", encoding = "utf-8") as f:
    lines = f.readlines()
    for line in lines:
        tmp = line.strip().split("\t")
        ent_0 = ent2txt[tmp[0]]
        rel = rel2txt[tmp[1]]
        ent_2 = ent2txt[tmp[2]]


        try:
            neighbor_0 = ent_neighbor[tmp[0]]
            neighbor_2 = ent_neighbor[tmp[2]]
            neighbor_0_prompt = ent_0 + " is related to the following entities:[ " + neighbor_0 + " ]. "
            neighbor_2_prompt = ent_2 + " is related to the following entities:[ " + neighbor_2 + " ]. "
        except Exception:
            neighbor_0_prompt = ''
            neighbor_2_prompt = ''

        prompt = neighbor_0_prompt + neighbor_2_prompt + "Now answer the question:" + "What is the relationship between" + " " + ent_0 + " and " +  ent_2 + "?"
        options = "|".join([rel2txt[key] for key in rel2txt])
        easy_prompt = prompt + " Please choose your answer from: " + options + "."
        
        tmp_str = "{\n\"instruction\": \"" + easy_prompt + "\",\n  \"input\": \"\",\n  \"output\": \""+ rel + "\"\n}"
        rel_lines_to_write_llama_lora_with_neighbor.append(tmp_str)


with open(output_file, "w", encoding = "utf-8") as f:
    tmp_str = "[\n" + ",\n".join(rel_lines_to_write_llama_lora_with_neighbor) +"]"
    f.write(tmp_str)
