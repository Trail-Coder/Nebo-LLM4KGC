import sys
sys.path.append('..')
sys.path.append('../..')

import json
import numpy as np
from tqdm import tqdm
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

ent2txt = get_txt("entity2text_capital.txt", 0, 1)
rel2txt = get_txt("relation2text.txt",0,1)
ent2neighbor = get_txt("entities_neighbors_en.csv",0,1)
ent2full_neighbor = get_txt("entities_neighbors_full_en.csv",0,1)

rel_lines_to_write_llama_lora_with_neighbor_full = []


with open("train.tsv", "r", encoding = "utf-8") as f:
    lines = f.readlines()
    for line in tqdm(lines):
        tmp = line.strip().split("\t")
        ent_0 = ent2txt[tmp[0]]
        rel = rel2txt[tmp[1]]
        ent_2 = ent2txt[tmp[2]]

        try:
            neighbor_0 = ent2neighbor[tmp[0]]
            full_neighbor_0 = ent2full_neighbor[tmp[0]]
            full_neighbor_0_prompt = ent_0 + " is related to the following entities:[ " + full_neighbor_0 + " ]. "
           
        except Exception:
            full_neighbor_0_prompt = ''
        
        try:
            neighbor_2 = ent2neighbor[tmp[2]]
            full_neighbor_2_prompt = ent_2 + " is related to the following entities:[ " + ent2full_neighbor[tmp[2]] + " ]. "
        except Exception:
            full_neighbor_2_prompt = ''

        prompt = "What is the relation between " + ent_0 + " and " +  ent_2 + "?"
        options = "|".join([rel2txt[key] for key in rel2txt])
        easy_prompt = prompt + " Please choose your answer from: " + options + "."

        prompt = full_neighbor_0_prompt + full_neighbor_2_prompt + "Now answer the question:" + easy_prompt
        
        tmp_str = "{\n\"instruction\": \"" + prompt + "\",\n  \"input\": \"\",\n  \"output\": \""+ rel + "\"\n}"
        rel_lines_to_write_llama_lora_with_neighbor_full.append(tmp_str)

with open(output_file, "w", encoding = "utf-8") as f:
    tmp_str = "[\n" + ",\n".join(rel_lines_to_write_llama_lora_with_neighbor_full) +"]"
    f.write(tmp_str)
print("done")

