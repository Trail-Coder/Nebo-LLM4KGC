import sys
sys.path.append('..')
sys.path.append('../..')
import json
import random
from tqdm import tqdm
from util_func import *
import argparse

parser = argparse.ArgumentParser(description="instruction files")
parser.add_argument("--output_file", type=str, required=True, help="Path to the output file")
args = parser.parse_args()
output_file = args.output_file

lines_to_write = []
labels = []


def get_txt(filename, position_a, position_b):
    txt_list = {}
    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            tmp = line.strip().split("\t")
            txt_list[tmp[position_a]] = tmp[position_b]
    return txt_list

ent2neighbor = get_txt("entities_neighbors_en.csv",0,1)
rel2txt = get_txt("relation2text.txt", 0,1)
ent2txt = get_txt("entity2text.txt", 0, 1)

options = "|".join([rel2txt[key] for key in rel2txt])
with open("test.tsv", "r", encoding="utf-8") as f:
    lines = f.readlines()

    for line in lines:
        tmp = line.strip().split("\t")

        ent0 = tmp[0]
        ent2 = tmp[2]
        label = rel2txt[tmp[1]]
        try:
            ent0_neighbor = ent2txt[ent0] + " has relation to the following neighbors: [" + ent2neighbor[ent0] + "]. "
            ent2_neighbor = ent2txt[ent2] + " has relation to the following neighbors: [" + ent2neighbor[ent2] + "]. "
        except Exception as e:
            ent0_neighbor = ''
            ent2_neighbor = ''
        prompt = ent0_neighbor + ent2_neighbor + "Now answer the question: What is the relationship between " + ent2txt[ent0] + " and " +  ent2txt[ent2] + " ? Please choose your answer from: " + options

        tmp_str = prompt + "\t" + label + "\n"
        lines_to_write.append(tmp_str)

with open(output_file, "w", encoding="utf-8") as ff:
    # ff.write("prompt\tresponse\n")
    ff.writelines(lines_to_write)
