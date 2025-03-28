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
full_lines_to_write = []
labels = []


ent2neighbor = get_txt("entities_neighbors_full_en.csv",0,1)
# ent2fulltxt = get_txt("entity2text_capital.txt", 0, 1)
rel2txt = get_txt("relation2text.txt", 0,1)
ent2txt = get_txt("entity2text_capital.txt", 0, 1)

with open("test.tsv", "r", encoding="utf-8") as f:
    lines = f.readlines()

    for line in lines:
        tmp = line.strip().split("\t")

        ent0 = tmp[0]
        rel = tmp[1]
        ent2 = tmp[2]
        label = tmp[3]
        

        try:
            ent0_neighbor = ent2txt[ent0] + " has the following attribution: [" + ent2neighbor[ent0] + "]. "        
        except Exception as e:
            ent0_neighbor = ''
        try:
            ent2_neighbor = ent2txt[ent2] + " has the following attribution: [" + ent2neighbor[ent2] + "]. "
        except Exception as e:
            ent2_neighbor = ''

        prompt = ent0_neighbor + ent2_neighbor + "Now answer the question: Is the following assert true? The {} of {} is {}.".format(rel2txt[rel], ent2txt[ent0], ent2txt[ent2])
        tmp_str = prompt + "\t" + label + "\n"
        lines_to_write.append(tmp_str)

with open(output_file, "w", encoding="utf-8") as ff:
    # ff.write("prompt\tresponse\n")
    ff.writelines(lines_to_write)