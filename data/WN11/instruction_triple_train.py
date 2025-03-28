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

def is_json(myjson):
    try:
        json.loads(myjson)
    except ValueError:
        return False
    return True

ent2txt = get_txt("entity2text.txt", 0, 1)
rel2txt = get_txt("relation2text.txt", 0, 1)
ent2neighbor = get_txt("entities_neighbors_full_en.csv", 0, 1)

ent_list = []
for ent in ent2txt:
    ent_list.append(ent)


lines_to_write_llama_lora = []
with open("train.tsv", "r") as f:
    lines = f.readlines()
    for line in tqdm(lines):
        tmp = line.strip().split("\t")
        
        ent0 = tmp[0]
        rel = tmp[1]
        ent2 = tmp[2]

        try:
            ent0_neighbor_list = ent2neighbor[ent0].split(", ")
            for x in ent0_neighbor_list:
                if x.find(rel2txt[rel]) != -1:
                    ent0_neighbor_list.remove(x)
            ent0_neighbors_str = ", ".join(ent0_neighbor_list)
            ent0_neighbor = ent2txt[ent0] + " is related to the following entities: [" + ent0_neighbors_str + "]. "
        except Exception as e:
            ent0_neighbor = ''
        try:
            ent2_neighbor = ent2txt[ent2] + " is related to the following entities: [" + ent2neighbor[ent2] + "]. "
        except Exception as e:
            ent2_neighbor = ''

        
        prompt = ent0_neighbor + ent2_neighbor + "Now answer the question: Is the following assert true? " + ent2txt[ent0] + " " + rel2txt[rel] + " " +  ent2txt[ent2] + ". "
        
        tmp_str = "{\n\"instruction\": \"" + prompt + "\",\n  \"input\": \"\",\n  \"output\": \""+ "Yes, this is true."+ "\"\n}"
        lines_to_write_llama_lora.append(tmp_str)
       
        rnd = random.random()

        if rnd <= 0.5:
            # corrupting head
            tmp_ent_list = set(ent_list)
            tmp_ent_list.remove(tmp[0])
            tmp_ent_list = list(tmp_ent_list)
            tmp_head = random.choice(tmp_ent_list)
            
            try:
                tmp_head_neighbor_list = ent2neighbor[ent0].split(", ")
                for x in tmp_head_neighbor_list:
                    if x.find(rel2txt[rel]) != -1:
                        tmp_head_neighbor_list.remove(x)
                tmp_head_neighbor_str = ", ".join(ent0_neighbor_list)
                tmp_head_neighbor = ent2txt[tmp_head] + " is related to the following entities: [" + tmp_head_neighbor_str + "]. "
            except Exception as e:
                tmp_head_neighbor = ''

            prompt = tmp_head_neighbor + ent2_neighbor + "Now answer the question: Is the following assert true? " + ent2txt[tmp_head] + " " + rel2txt[rel] + " " +  ent2txt[ent2] + ". "

            tmp_str = "{\n\"instruction\": \"" + prompt + "\",\n  \"input\": \"\",\n \"output\": \""+ "No, this is not true."+ "\"\n}"
            lines_to_write_llama_lora.append(tmp_str)

        else:
            # corrupting tail
            tmp_ent_list = set(ent_list)
            tmp_ent_list.remove(tmp[2])
            tmp_ent_list = list(tmp_ent_list)
            tmp_tail = random.choice(tmp_ent_list)

            try:
                tmp_tail_neighbor = ent2txt[tmp_tail] + " is related to the following entities: [" + ent2neighbor[tmp_tail] + "]. "
            except Exception as e:
                tmp_tail_neighbor = ''

            prompt = ent0_neighbor + tmp_tail_neighbor + "Now answer the question: Is the following assert true? " + ent2txt[ent0] + " " + rel2txt[rel] + " " +  ent2txt[tmp_tail] + ". "
            
            tmp_str = "{\n\"instruction\": \"" + prompt + "\",\n  \"input\": \"\",\n \"output\": \""+ "No, this is not true."+ "\"\n}"
            lines_to_write_llama_lora.append(tmp_str)

with open(output_file, "w") as f:
    tmp_str = "[\n" + ",\n".join(lines_to_write_llama_lora) +"]"
    f.write(tmp_str)
    # json.dump(lines_to_write_llama_lora, f)