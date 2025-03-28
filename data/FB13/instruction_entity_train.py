import sys
sys.path.append('..')
sys.path.append('../..')
import json
import random
from tqdm import tqdm
from util_func import *
import argparse

parser = argparse.ArgumentParser(description="Evaluation Script")

parser.add_argument("--head_file", type=str, required=True, help="Path to the head predictions file")
parser.add_argument("--tail_file", type=str, required=True, help="Path to the tail predictions file")
parser.add_argument("--top_k", type=int, default=10, help="Number of top predictions to consider (default: 10)")

args = parser.parse_args()

head_file = args.head_file
tail_file = args.tail_file
top_k = args.top_k

ent2txt = get_txt("entity2text_capital.txt", 0, 1)
rel2txt = get_txt("relation2text.txt", 0, 1)
ent2neighbor = get_txt("entities_neighbors_full_en.csv", 0, 1)

train_files = [head_file, tail_file]


for train_file in train_files:
    mode = "head" if train_file.find("head") != -1 else "tail"
    ent_list = []
    for ent in ent2txt:
        ent_list.append(ent)
    lines_to_write_llama_lora = []

    with open(train_file, "r") as f:
        lines = f.readlines()
        for line in tqdm(lines):
            tmp = line.strip().split("\t")
            
            ent0 = tmp[0]
            rel = tmp[1]
            ent2 = tmp[2]
            raw_triple = 'The {} of {} is {}'.format(rel2txt[rel], ent2txt[ent0], ent2txt[ent2])

            if mode == "head":

                if ent0 in tmp[3:]:
                    keep_list = tmp[3:]                    
                else:
                    keep_list = tmp[3:-1]
                    keep_list.append(ent0)
                random.shuffle(keep_list)
                options_list = [ent2txt[x] for x in keep_list]
                possible_ent_info = []
                for x in keep_list:
                    try:
                        possible_ent_info.append(ent2neighbor[x])
                    except Exception as e:
                        pass

                
                
                try:
                    ent2_neighbor = ent2txt[ent2] + " has the attributions: [" + ent2neighbor[ent2].replace(raw_triple, '') + "]. "
                except Exception as e:
                    ent2_neighbor = ''
                
                prompt = "The " + rel2txt[rel] + " of What/Who/When/Where/Why is " +  ent2txt[ent2] + "?"
                instruction = 'Below describes an incomplete triple, you need to select one from the candidate entities as the head entity. Now answer the quesion: ' + prompt + ' Please choose the head entity from: [' + ', '.join(options_list) + ' ]. You can make your choice based on the fowllowing information: ' + ent2_neighbor + ', '.join(possible_ent_info)
                
                ans_idx = options_list.index(ent2txt[ent0])
                options_list.pop(ans_idx)
                options_list.insert(0, ent2txt[ent0])
                # print(len(options_list))

                # answer = ent2txt[ent0] + '. [ ' + ', '.join(options_list) + ' ]'
                answer = ent2txt[ent0]

            elif mode == "tail":
                if ent2 in tmp[3:]:
                    keep_list = tmp[3:]                    
                else:
                    keep_list = tmp[3:-1]
                    keep_list.append(ent2)
                random.shuffle(keep_list)
                options_list = [ent2txt[x] for x in keep_list]
                possible_ent_info = []
                for x in keep_list:
                    try:
                        possible_ent_info.append(ent2neighbor[x])
                    except Exception as e:
                        pass
                
                try:
                    ent0_neighbor = ent2txt[ent0] + " has the following attributions: [" + ent2neighbor[ent0].replace(raw_triple, '') + "]. "
                except Exception as e:
                    ent0_neighbor = ''
                prompt =  "What/Who/When/Where/Why is the " + rel2txt[rel] + " of " + ent2txt[ent0] + " ?"
                instruction = 'Below describes an incomplete triple, you need to select one from the candidate entities as the head entity. Now answer the quesion: ' + prompt + ' Please choose the head entity from: [' + ', '.join(options_list) + ' ]. You can make your choice based on the fowllowing information: ' + ent0_neighbor + ', '.join(possible_ent_info)
                
                ans_idx = options_list.index(ent2txt[ent2])
                options_list.pop(ans_idx)
                options_list.insert(0, ent2txt[ent2])
                # print(len(options_list))

                # answer = ent2txt[ent2] + '. [ ' + ', '.join(options_list) + ' ]'
                answer = ent2txt[ent2]

            else:
                print("no such mode")

            tmp_str = "{\n\"instruction\": \"" + instruction + "\",\n  \"input\": \"\",\n  \"output\": \""+ answer+ "\"\n},\n"
            lines_to_write_llama_lora.append(tmp_str)

    with open('train_ent_'+mode+'_full_neighbor.json', 'w') as f:
        last = lines_to_write_llama_lora.pop(-1)
        lines_to_write_llama_lora.append(last[:-2])
        f.write("[\n")
        f.writelines(lines_to_write_llama_lora)
        f.write(']')