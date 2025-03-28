import sys
sys.path.append('..')
sys.path.append('../..')
import json
import random
from tqdm import tqdm
from util_func import *
import argparse

parser = argparse.ArgumentParser(description="instruction files")

parser.add_argument("--head_file", type=str, required=True, help="Path to the head predictions file")
parser.add_argument("--tail_file", type=str, required=True, help="Path to the tail predictions file")
parser.add_argument("--top_k", type=int, default=10, help="Number of top predictions to consider (default: 10)")
parser.add_argument("--output_file", type=str, required=True, help="Path to the output file")

args = parser.parse_args()

head_file = args.head_file
tail_file = args.tail_file
top_k = args.top_k
output_file = args.output_file

ent2txt = get_txt("entity2text_capital.txt", 0, 1)
rel2txt = get_txt("relation2text.txt", 0, 1)
ent2neighbor = get_txt("entities_neighbors_full_en.csv", 0, 1)

test_files = [head_file, tail_file]

for test_file in test_files:
    mode = "head" if test_file.find("head") != -1 else "tail"

    test_lines_to_write = []

    with open(test_file, "r") as f:
        lines = f.readlines()
        for line in tqdm(lines):
            tmp = line.strip().split("\t")
            ent0 = tmp[0]
            rel = tmp[1]
            ent2 = tmp[2]
            keep_list = tmp[3:3+top_k]
            options_list = [ent2txt[x] for x in keep_list]
            
            if mode == "head":
                label = ent2txt[ent0]
                try:
                    ent2_neighbor = ent2txt[ent2] + " has the following attributions: [" + ent2neighbor[ent2] + "]. "
                except Exception as e:
                    ent2_neighbor = ''

                possible_ent_info = []
                for x in keep_list:
                    try:
                        possible_ent_info.append(ent2neighbor[x])
                    except Exception as e:
                        pass
                
                prompt = "The " + rel2txt[rel] + " of What/Who/When/Where/Why is " +  ent2txt[ent2] + "?"
                instruction = 'Below describes an incomplete triple, you need to select one from the candidate entities as the head entity. Now answer the quesion: ' + prompt + ' Please choose the head entity from: [' + ', '.join(options_list) + ' ]. You can make your choice based on the fowllowing information: ' + ent2_neighbor + ', '.join(possible_ent_info)

            elif mode == "tail":
                label = ent2txt[ent2]

                try:
                    ent0_neighbor = ent2txt[ent0] + " has the following attributions: [" + ent2neighbor[ent0] + "]. "
                except Exception as e:
                    ent0_neighbor = ''
                
                possible_ent_info = []
                for x in keep_list:
                    try:
                        possible_ent_info.append(ent2neighbor[x])
                    except Exception as e:
                        pass
                
                prompt = "What/Who/When/Where/Why is the " + rel2txt[rel] + " of " + ent2txt[ent0] + " ?"
                instruction = 'Below describes an incomplete triple, you need to select one from the candidate entities as the head entity. Now answer the quesion: ' + prompt + ' Please choose the head entity from: [' + ', '.join(options_list) + ' ]. You can make your choice based on the fowllowing information: ' + ent0_neighbor + ', '.join(possible_ent_info)

            else:
                print("no such mode")

            tmp_str = instruction + '\t' + label + '\n'
            test_lines_to_write.append(tmp_str)

    with open(test_file.replace('.txt', '_top{}.csv'.format(str(top_k))), 'w') as f:
        # print(test_file)
        # print(test_lines_to_write[-1])
        f.writelines(test_lines_to_write)