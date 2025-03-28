import sys
import argparse
sys.path.append('..')
sys.path.append('../..')
sys.path.append('../../..')
# sys.path.append('../../../..')

import re
import numpy as np
from tqdm import tqdm
from util_func import *

id2ent = get_txt("entities.dict", 0, 1)
id2rel = get_txt("relations.dict", 0, 1)
parser = argparse.ArgumentParser(description="id2txt")
parser.add_argument("--mode", type=str, required=True, choices=["test", "train"], help="Mode of operation"
)
parser.add_argument("--head_file", type=str, required=True, help="Path to the head predictions file"
)
parser.add_argument("--tail_file", type=str, required=True, help="Path to the tail predictions file"
)
parser.add_argument("--head_res_file", type=str, required=True, help="Path to the head predictions result file"
)
parser.add_argument("--tail_res_file", type=str, required=True, help="Path to the tail predictions result file"
)

args = parser.parse_args()

mode = args.mode
head_file = args.head_file
tail_file = args.tail_file
head_res = args.head_res_file
tail_res = args.tail_res_file

test_lines = []

with open(head_file, "r", encoding="utf-8") as f:
    lines = f.readlines()
    for line in tqdm(lines):
        tmp = line.split('\t')

        triple = re.sub(r'\s+', '', tmp[0])[1:-1].split(',')
        pre_id = re.sub(r'\s+', '', tmp[1])[1:-1].split(',')
        # print(len(pre_id))

        head = id2ent[triple[0]]
        rel = id2rel[triple[1]]
        tail = id2ent[triple[2]]

        test_line = '{}\t{}\t{}'.format(head, rel, tail)

        for i in pre_id:
            test_line += '\t{}'.format(id2ent[i])
        
        test_lines.append(test_line+'\n')

with open(head_res,'w') as f:
    f.writelines(test_lines)


test_lines = []

with open(tail_file, "r", encoding="utf-8") as f:
    lines = f.readlines()
    for line in tqdm(lines):
        tmp = line.split('\t')

        triple = re.sub(r'\s+', '', tmp[0])[1:-1].split(',')
        pre_id = re.sub(r'\s+', '', tmp[1])[1:-1].split(',')

        head = id2ent[triple[0]]
        rel = id2rel[triple[1]]
        tail = id2ent[triple[2]]
        test_line = '{}\t{}\t{}'.format(head, rel, tail)

        for i in pre_id:
            test_line += '\t{}'.format(id2ent[i])
        
        test_lines.append(test_line+'\n')

with open(tail_res,'w') as f:
    f.writelines(test_lines)
    # print(test_lines[-1])
