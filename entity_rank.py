import re
import numpy as np
import argparse
from util_func import *

parser = argparse.ArgumentParser(description="Evaluation Script for FB13")
parser.add_argument("--dictionary", type=str, required=True, help="Path to the entity2textshort dictionary file. Try ../data/FB13/entity2textshort.txt")
parser.add_argument("--stage_1_output_head", type=str, required=True, help="Path to the Stage 1 output for head entities")
parser.add_argument("--stage_2_output_head", type=str, required=True, help="Path to the Stage 2 output for head entities")
parser.add_argument("--stage_1_output_tail", type=str, required=True, help="Path to the Stage 1 output for tail entities")
parser.add_argument("--stage_2_output_tail", type=str, required=True, help="Path to the Stage 2 output for tail entities")
parser.add_argument("--result_file", type=str, required=True, help="Path to the store results")
args = parser.parse_args()

dictionary = args.dictionary
Stage_1_output1 = args.stage_1_output_head
Stage_2_output1 = args.stage_2_output_head
Stage_1_output2 = args.stage_1_output_tail
Stage_2_output2 = args.stage_2_output_tail
result_file = args.result_file


def get_hits(s,n):
    return sum([1 for x in s if x <= n])/len(s)
def get_mrr(s):
    rr = []
    for r in s:
        rr.append(1.0/r)
    return sum(rr)/len(s)
def print_ans(l):
    print(len(l), l)
    print('Hits@1',get_hits(l, 1))
    print('Hits@3',get_hits(l, 3))
    print('Hits@10',get_hits(l, 10))
    print('Hits@20',get_hits(l, 20))
    print('MR:', np.mean(l))
    print('MRR',get_mrr(l))
def count(out1, out2, dic, result_file):
    rank = []
    pre = []
    outputs = []
    txt2ent = get_txt(dic, 1, 0)
    label_idx = 0 if out2.find('head') != -1 else 2

    with open(out2, 'r') as f1:
        ls = f1.readlines()
        for l in ls:
            tmp = l.strip().split('\t')
            if len(tmp) < 3:
                outputs.append('')
                # continue
            else:
                outputs.append(txt2ent[tmp[-2]])

    with open(out1, 'r') as f2:
        ans_lines = f2.readlines()
        for ans_line, output in zip(ans_lines, outputs):
            tmp = ans_line.strip().split('\t')
            tmp.insert(3, output)
            final_line = []
            for x in tmp:
                final_line.append(x.lower())
            pre.append(final_line)
        # print(len(ans_lines), len(outputs), len(pre))

    with open(result_file, 'a+') as t:
        t.writelines('\n'.join(map(str, pre)))


    for p in pre:
        label = p[label_idx]
        check = p[3:]
        idx = check.index(label)
        rank.append(idx+1)

    raw_rank = []
    with open(out1, 'r') as f3:
        ans_lines = f3.readlines()
        for ans_line in ans_lines:
            tmp = ans_line.strip().split('\t')
            raw_rank.append(1+tmp[3:].index(tmp[label_idx]))

    # print('# =================\n',out1.split('/')[-1])
    # print('---------RAW---------')
    # print_ans(raw_rank)
    # print('---------After LLM---------')
    # print_ans(rank)

    return raw_rank, rank


raw_rank1, rank1 = count(Stage_1_output1, Stage_2_output1, dictionary, result_file=result_file)
raw_rank2, rank2 = count(Stage_1_output2, Stage_2_output2, dictionary, result_file=result_file)


print('=====================')
print('Raw:')
print_ans(raw_rank1+raw_rank2)
print('-----\nLLM:')
print_ans(rank1 + rank2)
