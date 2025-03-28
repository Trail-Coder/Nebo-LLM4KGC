import datetime
import argparse


parser = argparse.ArgumentParser(description="Evaluation Script")

parser.add_argument(
    "--stage_1_output",
    type=str,
    required=True,
    help="Path to Stage 1 output file"
)

args = parser.parse_args()
file_b = args.stage_1_output

dataset = str(file_b.split('_')[1])

file_a = '{}/test.txt'.format(dataset)

file_a_head = '{}/test_head_{}.txt'.format(dataset, str(datetime.date.today()))
file_a_tail = '{}/test_tail_{}.txt'.format(dataset, str(datetime.date.today()))

with open(file_a, 'r') as f_a:
    lines_a = f_a.readlines()

with open(file_b, 'r') as f_b:
    lines_b = f_b.readlines()

assert len(lines_a) * 2 == len(lines_b), "error"


mid_point = len(lines_b) // 2


with open(file_a_head, 'w') as f_a_head:
    for line_a, line_b in zip(lines_a, lines_b[:mid_point]):
        f_a_head.write(line_a.split('\n')[0] +'\t' + line_b)


with open(file_a_tail, 'w') as f_a_tail:
    for line_a, line_b in zip(lines_a, lines_b[mid_point:]):
        f_a_tail.write(line_a.split('\n')[0] +'\t' + line_b)

print(file_a_head, file_a_tail)