import sys
sys.path.append('..')
sys.path.append('../..')
import pandas as pd
from util_func import *


ent2txt = get_txt("entity2text.txt", 0, 1)

def get_name(index):
    return ent2txt[index]
tsv_file = 'train_en_short.tsv'
csv_file = 'entities_neighbors_full_en.csv'

df = pd.read_csv(tsv_file, sep='\t', header=None, names=['entity', 'relation', "tail"]) 

df['entity'] = df['entity'].astype(str) 

col1 = df['entity']
col2 = df['relation']
col3 = df['tail']

df['combine'] = col1.apply(get_name) +' ' + col2 + ' ' + col3

grouped = df.groupby('entity')['combine'].apply(set)

result_df = grouped.reset_index()
result_df.columns = ['entity', 'neighbors']


def convert_to_string_list(neighbors):
    return [str(item) for item in neighbors]
result_df['neighbors'] = result_df['neighbors'].apply(convert_to_string_list)
result_df['neighbors'] = result_df['neighbors'].apply(lambda x: ', '.join(x))

result_df.to_csv(csv_file, header=True, sep="\t",index=False, encoding='utf-8')

print(f"CSV file '{csv_file}' has been created with entity-neighbor relationships.")
