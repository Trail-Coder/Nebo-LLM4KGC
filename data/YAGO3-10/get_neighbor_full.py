import pandas as pd

def get_txt(filename, position_a, position_b):
    txt_list = {}
    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            tmp = line.strip().split("\t")
            txt_list[tmp[position_a]] = tmp[position_b]
    return txt_list
ent2txt = get_txt("entity2text.txt", 0, 1)

def get_name(index):
    return ent2txt[index]

tsv_file = 'train_en.tsv'
csv_file = 'entities_neighbors_full_en.csv'

df = pd.read_csv(tsv_file, sep='\t', header=None, names=['entity', 'relation', "tail"]) 

df['entity'] = df['entity'].astype(str)

col1 = df['entity']
col2 = df['relation']
col3 = df['tail']

df['combine'] = col1.apply(get_name) +' '+col2+' '+col3

grouped = df.groupby('entity')['combine'].apply(set)

result_df = grouped.reset_index()
result_df.columns = ['entity', 'neighbors']

result_df['neighbors'] = result_df['neighbors'].apply(lambda x: ', '.join(x))

result_df.to_csv(csv_file, header=True, sep="\t",index=False, encoding='utf-8')

print(f"CSV file '{csv_file}' has been created with entity-neighbor relationships.")

