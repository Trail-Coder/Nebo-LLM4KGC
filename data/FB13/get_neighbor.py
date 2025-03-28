import pandas as pd

tsv_file = 'train_en_short.tsv'
csv_file = 'entities_neighbors_en.csv'

def convert_to_string_list(neighbors):
    return [str(item) for item in neighbors]

df = pd.read_csv(tsv_file, sep='\t', header=None, names=['entity', 'relation', "tail"])
df['entity'] = df['entity'].astype(str)
grouped = df.groupby('entity')['tail'].apply(set)
result_df = grouped.reset_index()
result_df.columns = ['entity', 'neighbors']
result_df['neighbors'] = result_df['neighbors'].apply(convert_to_string_list)
result_df['neighbors'] = result_df['neighbors'].apply(lambda x: ', '.join(x))
result_df.to_csv(csv_file, header=True, sep="\t",index=False, encoding='utf-8')

print(f"CSV file '{csv_file}' has been created with entity-neighbor relationships.")