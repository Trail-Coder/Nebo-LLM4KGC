def get_txt(filename, position_a, position_b):
    # 创建一个字典，使用 position_a 列的值作为键，position_b 列的值作为值。
    txt_list = {}
    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            tmp = line.strip().split("\t")
            txt_list[tmp[position_a]] = tmp[position_b]
    return txt_list

lines_to_write = []
rel2txt = get_txt("relation2text.txt", 0,1)
ent2txt = get_txt("entity2text_capital.txt", 0, 1)

with open("train.tsv","r",encoding="utf-8") as f:
    lines = f.readlines()
    for line in lines:
        tmp = line.strip().split("\t")
        lines_to_write.append(tmp[0]+"\t"+rel2txt[tmp[1]]+"\t"+ent2txt[tmp[2]]+"\n")

with open("train_en_short.tsv","w",encoding="utf-8") as ff:
    ff.writelines(lines_to_write)