from decoding.data import WebNLG
import sys
import csv

# 将triple 转换为中间由 | 分隔的str
def linearize_triple(triple):
    X_SEP = " | "
    out = []
    t = triple
    return t.subj + X_SEP + t.pred + X_SEP + t.obj

data = WebNLG()
splits = [sys.argv[1]]
data.load(splits,None)

# 构造ref 和 triple数据对（一个sample可能有多个triple， test数据集有多个target即ref，一个list，将sample和多个triples当作一个sample
cleanedData = []
for split in splits:
    for dataEntry in data.data[split]:
        for ref in dataEntry.refs:
                triplets = [linearize_triple(i) for i in dataEntry.data]
                cleanedData.append((ref,triplets))

# 将所有的triples通过 ▸ 拼接到一起
with open(f"webnlg-{split}.tex", 'w', encoding="UTF-8") as f:
    writer = csv.writer(f)
    for txt,triplets in cleanedData:
        writer.writerow([txt, " ▸ ".join(triplets)])

