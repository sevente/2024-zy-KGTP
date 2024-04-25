import json
import os

# returns JSON object as
# a dictionary
with open("./data/webnlg/test.json") as f:
    data = json.load(f)
print(type(data))
print(data.keys())
# print(data['data'])
print(type(data['data']))
print(len(data['data']))


data = data['data']

targets = []
for i in range(len(data)):
    targets.append(data[i]['out'])

save_path = "./data/webnlg/test.target"
with open(os.path.join(save_path), "w") as f:
    f.writelines([x + "\n" for x in targets])
    f.close()
