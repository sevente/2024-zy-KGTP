"""
    35分站，测试 data-to-text 任务
"""
import http.client
import os
from pathlib import Path
import linecache
import subprocess
import json
import csv
from tqdm import tqdm

def get_generated_text_from_chatgpt(triplet):
    conn = http.client.HTTPSConnection("35.aigcbest.top")

    chatgpt_sys_message = "You are a helpful assistant that outputs a text just based on the provided single triplet or multiple triplets which is seperated by ▸. "
    # extra_input = "Just generate text based on the provided triplet without producing any additional text. Do not say anything like 'Sorry, but the information provided is not correct.' Triplet: "
    extra_input = "Just generate text based on the provided triplet. Do not say anything like 'Sorry, but the information provided is not correct.' "

    payload = json.dumps({
       "model": "gpt-3.5-turbo",
       "messages": [
            {
                "role": "user",
                "content": extra_input + triplet + "."
                # "content": "Abilene,_Texas | cityServed | Abilene_Regional_Airport"
                # "content": "Aarhus | city | School of Business and Social Sciences at the Aarhus University ▸ European_University_Association | affiliation | School of Business and Social Sciences at the Aarhus University ▸ ""Thomas Pallesen"" | dean | Sch     ool of Business and Social Sciences at the Aarhus University ▸ 16000 | numberOfStudents | School of Business and Social Sciences at the Aarhus University ▸ Denmark | country | School of Business and Social Sciences at the Aarhus University ▸ ""Universitas Aarhu     siensis"" | latinName | School of Business and Social Sciences at the Aarhus University ▸ 1928 | established | School of Business and Social Sciences at the Aarhus University"
            },
            {
                "role": "system",
                "content": chatgpt_sys_message
            },
       ]
    })

    headers = {
       'Accept': 'application/json',
       'Authorization': 'sk-ePNlNfn5tKCzDiC2054e449069C54848B6Ef072d19A58407',
       'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
       'Content-Type': 'application/json'
    }
    conn.request("POST", "/v1/chat/completions", payload, headers)
    res = conn.getresponse()
    data = res.read()
    resp = data.decode("utf-8")
    resp = json.loads(resp)

    # print(resp)
    # print("content: ", resp["choices"][0]['message']['content'])
    return resp["choices"][0]['message']['content']


def get_char_lens(data_file):
    return [len(x) for x in Path(data_file).open().readlines()]


def generate_tex(source_file, target_file, save_path, mode="train", max_num=None):
    """

    :param source_file:
    :param target_file:
    :param save_path:
    :param mode:
    :return:
    """
    sentence_len_list = get_char_lens(source_file)
    output = {"data": []}
    for index in range(1, len(sentence_len_list) + 1):  # 一定要注意getline从1开始计数，line 0 是空的
        input_line = linecache.getline(str(source_file), index).rstrip("\n")
        triplets = input_line.replace("[SEP]", "▸")  # 将所有的triples通过 ▸ 拼接到一起
        output_line = linecache.getline(str(target_file), index).rstrip("\n")
        if mode != 'train':  # 只有测试集和验证集存在多个target
            output_line = eval(output_line)  # 将line转换为list, 例如： "['a', 'b']"  转换为 ['a', 'b']
            # 构造ref 和 triple数据对（一个sample可能有多个triple， test数据集有多个target即ref
            ref = output_line[0]  # 对于test / dev仅保留一个example
            example = {"in": triplets, "out": ref}
            output["data"].append(example)
        else:
            ref = output_line
            example = {"in": triplets, "out": ref}
            output["data"].append(example)

    return output["data"]


if __name__ == "__main__":
    # for dataset_name in ["webnlg", "webnlg2", "e2e", "dart"]:
    for dataset_name in ["dart", ]:
        save_path = f"/home/sdb/xx/path/LLM/ChatgptForD2T/{dataset_name}"
        data_path = f"/home/sdb/xx/path/datasets/HFdatasets/RUCAIBox/Data-to-text-Generation/{dataset_name}"
        # mode_list = ["train", "valid", "test", ]
        mode_list = ["test", ]

        if not os.path.exists(save_path):
            os.mkdir(save_path)

        for mode in mode_list:
            source_file = os.path.join(data_path, f"{mode}.src")
            target_file = os.path.join(data_path, f"{mode}.tgt")
            test_list = generate_tex(source_file, target_file, save_path, mode=mode, max_num=None)
            pred_list = []

            pred_file = os.path.join(save_path, "test")
            if os.path.exists(pred_file):
                pred_list_available = open(pred_file, "r").readlines()
                pred_list.extend(pred_list_available)
                print(pred_list)

            for idx, test_sample in tqdm(enumerate(test_list)):
                if len(pred_list) > idx:
                    continue
                data_triplet = test_sample["in"]
                try:
                    pred_text = get_generated_text_from_chatgpt(data_triplet)
                except:
                    # 解决频繁访问 api 被暂时停止访问的问题
                    import time
                    time.sleep(60)
                    pred_text = get_generated_text_from_chatgpt(data_triplet)
                pred_list.append(pred_text)
                print(f"in: {data_triplet}, pred: {pred_text}")

                with open(os.path.join(save_path, "test"), "a") as f:
                    f.writelines([pred_text.replace("\n", " ") + "\n", ])
                    f.close()

            # with open(os.path.join(save_path, "test"), "w") as f:
            #     f.writelines([x.replace("\n", " ") + "\n" for x in pred_list])
            #     f.close()
