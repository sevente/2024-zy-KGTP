# 从https://huggingface.co/datasets/RUCAIBox/Data-to-text-Generation 下载 webnlg v2.1 , webnlg v3.0


"Generate a natural language description for the given source."
import os
from pathlib import Path
import linecache
import subprocess
import json


# instruction = "Generate a text based on the following data."
instruction = "Describe the following data: "

def get_char_lens(data_file):
    return [len(x) for x in Path(data_file).open().readlines()]

def generate_prompt(source_file, target_file, save_path, mode="train", max_num=None):
    """

    :param source_file:
    :param target_file:
    :param save_path:
    :param mode:
    :return:
    """
    sentence_len_list = get_char_lens(source_file)
    instruction_list = []
    for index in range(1, len(sentence_len_list)+1):  # 一定要注意getline从1开始计数，line 0 是空的
        instruction_dict = dict()
        if max_num is not None and index > max_num:
            break
        input_line = linecache.getline(str(source_file), index).rstrip("\n")
        output_line = linecache.getline(str(target_file), index).rstrip("\n")
        if mode != 'train':
            output_line = eval(output_line)  # 将line转换为list, 例如： "['a', 'b']"  转换为 ['a', 'b']
            output_line_first = output_line[0]
        else:
            output_line_first = output_line
        instruction_dict["instruction"] = instruction
        instruction_dict["input"] = input_line
        instruction_dict["output"] = output_line_first

        instruction_list.append(instruction_dict)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if max_num is None:
        with open(os.path.join(save_path, mode + ".json"), "w", encoding='utf-8') as f:
            json.dump(instruction_list, f, ensure_ascii=False, indent=2)
    else:
        with open(os.path.join(save_path, mode + f"_{max_num}.json"), "w", encoding='utf-8') as f:
            json.dump(instruction_list, f, ensure_ascii=False, indent=2)


def generate_target_files(source_file, target_file, save_path, mode="test"):
    """
        将原始的tgt file，切分为多个 target file， 部分source对应多个target。
        原始tgt txt格式, 其实刚开始的数字为line num：
            1 ['There is a place in the city centre, Alimentum, that is not family-friendly.']
            2 ['In the city centre there is a venue name Alimentum, this is not a family-friendly venue.']
            3 ['Alimentum is not a family-friendly place, located in city centre.']
            4 ['Alimentum is not a family-friendly arena and is located in the city centre.']
            5 ['Alimentum is not a family-friendly place in the city centre.']
            6 ['Alimentum in city centre is not a family-friendly place.']
    :param source_file:
    :param target_file:
    :param save_path:
    :return:
    """
    sentence_len_list = get_char_lens(source_file)
    target_list = []
    target2_list = []
    target3_list = []
    for index in range(1, len(sentence_len_list) + 1):  # 一定要注意getline从1开始计数，line 0 是空的
        output_line = linecache.getline(str(target_file), index).rstrip("\n")
        output_line = eval(output_line)  # 将line转换为list, 例如： "['a', 'b']"  转换为 ['a', 'b']
        for i in range(3):
            try:
                target = output_line[i]
            except IndexError:
                target = ""
            if i == 0:
                target_list.append(target)
            elif i == 1:
                target2_list.append(target)
            elif i == 2:
                target3_list.append(target)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for idx, targets in enumerate([target_list, target2_list, target3_list]):
        with open(os.path.join(save_path, f"{mode}.target{idx+1}"), "w") as f:
            f.writelines([x + "\n" for x in targets])
            f.close()

if __name__ == "__main__":
    for dataset_name in ["webnlg", "webnlg2", "e2e", "dart"]:
        save_path = f"/home/sdb/xx/path/LLM/mvpData2Text/Dataset/{dataset_name}"
        data_path = f"/home/sdb/xx/path/datasets/HFdatasets/RUCAIBox/Data-to-text-Generation/{dataset_name}"
        mode_list = ["train", "valid", "test", ]

        for mode in mode_list:
            source_file = os.path.join(data_path, f"{mode}.src")
            target_file = os.path.join(data_path, f"{mode}.tgt")
            generate_prompt(source_file, target_file, save_path, mode=mode, max_num=None)
            if mode != "train":
                generate_target_files(source_file, target_file, save_path, mode=mode)
    print("Success!")
