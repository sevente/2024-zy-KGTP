"Generate a natural language description for the given source."
import os
from pathlib import Path
import linecache
import subprocess
import json


instruction = "Generate a text based on the following data."

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
        instruction_dict["instruction"] = instruction
        instruction_dict["input"] = input_line
        instruction_dict["output"] = output_line

        instruction_list.append(instruction_dict)
    if max_num is None:
        with open(os.path.join(save_path, mode + ".json"), "w", encoding='utf-8') as f:
            json.dump(instruction_list, f, ensure_ascii=False, indent=2)
    else:
        with open(os.path.join(save_path, mode + f"_{max_num}.json"), "w", encoding='utf-8') as f:
            json.dump(instruction_list, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    for dataset_name in ["webnlg17", "webnlg20", "e2e_clean", "DART"]:
        save_path = f"/home/sdb/xx/path/LLM/alpacaLoraData2Text/Dataset/{dataset_name}Instruct"
        data_path = f"/home/sdb/xx/path/LLM/references/ControlPrefixes/src/data/{dataset_name}"
        if dataset_name in ["webnlg17", "webnlg20"]:
            mode_list = ["train", "val", "test_both", "test_seen", "test_unseen"]  # _both _seen _unseen
        elif dataset_name in ["e2e_clean", ]:
            mode_list = ["train", "val", "test", ]
        elif dataset_name in ["DART", ]:
            mode_list = ["train", "val", "test_both", ]

        for mode in mode_list:
            if mode in ["test_seen", "test_unseen", "test_both", "test"]:
                # for idx in ["", 2, 3]:
                for idx in ["", ]:
                    source_file = os.path.join(data_path, f"{mode}.source")
                    target_file = os.path.join(data_path, f"{mode}.target{idx}")
                    # generate_prompt(source_file, target_file, save_path, mode=mode, max_num=1000)
                    generate_prompt(source_file, target_file, save_path, mode=mode, max_num=None)
                    source_path = os.path.join(save_path, f"{mode}.json")
                    if idx == "":
                        rename_path = os.path.join(save_path, f"{mode}0.json")
                    else:
                        rename_path = os.path.join(save_path, f"{mode}{idx}.json")
                    subprocess.run(["cp", source_path, rename_path])
            else:
                source_file = os.path.join(data_path, f"{mode}.source")
                target_file = os.path.join(data_path, f"{mode}.target")
                # generate_prompt(source_file, target_file, save_path, mode=mode, max_num=1000)
                generate_prompt(source_file, target_file, save_path, mode=mode, max_num=None)

    print("Success!")