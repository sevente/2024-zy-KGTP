#!/usr/bin/env python3

import sys
import argparse
import json
import logging

import os
from pathlib import Path
import linecache
import subprocess
import json
import csv


def get_char_lens(data_file):
    return [len(x) for x in Path(data_file).open().readlines()]


def generate_json(source_file, target_file, save_path, mode="train", max_num=None):
    """

    :param source_file:
    :param target_file:
    :param save_path:
    :param mode:
    :return:
    """
    sentence_len_list = get_char_lens(source_file)
    output = {"data": []}
    for index in range(1, len(sentence_len_list)+1):  # 一定要注意getline从1开始计数，line 0 是空的
        input_line = linecache.getline(str(source_file), index).rstrip("\n")
        triplets = input_line.replace("[SEP]", "▸")     # 将所有的triples通过 ▸ 拼接到一起
        output_line = linecache.getline(str(target_file), index).rstrip("\n")
        if mode != 'train':     # 只有测试集和验证集存在多个target
            output_line = eval(output_line)  # 将line转换为list, 例如： "['a', 'b']"  转换为 ['a', 'b']
            # 构造ref 和 triple数据对（一个sample可能有多个triple， test数据集有多个target即ref
            ref = output_line[0]    # 对于test / dev仅保留一个example
            example = {"in": triplets, "out": ref}
            output["data"].append(example)
        else:
            ref = output_line
            example = {"in": triplets, "out": ref}
            output["data"].append(example)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    mode = mode if mode != "valid" else "dev"
    with open(os.path.join(save_path, f"{mode}.json"), "w") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)


def generate_target_files(source_file, target_file, save_path, mode="train", max_num=None):
    sentence_len_list = get_char_lens(source_file)

    refs_list = [[], [], []]
    for index in range(1, len(sentence_len_list)+1):  # 一定要注意getline从1开始计数，line 0 是空的
        output_line = linecache.getline(str(target_file), index).rstrip("\n")

        if mode != 'train':  # 只有测试集和验证集存在多个target
            refs = eval(output_line)  # 将line转换为list, 例如： "['a', 'b']"  转换为 ['a', 'b']
            for i in range(3):
                try:
                    ref = refs[i]
                except IndexError:
                    ref = ""
                refs_list[i].append(ref)
        else:
            ref = output_line
            refs_list[0].append(ref)
            refs_list[1].append("")
            refs_list[2].append("")

    mode = mode if mode != "valid" else "dev"
    for i in range(3):
        with open(os.path.join(save_path, f"{mode}.target{i}"), "w") as f:
            f.writelines([x + "\n" for x in refs_list[i]])
            f.close()


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--dataset",
    #     type=str,
    #     required=True,
    #     help="Name of the dataset to be loaded, refers to the class attribute `name` of the class in `data.py`",
    # )
    # parser.add_argument(
    #     "--mode",
    #     choices=["plain", "linearize_triples", "linearize_triples_align"],
    #     required=True,
    #     help="Preprocessing mode",
    # )
    # parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    #
    # args = parser.parse_args()
    # random.seed(args.seed)
    # logger.info(args)

    for dataset_name in ["webnlg", "webnlg2", "e2e", "dart"]:
        save_path = f"/home/sdb/xx/path/LLM/references/critic-aware-decoding-new/mvp_dataset/{dataset_name}_for_decode"
        data_path = f"/home/sdb/xx/path/datasets/HFdatasets/RUCAIBox/Data-to-text-Generation/{dataset_name}"
        mode_list = ["train", "valid", "test", ]

        for mode in mode_list:
            source_file = os.path.join(data_path, f"{mode}.src")
            target_file = os.path.join(data_path, f"{mode}.tgt")
            generate_json(source_file, target_file, save_path, mode=mode, max_num=None)
            if mode != "train":
                generate_target_files(source_file, target_file, save_path, mode=mode)
    print("Success!")
