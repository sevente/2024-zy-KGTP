import pandas as pd
import os
import csv
import jsonlines


def clean(x):
    x = x.replace(",", "")
    return x


def to_jsonl(src_file, dst_file, max_num=None):
    all_files = os.listdir(src_file)
    if not os.path.exists(dst_file):
        os.mkdir(dst_file)
    for f in all_files:
        if max_num is None:
            if "valid_x_prompt" in f or "val_x_prompt" in f:
                val_in_list = open(os.path.join(src_file, f)).readlines()
                val_in_list = [line.replace("\n", "") for line in val_in_list]
            elif "valid_y_prompt" in f or "val_y_prompt" in f:
                val_out_list = open(os.path.join(src_file, f)).readlines()
                # val_out_list = [line.replace("\n", ",") for line in val_out_list]
                val_out_list = [line.replace("\n", "") for line in val_out_list]
            elif "train_x_prompt" in f:
                train_in_list = open(os.path.join(src_file, f)).readlines()
                train_in_list = [line.replace("\n", "") for line in train_in_list]
            elif "train_y_prompt" in f:
                train_out_list = open(os.path.join(src_file, f)).readlines()
                train_out_list = [line.replace("\n", "") for line in train_out_list]
        else:
            if f"val_{max_num}_x_prompt" in f:
                val_in_list = open(os.path.join(src_file, f)).readlines()
                val_in_list = [line.replace("\n", "") for line in val_in_list]
            elif f"val_{max_num}_y_prompt" in f:
                val_out_list = open(os.path.join(src_file, f)).readlines()
                # val_out_list = [line.replace("\n", ",") for line in val_out_list]
                val_out_list = [line.replace("\n", "") for line in val_out_list]
            elif f"train_{max_num}_x_prompt" in f:
                train_in_list = open(os.path.join(src_file, f)).readlines()
                train_in_list = [line.replace("\n", "") for line in train_in_list]
            elif f"train_{max_num}_y_prompt" in f:
                train_out_list = open(os.path.join(src_file, f)).readlines()
                train_out_list = [line.replace("\n", "") for line in train_out_list]

    val_items = []
    train_items = []

    for i in range(len(val_in_list)):
        val_items.append({"text": val_in_list[i], "summary": val_out_list[i]})
    for i in range(len(train_in_list)):
        train_items.append({"text": train_in_list[i], "summary": train_out_list[i]})
    with jsonlines.open(os.path.join(dst_file, "val.json"), 'w') as writer:
        writer.write_all(val_items)
    with jsonlines.open(os.path.join(dst_file, "train.json"), 'w') as writer:
        writer.write_all(train_items)

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        '--local_dataset_name', default="webnlg17", type=str,
        help='dataset name')
    args = parser.parse_args()
    dataset_name = args.local_dataset_name

    if dataset_name == "webnlg17":
        to_jsonl("/home/sdb/xx/path/LLM/SFTData2Text/Dataset/webnlg17Prompt",
                 "/home/sdb/xx/path/LLM/SFTData2Text/Benchmark/HuggingFace_Data2Text/webnlg17_JSON", max_num=None)
                 # "/home/sdb/xx/path/LLM/SFTData2Text/Benchmark/HuggingFace_Data2Text/webnlg17_JSON", max_num=1000)
    elif dataset_name == "webnlg20":
        to_jsonl("/home/sdb/xx/path/LLM/SFTData2Text/Dataset/webnlg20Prompt",
                 "/home/sdb/xx/path/LLM/SFTData2Text/Benchmark/HuggingFace_Data2Text/webnlg20_JSON", max_num=None)
        # "/home/sdb/xx/path/LLM/SFTData2Text/Benchmark/HuggingFace_Data2Text/webnlg20_JSON", max_num=1000)
    elif dataset_name == "e2e_clean":
        to_jsonl("/home/sdb/xx/path/LLM/SFTData2Text/Dataset/e2e_cleanPrompt",
                 "/home/sdb/xx/path/LLM/SFTData2Text/Benchmark/HuggingFace_Data2Text/e2e_clean_JSON", max_num=None)
        # "/home/sdb/xx/path/LLM/SFTData2Text/Benchmark/HuggingFace_Data2Text/e2e_clean_JSON", max_num=1000)
    elif dataset_name == "DART":
        to_jsonl("/home/sdb/xx/path/LLM/SFTData2Text/Dataset/DARTPrompt",
                 "/home/sdb/xx/path/LLM/SFTData2Text/Benchmark/HuggingFace_Data2Text/DART_JSON", max_num=None)
        # "/home/sdb/xx/path/LLM/SFTData2Text/Benchmark/HuggingFace_Data2Text/DART_JSON", max_num=1000)
    elif dataset_name == "webnlg":
        to_jsonl("/home/sdb/xx/path/LLM/SFTData2Text/Dataset/webnlg",
                 "/home/sdb/xx/path/LLM/SFTData2Text/Benchmark/HuggingFace_Data2Text/webnlg_JSON", max_num=None)
                 # "/home/sdb/xx/path/LLM/SFTData2Text/Benchmark/HuggingFace_Data2Text/webnlg17_JSON", max_num=1000)
    elif dataset_name == "webnlg2":
        to_jsonl("/home/sdb/xx/path/LLM/SFTData2Text/Dataset/webnlg2",
                 "/home/sdb/xx/path/LLM/SFTData2Text/Benchmark/HuggingFace_Data2Text/webnlg2_JSON", max_num=None)
        # "/home/sdb/xx/path/LLM/SFTData2Text/Benchmark/HuggingFace_Data2Text/webnlg20_JSON", max_num=1000)
    elif dataset_name == "e2e":
        to_jsonl("/home/sdb/xx/path/LLM/SFTData2Text/Dataset/e2e",
                 "/home/sdb/xx/path/LLM/SFTData2Text/Benchmark/HuggingFace_Data2Text/e2e_JSON", max_num=None)
        # "/home/sdb/xx/path/LLM/SFTData2Text/Benchmark/HuggingFace_Data2Text/e2e_clean_JSON", max_num=1000)
    elif dataset_name == "dart":
        to_jsonl("/home/sdb/xx/path/LLM/SFTData2Text/Dataset/dart",
                 "/home/sdb/xx/path/LLM/SFTData2Text/Benchmark/HuggingFace_Data2Text/dart_JSON", max_num=None)
        # "/home/sdb/xx/path/LLM/SFTData2Text/Benchmark/HuggingFace_Data2Text/dart_JSON", max_num=1000)
    else:
        assert "Error"







