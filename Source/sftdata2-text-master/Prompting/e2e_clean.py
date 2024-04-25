"Generate a natural language description for the given source."
import os
from pathlib import Path
import linecache
import subprocess


def get_char_lens(data_file):
    return [len(x) for x in Path(data_file).open().readlines()]

def generate_prompt(source_file, target_file, save_path, mode="train", max_num=None):
    """
        ref: references/ControlPrefixes/src/datatotext/utils.py

    :param source_file:
    :param target_file:
    :param save_path:
    :param mode:
    :return:
    """
    sentence_len_list = get_char_lens(source_file)
    data_x_prompt = []
    data_y_prompt = []
    source_prompt = "Generate a text for the given data: {}.  What is the generated text?"
    target_prompt = "The generated text is : {}"
    for index in range(1, len(sentence_len_list)+1):   # 一定要注意getline从1开始计数，line 0 是空的
        if max_num is not None and index > max_num:
            break
        source_line = source_prompt.format(linecache.getline(str(source_file), index).rstrip("\n"))
        tgt_line = target_prompt.format(linecache.getline(str(target_file), index).rstrip("\n"))

        data_x_prompt.append(source_line)
        data_y_prompt.append(tgt_line)
    if max_num is None:
        with open(os.path.join(save_path, mode + "_x_prompt.txt"), "w") as f:
            for i in data_x_prompt:
                f.write(i + "\n")
            f.close()

        with open(os.path.join(save_path, mode + "_y_prompt.txt"), "w") as f:
            for i in data_y_prompt:
                f.write(i + "\n")
            f.close()
    else:
        with open(os.path.join(save_path, mode + f"_{max_num}_x_prompt.txt"), "w") as f:
            for i in data_x_prompt:
                f.write(i + "\n")
            f.close()

        with open(os.path.join(save_path, mode + f"_{max_num}_y_prompt.txt"), "w") as f:
            for i in data_y_prompt:
                f.write(i + "\n")
            f.close()


if __name__ == "__main__":
    save_path = "/home/sdb/xx/path/LLM/SFTData2Text/Dataset/e2e_cleanPrompt"
    data_path = "/home/sdb/xx/path/LLM/references/ControlPrefixes/src/data/e2e_clean"

    for mode in ["train", "test", "val",]:
        if mode in ["test", ]:
            for idx in ["", 2, 3]:
                source_file = os.path.join(data_path, f"{mode}.source")
                target_file = os.path.join(data_path, f"{mode}.target{idx}")
                # generate_prompt(source_file, target_file, save_path, mode=mode, max_num=1000)
                generate_prompt(source_file, target_file, save_path, mode=mode, max_num=None)
                source_path = os.path.join(save_path, f"{mode}_y_prompt.txt")
                if idx == "":
                    rename_path = os.path.join(save_path, f"{mode}_y_prompt0.txt")
                else:
                    rename_path = os.path.join(save_path, f"{mode}_y_prompt{idx}.txt")
                subprocess.run(["cp", source_path, rename_path])
        else:
            source_file = os.path.join(data_path, f"{mode}.source")
            target_file = os.path.join(data_path, f"{mode}.target")
            # generate_prompt(source_file, target_file, save_path, mode=mode, max_num=1000)
            generate_prompt(source_file, target_file, save_path, mode=mode, max_num=None)

    print("Success!")
