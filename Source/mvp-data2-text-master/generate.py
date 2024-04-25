import os
import sys

from tqdm import tqdm
import argparse
import torch
import transformers
from peft import PeftModel
from transformers import (
    AutoConfig,
    GenerationConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    MvpForConditionalGeneration,
)
from datasets import load_dataset

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


def main(
    model_name: str = "",
    base_model: str = "",
    turn_on_chat: bool = False,
    test_file: str = "",
    save_path: str = "",
):
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    # tokenizer = AutoTokenizer.from_pretrained("RUCAIBox/mvp", cache_dir="/home/sdb/xx/path/modelZooHuggingFace/cache")
    # # if device == "cuda":
    # model = AutoModelForSeq2SeqLM.from_pretrained(
    #     base_model,
    #     cache_dir="/home/sdb/xx/path/modelZooHuggingFace/cache",
    #     # torch_dtype=torch.float16,
    #     # device_map="auto",
    #     device_map="cuda",
    # )

    if model_name == "mvp-data-to-text":
        tokenizer = AutoTokenizer.from_pretrained("RUCAIBox/mvp", cache_dir="/home/sdb/xx/path/modelZooHuggingFace/cache")
        # if device == "cuda":
        model = AutoModelForSeq2SeqLM.from_pretrained(
            base_model,
            cache_dir="/home/sdb/xx/path/modelZooHuggingFace/cache",
            # torch_dtype=torch.float16,
            # device_map="auto",
            device_map=device,
        )
    elif model_name == "mtl-data-to-text":
        tokenizer = AutoTokenizer.from_pretrained("RUCAIBox/mvp",
                                                  cache_dir="/home/sdb/xx/path/modelZooHuggingFace/cache")
        # if device == "cuda":
        model = MvpForConditionalGeneration.from_pretrained(
            "/home/sdb/xx/path/modelZooHuggingFace/MVPseries/mtl-data-to-text",
            # cache_dir="/home/sdb/xx/path/modelZooHuggingFace/cache",
            # torch_dtype=torch.float16,
            # device_map="auto",
            device_map=device,
        )
    elif model_name == "mvp":
        tokenizer = AutoTokenizer.from_pretrained("RUCAIBox/mvp", cache_dir="/home/sdb/xx/path/modelZooHuggingFace/cache")
        # if device == "cuda":
        model = AutoModelForSeq2SeqLM.from_pretrained(
            base_model,
            cache_dir="/home/sdb/xx/path/modelZooHuggingFace/cache",
            # torch_dtype=torch.float16,
            # device_map="auto",
            device_map=device,
        )
    elif model_name == "t5-large":
        config = AutoConfig.from_pretrained(base_model)
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            base_model,
            config=config,
            device_map=device,
        )

    # elif device == "mps":
    #     model = LlamaForCausalLM.from_pretrained(
    #         base_model,
    #         device_map={"": device},
    #         torch_dtype=torch.float16,
    #     )
    #     model = PeftModel.from_pretrained(
    #         model,
    #         lora_weights,
    #         device_map={"": device},
    #         torch_dtype=torch.float16,
    #     )
    # else:
    #     model = LlamaForCausalLM.from_pretrained(
    #         base_model, device_map={"": device}, low_cpu_mem_usage=True
    #     )
    #     model = PeftModel.from_pretrained(
    #         model,
    #         lora_weights,
    #         device_map={"": device},
    #     )

    # unwind broken decapoda-research config
    # model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    # model.config.bos_token_id = 1
    # model.config.eos_token_id = 2

    # if not load_8bit:
    #     model.half()  # seems to fix bugs for some users.

    model.eval()
    # if torch.__version__ >= "2" and sys.platform != "win32":
    #     model = torch.compile(model)

    def evaluate(
        input=None,
        # temperature=0.1,
        temperature=1.0,    # 调temperature不影响bleu分数
        top_p=0.75,
        # top_p=0.4,
        top_k=40,
        # top_k=10,
        do_sample=True,
        num_beams=4,
        max_new_tokens=128,
        stream_output=False,
        **kwargs,
    ):
        # input_prefix = "Describe the following data: "
        input_prefix = "Generate a text based on the following data: "
        input = input.replace("<H>", "|").replace("<T>", "|").replace("<R>", "|")[1:]  # 去除开始的 "|"
        input = input_prefix + input

        # print(input)

        inputs = tokenizer(input, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            do_sample=do_sample,
            **kwargs,
        )

        generate_params = {
            "input_ids": input_ids,
            "generation_config": generation_config,
            "return_dict_in_generate": True,
            "output_scores": True,
            "max_new_tokens": max_new_tokens,
        }

        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s, skip_special_tokens=True)
        return output

    """
        如果不需要评估测试数据集，那么comment以下代码段即可
    """
    # if test_file.endswith(".json") or test_file.endswith(".jsonl"):
    dataset = load_dataset("json", data_files=test_file)["train"]
    print(dataset)

    response_list = []
    for idx in tqdm(range(len(dataset))):
        # if idx > 3:
        #     break
        data_dict = dataset[idx]
        # print(data_dict)
        response = evaluate(input=data_dict['input'], )
        print(response)
        response_list.append(response)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(os.path.join(save_path, "predicted.txt"), "w") as f:
        f.writelines([x + "\n" for x in response_list])
        f.close()

    # 再cmd line以对话的形式输入
    if turn_on_chat:
        while True:
            instruction = input("Input:")
            if len(instruction.strip()) == 0:
                break
            print("Response:", evaluate(instruction))


if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default=None, type=str, required=True)
    parser.add_argument('--base_model', default=None, type=str, required=True)
    parser.add_argument('--turn_on_chat', action='store_true',
                        help='Switch to chat mode or just generate predictions for test dataset')
    parser.add_argument('--test_file', default=None, type=str,
                        help='test json file')
    parser.add_argument('--save_path', default=None, type=str,
                        help='path to save predictions/responses for test file')
    args = parser.parse_args()
    main(model_name=args.model_name, base_model=args.base_model,
         turn_on_chat=args.turn_on_chat, test_file=args.test_file, save_path=args.save_path)
