import os
import sys

from tqdm import tqdm
import argparse
import torch
from torch.utils.data import DataLoader
import transformers
from peft import PeftModel
from transformers import (
    GenerationConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
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
    model_name: str = "mvp-data-to-text",
    base_model: str = "",
    test_file: str = "",
    save_path: str = "",
):
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

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
    if model_name == "mvp":
        tokenizer = AutoTokenizer.from_pretrained("RUCAIBox/mvp", cache_dir="/home/sdb/xx/path/modelZooHuggingFace/cache")
        # if device == "cuda":
        model = AutoModelForSeq2SeqLM.from_pretrained(
            base_model,
            cache_dir="/home/sdb/xx/path/modelZooHuggingFace/cache",
            # torch_dtype=torch.float16,
            # device_map="auto",
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

    # ---- below is from original tokenizer -----
    # bos_token_id: 0
    # eos_token_id: 2
    # unk_token_id: 3
    # sep_token_id: 2
    # pad_token_id: 1

    print("bos_token_id: ", tokenizer.bos_token_id)
    print("eos_token_id: ", tokenizer.eos_token_id)
    print("unk_token_id: ", tokenizer.unk_token_id)
    print("sep_token_id: ", tokenizer.sep_token_id)
    print("pad_token_id: ", tokenizer.pad_token_id)

    # ref: https://discuss.huggingface.co/t/llama2-pad-token-for-batched-inference/48020/3
    tokenizer.pad_token = tokenizer.bos_token
    model.config.pad_token_id = model.config.bos_token_id
    tokenizer.padding_side = "left"  # This is important for causal LLM in inference

    # if not load_8bit:
    #     model.half()  # seems to fix bugs for some users.

    model.eval()

    # if torch.__version__ >= "2" and sys.platform != "win32":
    #     model = torch.compile(model)

    """
        如果不需要评估测试数据集，那么comment以下代码段即可
    """
    # if test_file.endswith(".json") or test_file.endswith(".jsonl"):
    train_dataset = load_dataset("json", data_files=test_file)["train"]
    print(train_dataset)

    def preprocess_function(data_point):
        instruction = data_point["instruction"]
        input_prefix = "Describe the following data: "
        input = data_point["input"]
        input = input.replace("<H>", "|").replace("<T>", "|").replace("<R>", "|")[1:]  # 去除开始的 "|"
        input = input_prefix + input
        output = data_point["output"]
        # print(input)
        # print(len(input))
        padding = "max_length"
        max_source_length = 1024
        return_attention_mask = False
        # model_inputs = tokenizer(input, max_length=max_source_length, padding=padding, truncation=True, return_attention_mask=False)
        model_inputs = tokenizer(input)
        tokenized_output = tokenizer(output, max_length=max_source_length, padding=padding, truncation=True)
        model_inputs["labels"] = tokenized_output["input_ids"]
        return model_inputs

    columns = list(train_dataset.features.keys())
    print("dataset features columns: ", columns)
    train_dataset = train_dataset.map(
        preprocess_function,
        batched=False,
        num_proc=1,
        remove_columns=columns,
        # load_from_cache_file=not data_args.overwrite_cache,
        desc="Running tokenizer on train dataset",
    )

    print("train_dataset after preprocess", train_dataset, "length: ", len(train_dataset))

    # Data collator
    label_pad_token_id = -100
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        # model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8,
        padding=True,
        return_tensors='pt'
    )

    dataloader_params = {
        "batch_size": 12,
        "collate_fn": data_collator,
        "num_workers": 1,
        "pin_memory": True,
        "drop_last": False
    }

    train_dataloader = DataLoader(train_dataset, **dataloader_params)

    temperature = 0.1
    top_p = 0.75
    top_k = 40
    num_beams = 4
    max_new_tokens = 128
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
    )

    preds_list = []
    eval_tqdm = tqdm(train_dataloader, desc="generating", dynamic_ncols=True)
    for i, batch_data in enumerate(eval_tqdm):
        # print(batch_data)
        input_ids = batch_data["input_ids"]
        # print(input_ids.shape)
        # print(input_ids[0])
        # print(input_ids[1])
        # decoded_inputs = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        # print(decoded_inputs)

        # print(type(input_ids))
        # print(input_ids.shape)
        with torch.no_grad():
            preds = model.generate(
                input_ids=input_ids.to(device),
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                # max_new_tokens=max_new_tokens,
            )
        # print(type(preds)) # transformers.generation.utils.BeamSearchEncoderDecoderOutput
        # print(type(preds.sequences)) # torch.Tensor
        # print(type(preds.sequences[0])) # torch.Tensor
        # print(preds.sequences[0])
        # print(preds.sequences.shape) # batch_size, max_source_length

        preds = preds.sequences
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        print(decoded_preds)

        labels = batch_data["labels"]
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        # print(decoded_labels)

        preds_list.extend(decoded_preds)

    print("len(preds_list): ", len(preds_list))

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(os.path.join(save_path, "predicted.txt"), "w") as f:
        f.writelines([x.replace("\n", "") + "\n" for x in preds_list])
        f.close()


if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default=None, type=str, required=True)
    parser.add_argument('--base_model', default=None, type=str, required=True)
    parser.add_argument('--test_file', default=None, type=str,
                        help='test json file')
    parser.add_argument('--save_path', default=None, type=str,
                        help='path to save predictions/responses for test file')
    args = parser.parse_args()
    main(model_name=args.model_name, base_model=args.base_model, test_file=args.test_file, save_path=args.save_path)
