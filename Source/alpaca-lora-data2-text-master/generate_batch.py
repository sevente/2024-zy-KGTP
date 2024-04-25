import os
import sys

from tqdm import tqdm
import argparse
import torch
from torch.utils.data import DataLoader
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, DataCollatorForSeq2Seq
from datasets import load_dataset

from utils.prompter import Prompter

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
    load_8bit: bool = False,
    base_model: str = "",
    lora_weights: str = "tloen/alpaca-lora-7b",
    prompt_template: str = "",
    turn_on_chat: bool = False,
    test_file: str = "",
    save_path: str = "",
):
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    prompter = Prompter(prompt_template)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
        )
    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )

    # unwind broken decapoda-research config
    # model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    # model.config.bos_token_id = 1
    # model.config.eos_token_id = 2

    # ---- below is from tokenizer -----
    # bos_token_id: 1
    # eos_token_id: 2
    # unk_token_id: 0
    # sep_token_id: None
    # pad_token_id: None

    print("bos_token_id: ", tokenizer.bos_token_id)
    print("eos_token_id: ", tokenizer.eos_token_id)
    print("unk_token_id: ", tokenizer.unk_token_id)
    print("sep_token_id: ", tokenizer.sep_token_id)
    print("pad_token_id: ", tokenizer.pad_token_id)

    # ref: https://discuss.huggingface.co/t/llama2-pad-token-for-batched-inference/48020/3
    tokenizer.pad_token = tokenizer.bos_token
    model.config.pad_token_id = model.config.bos_token_id
    tokenizer.padding_side = "left"  # This is important for causal LLM in inference

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    # if test_file.endswith(".json") or test_file.endswith(".jsonl"):
    train_dataset = load_dataset("json", data_files=test_file)["train"]
    print(train_dataset)

    def preprocess_function(data_point):
        # print(data_point)
        user_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
        )
        cutoff_len = 256
        tokenized_user_prompt = tokenizer(user_prompt,
                                            truncation=True,
                                            max_length=cutoff_len,
                                            padding=False,
                                            return_tensors=None, )  # return pt is False
        # print(tokenized_user_prompt)
        # tokenized_user_prompt = tokenizer(user_prompt, return_tensors="pt", max_length=1024, padding="max_length", truncation=True)
        # tokenized_user_prompt = tokenizer(user_prompt, return_tensors="pt", max_length=1024, padding="max_length")
        # print(len(tokenized_user_prompt["input_ids"]))
        # print(tokenized_user_prompt)
        # tokenized_output = tokenizer(data_point["output"], max_length=1024, padding="max_length", truncation=True)
        # tokenized_output = tokenizer(data_point["output"], return_tensors="pt", padding=True, truncation=True)
        # tokenized_user_prompt["labels"] = tokenized_output["input_ids"]

        return tokenized_user_prompt

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
        model=model,
        padding=True,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8,
        return_tensors="pt"
    )

    dataloader_params = {
        "batch_size": 12,
        "collate_fn": data_collator,
        "num_workers": 1,
        "pin_memory": True,
        "drop_last": False,
        "shuffle": False,
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
        # if i > 4:
        #     break

        # print(batch_data)
        input_ids = batch_data["input_ids"]
        # print(type(input_ids))
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
                max_new_tokens=max_new_tokens,
            )
        # print(type(preds)) # transformers.generation.utils.BeamSearchEncoderDecoderOutput
        # print(type(preds.sequences)) # torch.Tensor
        # print(type(preds.sequences[0])) # torch.Tensor
        # print(preds.sequences[0])
        # print(preds.sequences.shape) # batch_size, max_source_length

        preds = preds.sequences
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # print(decoded_preds)

        # labels = batch_data["labels"] # error IndexError: piece id is out of range.
        # decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        # print(decoded_labels)

        def get_response_batch(responses_list):
            out_list = []
            for response in responses_list:
                out = prompter.get_response(response)
                out_list.append(out)
            return out_list

        decoded_preds = get_response_batch(decoded_preds)

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
    parser.add_argument('--base_model', default=None, type=str, required=True)
    parser.add_argument('--lora_weights', default=None, type=str,
                        help="If None, perform inference on the base model")
    parser.add_argument('--load_8bit', action='store_true',
                        help='only use CPU for inference')
    parser.add_argument('--prompt_template_name', default="alpaca", type=str,
                        help='template fo prompt')
    parser.add_argument('--turn_on_chat', action='store_true',
                        help='Switch to chat mode or just generate predictions for test dataset')
    parser.add_argument('--test_file', default=None, type=str,
                        help='test json file')
    parser.add_argument('--save_path', default=None, type=str,
                        help='path to save predictions/responses for test file')
    args = parser.parse_args()
    main(args.load_8bit, args.base_model, args.lora_weights, args.prompt_template_name,
         args.turn_on_chat, test_file=args.test_file, save_path=args.save_path)
