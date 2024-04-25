import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import re
from datasets import load_metric
from nltk.tokenize import RegexpTokenizer
import os


def get_regexp_tokens(input_file):
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = []

    for sent in input_file:
        # print(sent)
        tokenized_sent = tokenizer.tokenize(sent)
        tokens.append(tokenized_sent)

    return tokens


def get_reference_tokens(input_file_list):
    """
        在有多个reference输入文件的情况下, 例如：
        input_file_list: [
                            ["The runway length of Aarhus Airport is 2702.0.", "The leader of Aarhus is Jacob Bundsgaard.",]
                            ["Aarhus Airport's runway length is 2702.0.", "Aarhus's leader is Jacob Bundsgaard.",]
                            ]
        表示每个prediction存在两个相对应的target可以用于bleu score的计算，那么需要对他们进行拼接
        Examples:

        >>> predictions = [
        ...     ["hello", "there", "general", "kenobi"],                             # tokenized prediction of the first sample
        ...     ["foo", "bar", "foobar"]                                             # tokenized prediction of the second sample
        ... ]
        >>> references = [
        ...     [["hello", "there", "general", "kenobi"], ["hello", "there", "!"]],  # tokenized references for the first sample (2 references)
        ...     [["foo", "bar", "foobar"]]                                           # tokenized references for the second sample (1 reference)
        ... ]
        >>> bleu = datasets.load_metric("bleu")
        >>> results = bleu.compute(predictions=predictions, references=references)
        >>> print(results["bleu"])

    :param input_file_list:
    :return:
    """
    input_token_list = []
    for input_file in input_file_list:
        input_token = get_regexp_tokens(input_file)
        input_token_list.append(input_token)

    input_file_num = len(input_token_list)
    input_file_sentences_num_list = []
    for i in range(input_file_num):
        input_file_sentences_num_list.append(len(input_token_list[i]))

    # 保证不同target文件输入大小一致
    assert len(set(input_file_sentences_num_list)) == 1

    concat_reference_tokens = []
    # 将不同target的sentence拼接到一起
    for i in range(input_file_sentences_num_list[0]):
        reference_tokens = []
        for j in range(input_file_num):
            ref_token = input_token_list[j][i]
            if ref_token == []:
                continue
            reference_tokens.append(ref_token)
        concat_reference_tokens.append(reference_tokens)

    return concat_reference_tokens


def metric_with_bleu(gt_text_list, predicted_text, dataset):
    """
        gt_text_list 要求为list的形式，详细格式见 get_reference_tokens

    :param gt_text_list:
    :param predicted_text:
    :param dataset:
    :return:
    """

    output_data = []
    for i in range(len(predicted_text)):
        predicted_line = predicted_text[i]
        try:
            if dataset in ["webnlg17", "webnlg20", "e2e_clean", "dart", "webnlg", "webnlg2", "e2e", "dart"]:
                # if dataset in ["webnlg17", "webnlg20"]:
                pattern = r'The generated text is :(.*)'
                # Use re.search to find the match
                predicted_match = re.search(pattern, predicted_line)
                # Extract the text if there is a match
                if predicted_match:
                    out = predicted_match.group(1)
                    output_data.append(out)
                    # print(out, gt_out)
                else:
                    # print("No Matched Prompt predicted_text, idx: ", i, "directly append generated text to list")
                    output_data.append(predicted_line)
        except Exception:
            # print("No Matched Prompt")
            pass

    gt_data_list = []
    for idx in range(len(gt_text_list)):
        gt_data = []
        gt_text = gt_text_list[idx]
        for i in range(len(gt_text)):
            gt_line = gt_text[i]
            try:
                if dataset in ["webnlg17", "webnlg20", "e2e_clean", "dart", "webnlg", "webnlg2", "e2e", "dart"]:
                    # if dataset in ["webnlg17", "webnlg20"]:
                    pattern = r'The generated text is :(.*)'
                    # Use re.search to find the match
                    gt_match = re.search(pattern, gt_line)
                    # Extract the text if there is a match
                    if gt_match:
                        gt_out = gt_match.group(1)
                        gt_data.append(gt_out)
                        # print(out, gt_out)
                    else:
                        # print("No Matched Prompt gt_text, idx: ", i, "directly append generated text to list")
                        gt_data.append(gt_line)

            except Exception:
                pass
                # print("No Matched Prompt")

        gt_data_list.append(gt_data)

    output_data_tokens = get_regexp_tokens(output_data)
    gt_data_list_tokens = get_reference_tokens(gt_data_list)

    metric = load_metric(path=r"/home/sdb/xx/path/datasets/HFdatasets/metrics/bleu/bleu.py",
                         cache_dir="/home/sdb/xx/path/datasets/HFdatasets/metrics",
                         trust_remote_code=True)

    results = metric.compute(references=gt_data_list_tokens, predictions=output_data_tokens)
    return results

if __name__ == "__main__":
    dataset = "webnlg" # webnlg webnlg2 e2e dart
    # test_file = f"/home/sdb/xx/path/LLM/references/critic-aware-decoding-new/data/webnlg"
    test_file = f"/home/sdb/xx/path/LLM/references/critic-aware-decoding-new/mvp_dataset/{dataset}_for_decode"
    gt_text_list = []
    test_type = ""

    for test_targe_file in ["0", "1", "2",]:
        gt_file = os.path.join(test_file, f"test.target{test_targe_file}")
        gt_text = open(gt_file, "r").readlines()
        gt_text_list.append(gt_text)

    # save_path = f"/home/sdb/xx/path/LLM/references/critic-aware-decoding-new/experiments/webnlg/"
    # save_path = f"/home/sdb/xx/path/LLM/references/critic-aware-decoding-new/experiments/{dataset}_mvp/"
    save_path = f"/home/sdb/xx/path/LLM/references/critic-aware-decoding-new/experiments/{dataset}_mvp_t5FT/"
    # pred_file = os.path.join(save_path, "FILE_NAME")
    pred_file = os.path.join(save_path, "test")
    pred_text = open(pred_file, "r").readlines()
    print("predicted.txt :", len(pred_text))

    results = metric_with_bleu(gt_text_list, pred_text, dataset="webnlg")

    print(results)


    from argparse import ArgumentParser

    # parser = ArgumentParser()
    # parser.add_argument('--model_name', default=None, type=str, required=True)
    # parser.add_argument(
    #     '--local_dataset_name', default="webnlg17", type=str,
    #     help='dataset name')
    # parser.add_argument('--from_batch', action='store_true',
    #                     help='wether dataset from batch')
    # args = parser.parse_args()
    # dataset_name = args.local_dataset_name
    # model_name = args.model_name
    #
    # if dataset_name in ["webnlg17", "webnlg20", "e2e_clean", "DART"]:
    #     test_file = f"/home/sdb/xx/path/LLM/references/ControlPrefixes/src/data/{dataset_name}"
    #     gt_text_list = []
    #
    #     if dataset_name in ["webnlg17", "webnlg20"]:
    #         test_type = "_both"  # _both _seen _unseen
    #     elif dataset_name in ["e2e_clean", ]:
    #         test_type = ""
    #     elif dataset_name in ["DART", ]:
    #         test_type = "_both"
    #
    #     for test_targe_file in ["", "2", "3"]:
    #         gt_file = os.path.join(test_file, f"test{test_type}.target{test_targe_file}")
    #         gt_text = open(gt_file, "r").readlines()
    #         gt_text_list.append(gt_text)
    #         print(f"test{test_type}.target{test_targe_file} :", len(gt_text))
    #
    #     if args.from_batch:
    #         print("--- batch mode ---")
    #         save_path = f"{dataset_name}-{model_name}_pred_batch"
    #     else:
    #         print("--- batch size of 1 mode ---")
    #         save_path = f"{dataset_name}-{model_name}_pred"
    #
    #     pred_file = os.path.join(save_path, "predicted.txt")
    #     pred_text = open(pred_file, "r").readlines()
    #     print("predicted.txt :", len(pred_text))
    #
    #     results = metric_with_bleu(gt_text_list, pred_text, dataset=dataset_name)
    #
    #     print(results)
    # elif dataset_name in ["webnlg", "webnlg2", "e2e", "dart"]:
    #     # MVP 项目提供的数据集
    #     test_file = f"/home/sdb/xx/path/LLM/mvpData2Text/Dataset/{dataset_name}"
    #     gt_text_list = []
    #     test_type = ""
    #
    #     for test_targe_file in ["1", "2", "3"]:
    #         gt_file = os.path.join(test_file, f"test{test_type}.target{test_targe_file}")
    #         gt_text = open(gt_file, "r").readlines()
    #         gt_text_list.append(gt_text)
    #         print(f"test{test_type}.target{test_targe_file} :", len(gt_text))
    #
    #     if args.from_batch:
    #         print("--- batch mode ---")
    #         save_path = f"{dataset_name}-{model_name}_pred_batch"
    #     else:
    #         print("--- batch size of 1 mode ---")
    #         save_path = f"{dataset_name}-{model_name}_pred"
    #
    #     pred_file = os.path.join(save_path, "predicted.txt")
    #     pred_text = open(pred_file, "r").readlines()
    #     print("predicted.txt :", len(pred_text))
    #
    #     results = metric_with_bleu(gt_text_list, pred_text, dataset=dataset_name)
    #
    #     print(results)

