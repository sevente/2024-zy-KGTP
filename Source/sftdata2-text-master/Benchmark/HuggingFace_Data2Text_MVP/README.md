运行 Prompting/mvp_data2text_preprocessing.py, 生成带prompt的数据 以及 target文件，Dataset/webnlg。 针对test数据集， 为了计算bleu score可能存在多个target文件.

随后运行 prepare_hf.py: convert prompt dataset train/val files to json format for training HuggingFace models

然后在Benchmark/HuggingFace_Data2Text_MVP 文件夹下运行 example_script.sh 脚本，先微调模型，随后测试模型，生成 webnlg_Pretrained_T5_pred 中的predictions文件。

最后在Benchmark/HuggingFace_Data2Text_MVP 文件夹下运行 metrics.py, 输出BLEU score。

metrics.py: calculating BLEU score of the output predictions.

# 以下为参考仓库的readme
ref: https://github.com/HaoUNSW/PISA.git

run_hf.py: the script for training HuggingFace models with the converted PISA dataset. This script is based on the official HuggingFace examples: https://github.com/huggingface/transformers/blob/main/examples/legacy/seq2seq/finetune_trainer.py

run_inference.py: run evaluation on the test set of PISA and return RMSE, MAE, Missing_Rate.

An example script of training/evaluation is given in example_script.sh

```bash
python3 run_hf.py \
    --model_name_or_path google_pegasus-xsum\  # can be replaced with other models
    --do_train \
    --do_eval \  
    --seed=66 \  # can select differenet seeds
    --save_total_limit=1 \
    --train_file SG/train.json \  # need to warp the PISA data (txt files) into json format via prepare_hf.py
    --validation_file SG/val.json \
    --output_dir SG_Pretrained_Pegasus \
    --per_device_train_batch_size=16 \
    --overwrite_output_dir \
    --predict_with_generate

python3 /run_inference.py \
      -t SG \  # path to the test set of PISA
      -m SG_Pretrained_Pegasus \
      -s G_Pretrained_Pegasus_pred \
      -d SG \
      --model_name pegasus \
      -b 16

```

