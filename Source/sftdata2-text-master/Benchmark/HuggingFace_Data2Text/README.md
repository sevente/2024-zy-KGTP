运行 Promting/webnlg2017.py, 生成带prompt的数据，Dataset/webnlg17Prompt。 针对test数据集， 为了计算bleu score可能存在多个target文件.

随后运行 prepare_hf.py: convert prompt dataset train/val files to json format for training HuggingFace models

然后在Benchmark/HuggingFace_Data2Text文件夹下运行 example_script.sh 脚本，先微调模型，随后测试模型，生成 webnlg17_Pretrained_T5_pred 中的predictions文件。

最后在Benchmark/HuggingFace_Data2Text文件夹下运行 metrics.py, 输出BLEU score。

metrics.py: calculating BLEU score of the output predictions.

## 问题记录：
- run_hf.py 在训练模型过程中，出现out of memory的问题。 我们想通过减小配置的参数per_device_train_batch_size也即batch size，来解决该问题，
但是又想resume from checkpoint，Transformer库的Trainer训练管理器会直接从保存的 trainer_state.json 文件中恢复训练参数，而忽略我们从cmd line
命令行输入的参数，可以通过直接修改trainer_state.json文件中的batch size来解决该问题。 torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 12.00 MiB. GPU 5 has a total capacty of 23.69 GiB of which 5.69 MiB is free. Including non-PyTorch memory, this process has 23.68 GiB memory in use. Of the allocated memory 22.72 GiB is allocated by
PyTorch, and 567.06 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF



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

