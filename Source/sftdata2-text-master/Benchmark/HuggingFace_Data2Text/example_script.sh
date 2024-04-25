#!/usr/bin/env bash


# webnlg17
#python3 prepare_hf.py --local_dataset_name webnlg17
#
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --num_processes 8 run_hf.py \
#    --model_name_or_path "/home/sdb/xx/path/modelZooHuggingFace/t5-large" \
#    --do_train \
#    --seed=88 \
#    --save_total_limit=1 \
#    --local_dataset_name webnlg17 \
#    --train_file webnlg17_JSON/train.json \
#    --validation_file webnlg17_JSON/val.json \
#    --output_dir webnlg17_Pretrained_T5 \
#    --per_device_train_batch_size=8 \
#    --predict_with_generate \
#    --num_train_epochs 5.0 \
#    --save_strategy steps \
#    --evaluation_strategy steps \
#    --lr_scheduler_type cosine \
#    --load_best_model_at_end \
#    --overwrite_output_dir \
#    --eval_steps 200 \
#    --save_steps 200 \
#    --log_on_each_node False \
#    --log_level info
#
#CUDA_VISBLE_DEVICES=0,1,2 python3 run_inference.py \
#      --test_file "/home/sdb/xx/path/LLM/SFTData2Text/Dataset/webnlg17Prompt" \
#      --save_path webnlg17_Pretrained_T5_pred \
#      --dataset_name webnlg17 \
#      --model_name T5 \
#      --batch_size 16 \
#      --model_path "/home/sdb/xx/path/LLM/SFTData2Text/Benchmark/HuggingFace_Data2Text/webnlg17_Pretrained_T5/"
#
#
#python3 metrics.py --local_dataset_name webnlg17


# webnlg20
#python3 prepare_hf.py --local_dataset_name webnlg20
#
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --num_processes 8 run_hf.py \
#    --model_name_or_path "/home/sdb/xx/path/modelZooHuggingFace/t5-large" \
#    --do_train \
#    --seed=88 \
#    --save_total_limit=1 \
#    --local_dataset_name webnlg20 \
#    --train_file webnlg20_JSON/train.json \
#    --validation_file webnlg20_JSON/val.json \
#    --output_dir webnlg20_Pretrained_T5 \
#    --per_device_train_batch_size=8 \
#    --predict_with_generate \
#    --num_train_epochs 5.0 \
#    --save_strategy steps \
#    --evaluation_strategy steps \
#    --lr_scheduler_type cosine \
#    --load_best_model_at_end \
#    --eval_steps 200 \
#    --save_steps 200 \
#    --log_on_each_node False \
#    --log_level info
##    --resume_from_checkpoint True
##    --overwrite_output_dir \
#
#CUDA_VISBLE_DEVICES=0,1,2 python3 run_inference.py \
#      --test_file "/home/sdb/xx/path/LLM/SFTData2Text/Dataset/webnlg20Prompt" \
#      --save_path webnlg20_Pretrained_T5_pred \
#      --dataset_name webnlg20 \
#      --model_name T5 \
#      --batch_size 16 \
#      --model_path "/home/sdb/xx/path/LLM/SFTData2Text/Benchmark/HuggingFace_Data2Text/webnlg20_Pretrained_T5/"
#
#python3 metrics.py --local_dataset_name webnlg20


# e2e_clean
#python3 prepare_hf.py --local_dataset_name e2e_clean
#
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --num_processes 8 run_hf.py \
#    --model_name_or_path "/home/sdb/xx/path/modelZooHuggingFace/t5-large" \
#    --do_train \
#    --seed=88 \
#    --save_total_limit=1 \
#    --local_dataset_name e2e_clean \
#    --train_file e2e_clean_JSON/train.json \
#    --validation_file e2e_clean_JSON/val.json \
#    --output_dir e2e_clean_Pretrained_T5 \
#    --per_device_train_batch_size=8 \
#    --predict_with_generate \
#    --num_train_epochs 5.0 \
#    --save_strategy steps \
#    --evaluation_strategy steps \
#    --lr_scheduler_type cosine \
#    --load_best_model_at_end \
#    --overwrite_output_dir \
#    --eval_steps 200 \
#    --save_steps 200 \
#    --log_on_each_node False \
#    --log_level info
#
#CUDA_VISBLE_DEVICES=0,1,2 python3 run_inference.py \
#      --test_file "/home/sdb/xx/path/LLM/SFTData2Text/Dataset/e2e_cleanPrompt" \
#      --save_path e2e_clean_Pretrained_T5_pred \
#      --dataset_name e2e_clean \
#      --model_name T5 \
#      --batch_size 16 \
#      --model_path "/home/sdb/xx/path/LLM/SFTData2Text/Benchmark/HuggingFace_Data2Text/e2e_clean_Pretrained_T5/"
#
#python3 metrics.py --local_dataset_name e2e_clean


## DART
#python3 prepare_hf.py --local_dataset_name DART
#
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --num_processes 8 run_hf.py \
#    --model_name_or_path "/home/sdb/xx/path/modelZooHuggingFace/t5-large" \
#    --do_train \
#    --seed=88 \
#    --save_total_limit=1 \
#    --local_dataset_name DART \
#    --train_file DART_JSON/train.json \
#    --validation_file DART_JSON/val.json \
#    --output_dir DART_Pretrained_T5 \
#    --per_device_train_batch_size=8 \
#    --predict_with_generate \
#    --num_train_epochs 5.0 \
#    --save_strategy steps \
#    --evaluation_strategy steps \
#    --lr_scheduler_type cosine \
#    --load_best_model_at_end \
#    --eval_steps 200 \
#    --save_steps 200 \
#    --log_on_each_node False \
#    --log_level info
##    --resume_from_checkpoint
##    --overwrite_output_dir \
#
#CUDA_VISBLE_DEVICES=0,1,2 python3 run_inference.py \
#      --test_file "/home/sdb/xx/path/LLM/SFTData2Text/Dataset/DARTPrompt" \
#      --save_path DART_Pretrained_T5_pred \
#      --dataset_name DART \
#      --model_name T5 \
#      --batch_size 16 \
#      --model_path "/home/sdb/xx/path/LLM/SFTData2Text/Benchmark/HuggingFace_Data2Text/DART_Pretrained_T5/"
#
#python3 metrics.py --local_dataset_name DART


# webnlg
#dataset_name='webnlg'
#python3 prepare_hf.py --local_dataset_name ${dataset_name}

#CUDA_VISIBLE_DEVICES=2,3,4,5,  accelerate launch --num_processes 4 run_hf.py \
#    --model_name_or_path "/home/sdb/xx/path/modelZooHuggingFace/t5-large" \
#    --do_train \
#    --seed=88 \
#    --save_total_limit=3 \
#    --local_dataset_name ${dataset_name} \
#    --train_file ${dataset_name}_JSON/train.json \
#    --validation_file ${dataset_name}_JSON/val.json \
#    --output_dir ${dataset_name}_Pretrained_T5 \
#    --per_device_train_batch_size=8 \
#    --predict_with_generate \
#    --num_train_epochs 5.0 \
#    --save_strategy steps \
#    --evaluation_strategy steps \
#    --lr_scheduler_type cosine \
#    --load_best_model_at_end \
#    --overwrite_output_dir \
#    --eval_steps 500 \
#    --save_steps 500 \
#    --log_on_each_node False \
#    --log_level info

#CUDA_VISBLE_DEVICES=2,3, python3 run_inference.py \
#      --test_file "/home/sdb/xx/path/LLM/SFTData2Text/Dataset/${dataset_name}" \
#      --save_path ${dataset_name}_Pretrained_T5_pred \
#      --dataset_name ${dataset_name} \
#      --model_name T5 \
#      --batch_size 16 \
#      --model_path "/home/sdb/xx/path/LLM/SFTData2Text/Benchmark/HuggingFace_Data2Text/${dataset_name}_Pretrained_T5/"

#python3 metrics.py --local_dataset_name ${dataset_name}


## webnlg2
#dataset_name='webnlg2'
#python3 prepare_hf.py --local_dataset_name ${dataset_name}
#
#CUDA_VISIBLE_DEVICES=2,3,4,5, accelerate launch --num_processes 4 run_hf.py \
#    --model_name_or_path "/home/sdb/xx/path/modelZooHuggingFace/t5-large" \
#    --do_train \
#    --seed=88 \
#    --save_total_limit=3 \
#    --local_dataset_name ${dataset_name} \
#    --train_file ${dataset_name}_JSON/train.json \
#    --validation_file ${dataset_name}_JSON/val.json \
#    --output_dir ${dataset_name}_Pretrained_T5 \
#    --per_device_train_batch_size=8 \
#    --predict_with_generate \
#    --num_train_epochs 5.0 \
#    --save_strategy steps \
#    --evaluation_strategy steps \
#    --lr_scheduler_type cosine \
#    --load_best_model_at_end \
#    --overwrite_output_dir \
#    --eval_steps 500 \
#    --save_steps 500 \
#    --log_on_each_node False \
#    --log_level info
#
#CUDA_VISBLE_DEVICES=2,3, python3 run_inference.py \
#      --test_file "/home/sdb/xx/path/LLM/SFTData2Text/Dataset/${dataset_name}" \
#      --save_path ${dataset_name}_Pretrained_T5_pred \
#      --dataset_name ${dataset_name} \
#      --model_name T5 \
#      --batch_size 16 \
#      --model_path "/home/sdb/xx/path/LLM/SFTData2Text/Benchmark/HuggingFace_Data2Text/${dataset_name}_Pretrained_T5/"
#
#
#python3 metrics.py --local_dataset_name ${dataset_name}
#
## e2e
#dataset_name='e2e'
#python3 prepare_hf.py --local_dataset_name ${dataset_name}
#
#CUDA_VISIBLE_DEVICES=2,3,4,5, accelerate launch --num_processes 4 run_hf.py \
#    --model_name_or_path "/home/sdb/xx/path/modelZooHuggingFace/t5-large" \
#    --do_train \
#    --seed=88 \
#    --save_total_limit=3 \
#    --local_dataset_name ${dataset_name} \
#    --train_file ${dataset_name}_JSON/train.json \
#    --validation_file ${dataset_name}_JSON/val.json \
#    --output_dir ${dataset_name}_Pretrained_T5 \
#    --per_device_train_batch_size=8 \
#    --predict_with_generate \
#    --num_train_epochs 5.0 \
#    --save_strategy steps \
#    --evaluation_strategy steps \
#    --lr_scheduler_type cosine \
#    --load_best_model_at_end \
#    --overwrite_output_dir \
#    --eval_steps 500 \
#    --save_steps 500 \
#    --log_on_each_node False \
#    --log_level info
#
#CUDA_VISBLE_DEVICES=2,3, python3 run_inference.py \
#      --test_file "/home/sdb/xx/path/LLM/SFTData2Text/Dataset/${dataset_name}" \
#      --save_path ${dataset_name}_Pretrained_T5_pred \
#      --dataset_name ${dataset_name} \
#      --model_name T5 \
#      --batch_size 16 \
#      --model_path "/home/sdb/xx/path/LLM/SFTData2Text/Benchmark/HuggingFace_Data2Text/${dataset_name}_Pretrained_T5/"
#
#
#python3 metrics.py --local_dataset_name ${dataset_name}

# dart
dataset_name='dart'
#python3 prepare_hf.py --local_dataset_name ${dataset_name}
#
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 accelerate launch --num_processes 6 run_hf.py \
#    --model_name_or_path "/home/sdb/xx/path/modelZooHuggingFace/t5-large" \
#    --do_train \
#    --seed=88 \
#    --save_total_limit=3 \
#    --local_dataset_name ${dataset_name} \
#    --train_file ${dataset_name}_JSON/train.json \
#    --validation_file ${dataset_name}_JSON/val.json \
#    --output_dir ${dataset_name}_Pretrained_T5 \
#    --per_device_train_batch_size=6 \
#    --fp16=True \
#    --predict_with_generate \
#    --num_train_epochs 5.0 \
#    --save_strategy steps \
#    --evaluation_strategy steps \
#    --lr_scheduler_type cosine \
#    --load_best_model_at_end \
#    --overwrite_output_dir \
#    --eval_steps 500 \
#    --save_steps 500 \
#    --log_on_each_node False \
#    --log_level info \
#    --resume_from_checkpoint "dart_Pretrained_T5/checkpoint-500/"

CUDA_VISBLE_DEVICES=2,3, python3 run_inference.py \
      --test_file "/home/sdb/xx/path/LLM/SFTData2Text/Dataset/${dataset_name}" \
      --save_path ${dataset_name}_Pretrained_T5_pred \
      --dataset_name ${dataset_name} \
      --model_name T5 \
      --batch_size 16 \
      --model_path "/home/sdb/xx/path/LLM/SFTData2Text/Benchmark/HuggingFace_Data2Text/${dataset_name}_Pretrained_T5/"


python3 metrics.py --local_dataset_name ${dataset_name}

















