#!/usr/bin/env bash


## webnlg
#dataset_name='webnlg'
#
#python3 prepare_hf.py --local_dataset_name ${dataset_name}
#
#CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --num_processes 4 run_hf.py \
#    --model_name_or_path "RUCAIBox/mvp-data-to-text" \
#    --do_train \
#    --seed=88 \
#    --save_total_limit=3 \
#    --local_dataset_name ${dataset_name} \
#    --train_file "${dataset_name}_JSON/train.json" \
#    --validation_file "${dataset_name}_JSON/val.json" \
#    --output_dir "${dataset_name}_Pretrained_MVPD2T" \
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
#    --log_level info \
#    --cache_dir "/home/sdb/xx/path/modelZooHuggingFace/cache"
#
#CUDA_VISBLE_DEVICES=4,5 python3 run_inference.py \
#      --test_file "/home/sdb/xx/path/LLM/SFTData2Text/Dataset/${dataset_name}" \
#      --save_path "${dataset_name}_Pretrained_MVPD2T_pred" \
#      --dataset_name ${dataset_name} \
#      --model_name mvp-data-to-text \
#      --batch_size 16 \
#      --model_path "/home/sdb/xx/path/LLM/SFTData2Text/Benchmark/HuggingFace_Data2Text_MVP/${dataset_name}_Pretrained_MVPD2T/"
#
#
#python3 metrics.py --local_dataset_name ${dataset_name}


##webnlg2
dataset_name='webnlg2'
#
#python3 prepare_hf.py --local_dataset_name ${dataset_name}
#
#CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --num_processes 4 run_hf.py \
#    --model_name_or_path "RUCAIBox/mvp-data-to-text" \
#    --do_train \
#    --seed=88 \
#    --save_total_limit=3 \
#    --local_dataset_name ${dataset_name} \
#    --train_file "${dataset_name}_JSON/train.json" \
#    --validation_file "${dataset_name}_JSON/val.json" \
#    --output_dir "${dataset_name}_Pretrained_MVPD2T" \
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
#    --log_level info \
#    --cache_dir "/home/sdb/xx/path/modelZooHuggingFace/cache"
#
#CUDA_VISBLE_DEVICES=4,5 python3 run_inference.py \
#      --test_file "/home/sdb/xx/path/LLM/SFTData2Text/Dataset/${dataset_name}" \
#      --save_path "${dataset_name}_Pretrained_MVPD2T_pred" \
#      --dataset_name ${dataset_name} \
#      --model_name mvp-data-to-text \
#      --batch_size 16 \
#      --model_path "/home/sdb/xx/path/LLM/SFTData2Text/Benchmark/HuggingFace_Data2Text_MVP/${dataset_name}_Pretrained_MVPD2T/"
#
#
python3 metrics.py --local_dataset_name ${dataset_name}
#
#
##e2e
dataset_name='e2e'
#
#python3 prepare_hf.py --local_dataset_name ${dataset_name}
#
#CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --num_processes 4 run_hf.py \
#    --model_name_or_path "RUCAIBox/mvp-data-to-text" \
#    --do_train \
#    --seed=88 \
#    --save_total_limit=3 \
#    --local_dataset_name ${dataset_name} \
#    --train_file "${dataset_name}_JSON/train.json" \
#    --validation_file "${dataset_name}_JSON/val.json" \
#    --output_dir "${dataset_name}_Pretrained_MVPD2T" \
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
#    --log_level info \
#    --cache_dir "/home/sdb/xx/path/modelZooHuggingFace/cache"
#
#CUDA_VISBLE_DEVICES=4,5 python3 run_inference.py \
#      --test_file "/home/sdb/xx/path/LLM/SFTData2Text/Dataset/${dataset_name}" \
#      --save_path "${dataset_name}_Pretrained_MVPD2T_pred" \
#      --dataset_name ${dataset_name} \
#      --model_name mvp-data-to-text \
#      --batch_size 16 \
#      --model_path "/home/sdb/xx/path/LLM/SFTData2Text/Benchmark/HuggingFace_Data2Text_MVP/${dataset_name}_Pretrained_MVPD2T/"
#
#
python3 metrics.py --local_dataset_name ${dataset_name}


#dart
dataset_name='dart'

#python3 prepare_hf.py --local_dataset_name ${dataset_name}

#CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --num_processes 4 run_hf.py \
#    --model_name_or_path "RUCAIBox/mvp-data-to-text" \
#    --do_train \
#    --seed=88 \
#    --save_total_limit=3 \
#    --local_dataset_name ${dataset_name} \
#    --train_file "${dataset_name}_JSON/train.json" \
#    --validation_file "${dataset_name}_JSON/val.json" \
#    --output_dir "${dataset_name}_Pretrained_MVPD2T" \
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
#    --log_level info \
#    --cache_dir "/home/sdb/xx/path/modelZooHuggingFace/cache" \
##    --resume_from_checkpoint "dart_Pretrained_MVPD2T/checkpoint-3200/"

#CUDA_VISBLE_DEVICES=4,5 python3 run_inference.py \
#      --test_file "/home/sdb/xx/path/LLM/SFTData2Text/Dataset/${dataset_name}" \
#      --save_path "${dataset_name}_Pretrained_MVPD2T_pred" \
#      --dataset_name ${dataset_name} \
#      --model_name mvp-data-to-text \
#      --batch_size 16 \
#      --model_path "/home/sdb/xx/path/LLM/SFTData2Text/Benchmark/HuggingFace_Data2Text_MVP/${dataset_name}_Pretrained_MVPD2T/"

python3 metrics.py --local_dataset_name ${dataset_name}














