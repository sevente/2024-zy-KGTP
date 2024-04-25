#!/usr/bin/env bash

# webnlg17
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python finetune.py \
#    --base_model '/home/sdb/xx/path/modelZooHuggingFace/Llama-2-7b-hf' \
#    --data_path './Dataset/webnlg17Instruct/train.json' \
#    --output_dir './webnlg17-lora-alpaca' \
#    --batch_size 128 \
#    --micro_batch_size 16 \
#    --num_epochs 5.0 \
#    --learning_rate 1e-4 \
#    --cutoff_len 512 \
#    --val_set_size 2000 \
#    --lora_r 8 \
#    --lora_alpha 16 \
#    --lora_dropout 0.05 \
#    --lora_target_modules '[q_proj,v_proj]' \
#    --train_on_inputs \
#    --group_by_length
##    --resume_from_checkpoint './lora-alpaca/checkpoint-4'

#CUDA_VISIBLE_DEVICES=0,1,2,3 python generate.py \
#    --base_model '/home/sdb/xx/path/modelZooHuggingFace/Llama-2-7b-hf' \
#    --lora_weights './webnlg17-lora-alpaca' \
#    --load_8bit \
#    --prompt_template_name 'alpaca' \
#    --test_file "./Dataset/webnlg17Instruct/test_both0.json" \
#    --save_path 'webnlg17-lora-alpaca_pred' \
##    --turn_on_chat

#CUDA_VISIBLE_DEVICES=0,1,2,3 python generate_batch.py \
#    --base_model '/home/sdb/xx/path/modelZooHuggingFace/Llama-2-7b-hf' \
#    --lora_weights './webnlg17-lora-alpaca' \
#    --load_8bit \
#    --prompt_template_name 'alpaca' \
#    --test_file "./Dataset/webnlg17Instruct/test_both0.json" \
#    --save_path 'webnlg17-lora-alpaca_pred_batch' &
##    --turn_on_chat
#
#python metrics.py \
#    --local_dataset_name 'webnlg17'


# webnlg20
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python finetune.py \
#    --base_model '/home/sdb/xx/path/modelZooHuggingFace/Llama-2-7b-hf' \
#    --data_path './Dataset/webnlg20Instruct/train.json' \
#    --output_dir './webnlg20-lora-alpaca' \
#    --batch_size 128 \
#    --micro_batch_size 16 \
#    --num_epochs 5.0 \
#    --learning_rate 1e-4 \
#    --cutoff_len 512 \
#    --val_set_size 2000 \
#    --lora_r 8 \
#    --lora_alpha 16 \
#    --lora_dropout 0.05 \
#    --lora_target_modules '[q_proj,v_proj]' \
#    --train_on_inputs \
#    --group_by_length
##    --resume_from_checkpoint './lora-alpaca/checkpoint-4'

#CUDA_VISIBLE_DEVICES=0,1, python generate.py \
#    --base_model '/home/sdb/xx/path/modelZooHuggingFace/Llama-2-7b-hf' \
#    --lora_weights './webnlg20-lora-alpaca' \
#    --load_8bit \
#    --prompt_template_name 'alpaca' \
#    --test_file "./Dataset/webnlg20Instruct/test_both0.json" \
#    --save_path 'webnlg20-lora-alpaca_pred' &
##    --turn_on_chat

#CUDA_VISIBLE_DEVICES=4,5,6,7 python generate_batch.py \
#    --base_model '/home/sdb/xx/path/modelZooHuggingFace/Llama-2-7b-hf' \
#    --lora_weights './webnlg20-lora-alpaca' \
#    --load_8bit \
#    --prompt_template_name 'alpaca' \
#    --test_file "./Dataset/webnlg20Instruct/test_both0.json" \
#    --save_path 'webnlg20-lora-alpaca_pred'
##    --turn_on_chat

#python metrics.py \
#    --local_dataset_name 'webnlg20'


# e2e_clean
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python finetune.py \
#    --base_model '/home/sdb/xx/path/modelZooHuggingFace/Llama-2-7b-hf' \
#    --data_path './Dataset/e2e_cleanInstruct/train.json' \
#    --output_dir './e2e_clean-lora-alpaca' \
#    --batch_size 128 \
#    --micro_batch_size 16 \
#    --num_epochs 5.0 \
#    --learning_rate 1e-4 \
#    --cutoff_len 512 \
#    --val_set_size 2000 \
#    --lora_r 8 \
#    --lora_alpha 16 \
#    --lora_dropout 0.05 \
#    --lora_target_modules '[q_proj,v_proj]' \
#    --train_on_inputs \
#    --group_by_length
##    --resume_from_checkpoint './lora-alpaca/checkpoint-4'

#CUDA_VISIBLE_DEVICES=2,3, python generate.py \
#    --base_model '/home/sdb/xx/path/modelZooHuggingFace/Llama-2-7b-hf' \
#    --lora_weights './e2e_clean-lora-alpaca' \
#    --load_8bit \
#    --prompt_template_name 'alpaca' \
#    --test_file "./Dataset/e2e_cleanInstruct/test0.json" \
#    --save_path 'e2e_clean-lora-alpaca_pred' \
##    --turn_on_chat

#CUDA_VISIBLE_DEVICES=0,1,2,3 python generate_batch.py \
#    --base_model '/home/sdb/xx/path/modelZooHuggingFace/Llama-2-7b-hf' \
#    --lora_weights './e2e_clean-lora-alpaca' \
#    --load_8bit \
#    --prompt_template_name 'alpaca' \
#    --test_file "./Dataset/e2e_cleanInstruct/test0.json" \
#    --save_path 'e2e_clean-lora-alpaca_pred' \
##    --turn_on_chat

#python metrics.py \
#    --local_dataset_name 'e2e_clean'

# DART
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python finetune.py \
#    --base_model '/home/sdb/xx/path/modelZooHuggingFace/Llama-2-7b-hf' \
#    --data_path './Dataset/DARTInstruct/train.json' \
#    --output_dir './DART-lora-alpaca' \
#    --batch_size 128 \
#    --micro_batch_size 16 \
#    --num_epochs 5.0 \
#    --learning_rate 1e-4 \
#    --cutoff_len 512 \
#    --val_set_size 2000 \
#    --lora_r 8 \
#    --lora_alpha 16 \
#    --lora_dropout 0.05 \
#    --lora_target_modules '[q_proj,v_proj]' \
#    --train_on_inputs \
#    --group_by_length
##    --resume_from_checkpoint './lora-alpaca/checkpoint-4'

CUDA_VISIBLE_DEVICES=0,1 python generate.py \
    --base_model '/home/sdb/xx/path/modelZooHuggingFace/Llama-2-7b-hf' \
    --lora_weights './DART-lora-alpaca' \
    --load_8bit \
    --prompt_template_name 'alpaca' \
    --test_file "./Dataset/DARTInstruct/test_both0.json" \
    --save_path 'DART-lora-alpaca_pred' \
#    --turn_on_chat

#CUDA_VISIBLE_DEVICES=0,1,2,3 python generate_batch.py \
#    --base_model '/home/sdb/xx/path/modelZooHuggingFace/Llama-2-7b-hf' \
#    --lora_weights './DART-lora-alpaca' \
#    --load_8bit \
#    --prompt_template_name 'alpaca' \
#    --test_file "./Dataset/DARTInstruct/test_both0.json" \
#    --save_path 'DART-lora-alpaca_pred' \
##    --turn_on_chat

python metrics.py \
    --local_dataset_name 'DART'