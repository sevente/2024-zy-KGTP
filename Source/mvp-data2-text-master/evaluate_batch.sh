# model: mvp-data-to-text


mvp_model='mvp'
#mvp_model='mvp-data-to-text'
#mvp_model='mtl-data-to-text'

#CUDA_VISIBLE_DEVICES=4, python generate_batch.py \
#    --model_name ${mvp_model} \
#    --base_model "RUCAIBox/${mvp_model}" \
#    --test_file "./Dataset/webnlg17Instruct/test_both0.json" \
#    --save_path "webnlg17-${mvp_model}_pred_batch" &
#
#CUDA_VISIBLE_DEVICES=5, python generate_batch.py \
#    --model_name ${mvp_model} \
#    --base_model "RUCAIBox/${mvp_model}" \
#    --test_file "./Dataset/webnlg20Instruct/test_both0.json" \
#    --save_path "webnlg20-${mvp_model}_pred_batch" &

#CUDA_VISIBLE_DEVICES=2, python generate_batch.py \
#    --model_name ${mvp_model} \
#    --base_model "RUCAIBox/${mvp_model}" \
#    --test_file "./Dataset/e2e_cleanInstruct/test0.json" \
#    --save_path "e2e_clean-${mvp_model}_pred_batch" &
#
#
#CUDA_VISIBLE_DEVICES=3, python generate_batch.py \
#    --model_name ${mvp_model} \
#    --base_model "RUCAIBox/${mvp_model}" \
#    --test_file "./Dataset/DARTInstruct/test_both0.json" \
#    --save_path "DART-${mvp_model}_pred_batch"

python metrics.py \
    --model_name "${mvp_model}" \
    --local_dataset_name 'webnlg17' \
    --from_batch

python metrics.py \
    --model_name "${mvp_model}"  \
    --local_dataset_name 'webnlg20' \
    --from_batch

#python metrics.py \
#    --model_name "${mvp_model}" \
#    --local_dataset_name 'e2e_clean'
#    --from_batch
#
#python metrics.py \
#    --model_name "${mvp_model}"  \
#    --local_dataset_name 'DART'
#    --from_batch



# 以下暂时没用

#mvp_model='mvp-data-to-text'

#CUDA_VISIBLE_DEVICES=0, python generate_batch.py \
#    --base_model 'RUCAIBox/mvp-data-to-text' \
#    --test_file "./Dataset/webnlg17Instruct/test_both0.json" \
#    --save_path 'webnlg17-mvp-data-to-text_pred'
#
#CUDA_VISIBLE_DEVICES=1, python generate_batch.py \
#    --base_model 'RUCAIBox/mvp-data-to-text' \
#    --test_file "./Dataset/webnlg20Instruct/test_both0.json" \
#    --save_path 'webnlg20-mvp-data-to-text_pred'

#CUDA_VISIBLE_DEVICES=2, python generate_batch.py \
#    --base_model 'RUCAIBox/mvp-data-to-text' \
#    --test_file "./Dataset/e2e_cleanInstruct/test0.json" \
#    --save_path 'e2e_clean-mvp-data-to-text_pred'&
#
#
#CUDA_VISIBLE_DEVICES=3, python generate_batch.py \
#    --base_model 'RUCAIBox/mvp-data-to-text' \
#    --test_file "./Dataset/DARTInstruct/test_both0.json" \
#    --save_path 'DART-mvp-data-to-text_pred'

#python metrics.py \
#    --model_name 'mvp-data-to-text' \
#    --local_dataset_name 'webnlg17'

#python metrics.py \
#    --model_name 'mvp-data-to-text' \
#    --local_dataset_name 'webnlg20'

#python metrics.py \
#    --model_name 'mvp-data-to-text' \
#    --local_dataset_name 'e2e_clean'
#
#python metrics.py \
#    --model_name 'mvp-data-to-text' \
#    --local_dataset_name 'DART'

