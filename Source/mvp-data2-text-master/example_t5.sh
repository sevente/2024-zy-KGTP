
mvp_model='t5-large'

#CUDA_VISIBLE_DEVICES=4, python generate.py \
#    --model_name ${mvp_model} \
#    --base_model "RUCAIBox/${mvp_model}" \
#    --test_file "./Dataset/webnlg17Instruct/test_both0.json" \
#    --save_path "webnlg17-${mvp_model}_pred" &
##    --turn_on_chat

#CUDA_VISIBLE_DEVICES=5, python generate.py \
#    --model_name ${mvp_model} \
#    --base_model "RUCAIBox/${mvp_model}" \
#    --test_file "./Dataset/webnlg20Instruct/test_both0.json" \
#    --save_path "webnlg20-${mvp_model}_pred"
##    --turn_on_chat

#CUDA_VISIBLE_DEVICES=2, python generate.py \
#    --model_name ${mvp_model} \
#    --base_model "RUCAIBox/${mvp_model}" \
#    --test_file "./Dataset/e2e_cleanInstruct/test0.json" \
#    --save_path "e2e_clean-${mvp_model}_pred" \
##    --turn_on_chat

#CUDA_VISIBLE_DEVICES=3, python generate.py \
#    --model_name ${mvp_model} \
#    --base_model "RUCAIBox/${mvp_model}" \
#    --test_file "./Dataset/DARTInstruct/test_both0.json" \
#    --save_path "DART-${mvp_model}_pred" \
##    --turn_on_chat


#CUDA_VISIBLE_DEVICES=6, python generate.py \
#    --model_name ${mvp_model} \
#    --base_model "/home/sdb/xx/path/modelZooHuggingFace/t5-large" \
#    --test_file "./Dataset/webnlg/test.json" \
#    --save_path "webnlg-${mvp_model}_pred" &
##    --turn_on_chat
#
#CUDA_VISIBLE_DEVICES=7, python generate.py \
#    --model_name ${mvp_model} \
#    --base_model "/home/sdb/xx/path/modelZooHuggingFace/t5-large" \
#    --test_file "./Dataset/webnlg2/test.json" \
#    --save_path "webnlg2-${mvp_model}_pred"  &
##    --turn_on_chat
#
#CUDA_VISIBLE_DEVICES=0, python generate.py \
#    --model_name ${mvp_model} \
#    --base_model "/home/sdb/xx/path/modelZooHuggingFace/t5-large" \
#    --test_file "./Dataset/e2e/test.json" \
#    --save_path "e2e-${mvp_model}_pred" &
##    --turn_on_chat
#
#CUDA_VISIBLE_DEVICES=1, python generate.py \
#    --model_name ${mvp_model} \
#    --base_model "/home/sdb/xx/path/modelZooHuggingFace/t5-large" \
#    --test_file "./Dataset/dart/test.json" \
#    --save_path "dart-${mvp_model}_pred"
##    --turn_on_chat


#python metrics.py \
#    --model_name "${mvp_model}" \
#    --local_dataset_name 'webnlg17'
#
#python metrics.py \
#    --model_name "${mvp_model}"  \
#    --local_dataset_name 'webnlg20'

#python metrics.py \
#    --model_name "${mvp_model}" \
#    --local_dataset_name 'e2e_clean'
#
#python metrics.py \
#    --model_name "${mvp_model}"  \
#    --local_dataset_name 'DART'

python metrics.py \
    --model_name "${mvp_model}" \
    --local_dataset_name 'webnlg'

python metrics.py \
    --model_name "${mvp_model}"  \
    --local_dataset_name 'webnlg2'

python metrics.py \
    --model_name "${mvp_model}" \
    --local_dataset_name 'e2e'

python metrics.py \
    --model_name "${mvp_model}"  \
    --local_dataset_name 'dart'