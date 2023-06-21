cd ..
task=trec
k=12
bs=512
template="*cls**mask*:*+sent_0**sep+*"
mapping="{0:'Expression',1:'Entity',2:'Description',3:'Human',4:'Location',5:'Number'}"
type=test

#   1. SHAP value for each class
for seed in 13 21 42 87 100
do
    CUDA_VISIBLE_DEVICES=0 python3 analysis.py \
        --task_name $task \
        --data_dir data/few-shot/$task/$k-4-$seed \
        --model_name_or_path exp_result/mav-ssl-singleaug_mask-$task/seed$seed \
        --output_dir exp_result/mav-ssl-singleaug_mask-$task \
        --dataloader_num_workers 8 \
        --num_k $k \
        --max_seq_length 256 \
        --inf_batch_size $bs \
        --per_device_eval_batch_size 64 \
        --eval_accumulation_steps 10 \
        --seed $seed \
        --template $template \
        --mapping $mapping \
        --inf_data_type $type \
        --analysis_type shap_value
done


#   2. t-SNE visualization of [MASK] representation
for seed in 13 21 42 87 100
do
    CUDA_VISIBLE_DEVICES=0 python3 analysis.py \
        --task_name $task \
        --data_dir data/few-shot/$task/$k-4-$seed \
        --model_name_or_path exp_result/mav-ssl-singleaug_mask-$task/seed$seed \
        --output_dir exp_result/mav-ssl-singleaug_mask-$task \
        --dataloader_num_workers 8 \
        --num_k $k \
        --max_seq_length 256 \
        --inf_batch_size $bs \
        --per_device_eval_batch_size 64 \
        --eval_accumulation_steps 10 \
        --seed $seed \
        --template $template \
        --mapping $mapping \
        --inf_data_type $type \
        --analysis_type tsne \
        --return_mask_rep
done