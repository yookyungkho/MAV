cd ..
task=trec
k=12
template="*cls**mask*:*+sent_0**sep+*"
mapping="{0:'Expression',1:'Entity',2:'Description',3:'Human',4:'Location',5:'Number'}"


# ------------- 1. Semi-supervised (ours) -------------
for seed in 13 21 42 87 100
do
  CUDA_VISIBLE_DEVICES=0 python3 run.py \
    --task_name $task \
    --data_dir data/few-shot/$task/$k-4-$seed \
    --do_train \
    --do_eval \
    --do_predict \
    --evaluate_during_training \
    --model_name_or_path roberta-base \
    --dataloader_num_workers 8 \
    --num_k $k \
    --max_seq_length 256 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 2 \
    --eval_accumulation_steps 10 \
    --learning_rate 1e-5 \
    --max_steps 0 \
    --logging_steps 100 \
    --eval_nums 20 \
    --num_train_epochs 200 \
    --output_dir exp_result/mav-ssl-singleaug_mask-$task \
    --seed $seed \
    --template $template \
    --mapping $mapping \
    --base_mode ssl \
    --threshold 0.95 \
    --lam1 0.5 \
    --lam2 0.1 \
    --use_st_loss \
    --st_loss_type fix_sflm \
    --use_mlm_loss \
    --overwrite_output_dir \
    --wandb_group mav-ssl-singleaug_mask-$task-$seed \
    --thresh_warmup \
    --single_aug \
    --single_aug_type RandomMask \
    --lm_freeze
done

python3 calculate_result.py \
  --result_dir exp_result/mav-ssl-singleaug_mask-$task \
  --task $task \
  --bin_mode remove


# 2. ------------- Small Supervised -------------
for seed in 13 21 42 87 100
do
  CUDA_VISIBLE_DEVICES=0 python3 run.py \
    --task_name $task \
    --data_dir data/few-shot/$task/$k-4-$seed \
    --do_train \
    --do_eval \
    --do_predict \
    --evaluate_during_training \
    --model_name_or_path roberta-base \
    --dataloader_num_workers 8 \
    --num_k $k \
    --max_seq_length 256 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 2 \
    --eval_accumulation_steps 10 \
    --learning_rate 1e-5 \
    --max_steps 0 \
    --logging_steps 100 \
    --eval_nums 20 \
    --num_train_epochs 200 \
    --output_dir exp_result/mav-small_sup-$task \
    --seed $seed \
    --template $template \
    --mapping $mapping \
    --base_mode sup \
    --train_type train \
    --overwrite_output_dir \
    --wandb_group mav-small_sup-$task-$seed
done

python3 calculate_result.py \
  --result_dir exp_result/mav-small_sup-$task \
  --task $task \
  --bin_mode remove


# 3. ------------- Full Supervised -------------
for seed in 13 21 42 87 100
do
  CUDA_VISIBLE_DEVICES=0 python3 run.py \
    --task_name $task \
    --data_dir data/few-shot/$task/$k-4-$seed \
    --do_train \
    --do_eval \
    --do_predict \
    --evaluate_during_training \
    --model_name_or_path roberta-base \
    --dataloader_num_workers 8 \
    --num_k $k \
    --max_seq_length 256 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 2 \
    --eval_accumulation_steps 10 \
    --learning_rate 1e-5 \
    --max_steps 0 \
    --logging_steps 100 \
    --eval_nums 20 \
    --num_train_epochs 200 \
    --output_dir exp_result/mav-full_sup-$task \
    --seed $seed \
    --template $template \
    --mapping $mapping \
    --base_mode sup \
    --train_type full_train \
    --overwrite_output_dir \
    --wandb_group mav-full_sup-$task-$seed
done

python3 calculate_result.py \
  --result_dir exp_result/mav-full_sup-$task \
  --task $task \
  --bin_mode remove
