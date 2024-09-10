export NCCL_DEBUG=INFO
export NCCL_LAUNCH_MODE=PARALLEL
export NCCL_IB_HCA=mlx5_
export NCCL_IB_TC=136
export NCCL_IB_SL=5
export NCCL_IB_GID_INDEX=3

export PATH=/usr/local/cuda/bin:$PATH


llama_path=path to llama model
export DATA_ROOT=path to your data
export SAVE_ROOT=path to save your model
mkdir -p $SAVE_ROOT


CUDA_VISIBLE_DEVICES=1 WANDB_MODE=offline python -m torch.distributed.run  --master_port=20005 --nproc_per_node=1 SocialRecNet/train.py \
    --deepspeed SocialRecNet/config/config.json \
    --data $DATA_ROOT \
    --output_dir ${SAVE_ROOT} \
    --manifest_files "your_training.jsonl" \
    --remove_unused_columns False \
    --seed 1 \c
    --do_train True \
    \
    --learning_rate 5e-5 \
    --weight_decay 0.05 \
    --max_grad_norm 1.0 \
    --warmup_steps 1000 \
    \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 12 \
    --num_train_epochs 10 \
    \
    --llama_model $llama_path \
    \
    --disable_tqdm True \
    \
    --logging_steps 10 \
    --save_steps 200 \
    --save_total_limit 1
