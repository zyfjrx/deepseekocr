export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 设置模型路径和数据路径
MODEL_PATH="/workspace/deepseekocr/checkpoints/qwen3_8b_ocr_proj/checkpoint-310"
DATA_PATH="/workspace/ecg_iamge_warmup_40k.jsonl"
OUTPUT_DIR="checkpoints/ecg_qwen3_8b_ocr_proj_0108"

# 设置分布式参数 (8卡)
NUM_GPUS=8

PER_DEVICE_BATCH_SIZE=8
GRADIENT_ACCUMULATION_STEPS=2

# 启动训练
torchrun --nproc_per_node=$NUM_GPUS --master_port=29501 stage1_ecg_qwen_train.py \
    --model_name_or_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --deepspeed zero2.json \
    --bf16 True \
    --num_train_epochs 3 \
    --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --eval_steps 100 \
    --eval_strategy "steps" \
    --save_strategy "epoch" \
    --save_total_limit 2 \
    --learning_rate 1e-3 \
    --weight_decay 0.05 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to "tensorboard" \
    --dataloader_num_workers 8 \
    --freeze_vision True \
    --remove_unused_columns False