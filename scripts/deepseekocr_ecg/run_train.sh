#!/bin/bash

# 设置模型路径和数据路径
MODEL_PATH="/root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-OCR/snapshots/2c968b433af61a059311cbf8997765023806a24d"
ECG_WEIGHT_PATH="/root/.cache/huggingface/hub/ecg/epoch_20.pt"
DATA_PATH="/workspace/ECG_Grounding_100k.jsonl"
OUTPUT_DIR="checkpoints/deepseekocr_ecg_sft"

# 设置分布式参数 (8卡)
NUM_GPUS=8

# 如果显存非常充裕，可以调大 PER_DEVICE_BATCH_SIZE
# H800 上对于 0.5B 模型，单卡 batch 可能可以开到 16 甚至更大
PER_DEVICE_BATCH_SIZE=8
GRADIENT_ACCUMULATION_STEPS=2

# 启动训练
torchrun --nproc_per_node=$NUM_GPUS --master_port=29501 qwen_train.py \
    --model_name_or_path $MODEL_PATH \
    --ecg_weight_path $ECG_WEIGHT_PATH \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --max_length 8196 \
    --deepspeed ds_config_zero2.json \
    --bf16 True \
    --num_train_epochs 6 \
    --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --evaluation_strategy "steps" \
    --eval_steps 1000 \
    --save_strategy "epoch" \
    --save_total_limit 5 \
    --learning_rate 2e-5 \
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