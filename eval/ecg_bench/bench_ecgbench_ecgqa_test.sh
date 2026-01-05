#!/bin/bash


SAVE_DIR=./eval_outputs

model_path='/root/.cache/huggingface/checkpoints/cfel-yhz-0903-8-100K/'

# Set directories and files
save_dir='./eval_outputs/deepseekocr-ecg/ecg-bench-test/cpsc-test'
if [ ! -d "$save_dir" ]; then
    mkdir -p "$save_dir"
fi

CUDA_VISIBLE_DEVICES=0 python model_ecg_resume.py \
    --model-path "$model_path" \
    --image-folder "/root/.cache/huggingface/hub/datasets--PULSE-ECG--ECGBench/snapshots/cc7bfe06da6b7ca5b4e890d95e1a4099e4b248e2/ecgqa-test/images" \
    --question-file "/root/.cache/huggingface/hub/datasets--PULSE-ECG--ECGBench/snapshots/cc7bfe06da6b7ca5b4e890d95e1a4099e4b248e2/ecgqa-test/ecgqa_test.jsonl" \
    --answers-file "${save_dir}/step-final.jsonl" \
    --ecg-folder "./root/.cache/huggingface/hub/ecg" \
