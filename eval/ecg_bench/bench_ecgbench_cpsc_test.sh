#!/bin/bash


SAVE_DIR=./eval_outputs

model_path='/root/.cache/huggingface/checkpoints/cfel-yhz-0903-8-100K/'

# Set directories and files
save_dir='./eval_outputs/deepseekocr-ecg/ecg-bench-test/cpsc-test'
if [ ! -d "$save_dir" ]; then
    mkdir -p "$save_dir"
fi

CUDA_VISIBLE_DEVICES=1 python model_ecg_resume.py \
    --model-path "$model_path" \
    --image-folder "/root/.cache/huggingface/hub/datasets--PULSE-ECG--ECGBench/snapshots/cc7bfe06da6b7ca5b4e890d95e1a4099e4b248e2/cpsc-test/images" \
    --question-file "/root/.cache/huggingface/hub/datasets--LANSG--ECG-Grounding/snapshots/f6a6796056fc5a1e4afd48539ceb325ac5dd0fd8/ecg_bench/cpsc-test.json" \
    --answers-file "${save_dir}/step-final.jsonl" \
    --ecg-folder "/root/.cache/huggingface/hub/ecg/ecg_ptbxl_benchmarking/data" \
