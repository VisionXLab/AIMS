#!/bin/bash
set -euo pipefail

# ===================== GPU / batch config =====================
# 单卡 batch 并行：仍然只使用物理 GPU 3，但一次 generate 处理多个 FaithScore 样本。A100-40GB 建议先试 batch=4，若 OOM 改成 2。
export CUDA_VISIBLE_DEVICES=0
BATCH_SIZE=${BATCH_SIZE:-1}

# ===================== FaithScore inference =====================
python qwen2_5vl_parallel.py \
    --exp-tag "baseline" \
    --model-name qwen2_5vl \
    --model-path /root/autodl-tmp/Qwen2.5-VL-3B-Instruct \
    --data-path /root/autodl-tmp/aims_benchmarks/faithscore/images \
    --data-json /root/autodl-tmp/aims_benchmarks/faithscore/coco2014_sample_1k.jsonl \
    --batch-size ${BATCH_SIZE} \
    --num-images 1000 \
    --start-layer 2 \
    --end-layer 35 \
    --alpha 0.2 \
    --beam 1 \
    --visual-branch \
    --visual-sigma 1. \
    --prefill-branch \
    --prefill-sigma 1. \
    --decode-branch \
    --decode-window 1 \
    --decode-sigma 1. \
    --debug-number 64 \
    --bench-type faithscore \
