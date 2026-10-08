#!/bin/bash
set -euo pipefail

# ===================== GPU / batch config =====================
export CUDA_VISIBLE_DEVICES=0
BATCH_SIZE=${BATCH_SIZE:-2}

# ===================== FaithScore inference =====================
python qwen2_5vl_parallel.py \
    --exp-tag "aims_test" \
    --model-path /root/autodl-tmp/Qwen2.5-VL-3B-Instruct \
    --model-name qwen2_5vl \
    --data-path /root/autodl-tmp/aims_benchmarks/amber/image \
    --data-json ./AMBER/data/query/query_generative.json \
    --batch-size ${BATCH_SIZE} \
    --num-images 5000 \
    --use-qsteer-adaptive \
    --start-layer 2 \
    --end-layer 35 \
    --alpha 0.08 \
    --beam 1 \
    --visual-branch \
    --visual-sigma 1.5 \
    --prefill-branch \
    --prefill-sigma 1.2 \
    --decode-branch \
    --decode-window 1 \
    --decode-sigma 1.1 \
    --debug-number 32 \
    --bench-type amber \
