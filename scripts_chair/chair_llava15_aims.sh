#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

python chair_eval.py \
    --exp-tag "" \
    --ablation "adaptive_mode" \
    --model llava-1.5 \
    --data-path /root/autodl-tmp/aims_benchmarks/chair_coco/val2014_random500 \
    --use-qsteer-adaptive \
    --alpha 0.1 \
    --start-layer 2 \
    --end-layer 32 \
    --alpha 0.06 \
    --beam 1 \
    --visual-branch \
    --visual-sigma 1.5 \
    --prefill-branch \
    --prefill-sigma 1.2 \
    --decode-branch \
    --decode-window 1 \
    --decode-sigma 1.1 \
    --debug-number 32 \
