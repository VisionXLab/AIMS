#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

python qwen35vl_chair.py \
  --exp-tag "" \
  --model-path /root/autodl-tmp/Qwen3.5-9B \
  --data-path /root/autodl-tmp/aims_benchmarks/chair_coco/val2014_random500 \
  --beam 1 \
  --max-tokens 512 \
  --use-qsteer-adaptive \
  --bench-type chair \
  --alpha 0.32 \
  --visual-branch \
  --visual-sigma 1.5 \
  --prefill-branch \
  --prefill-sigma 1.2 \
  --decode-branch \
  --decode-window 1 \
  --decode-sigma 1. \
  --debug-number 32 \