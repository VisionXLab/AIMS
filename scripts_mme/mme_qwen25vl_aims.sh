#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

python qwen2_5vl_chair.py \
    --exp-tag "test" \
    --model-path /root/autodl-tmp/Qwen2.5-VL-3B-Instruct \
    --data-path /root/autodl-tmp/aims_benchmarks/MME_Benchmark_release_version/MME_Benchmark \
    --use-qsteer-adaptive \
    --start-layer 2 \
    --end-layer 35 \
    --alpha 0.01 \
    --beam 1 \
    --visual-branch \
    --visual-sigma 1. \
    --prefill-branch \
    --prefill-sigma 1. \
    --decode-branch \
    --decode-window 1 \
    --decode-sigma 1. \
    --debug-number -1 \
    --bench-type mme \