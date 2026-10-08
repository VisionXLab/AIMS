#!/bin/bash
export CUDA_VISIBLE_DEVIDES=0

python qwen35vl_chair.py \
  --exp-tag "test" \
  --model-path /root/autodl-tmp/Qwen3.5-9B \
  --data-path /root/autodl-tmp/aims_benchmarks/MME_Benchmark_release_version/MME_Benchmark \
  --beam 1 \
  --max-tokens 512 \
  --use-qsteer-adaptive \
  --bench-type mme \
  --alpha 0.3 \
  --visual-branch \
  --visual-sigma 1. \
  --prefill-branch \
  --prefill-sigma 1. \
  --decode-branch \
  --decode-window 1 \
  --decode-sigma 1. \
  --debug-number -1 \