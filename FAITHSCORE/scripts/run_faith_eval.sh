# ！/bin/bash
export CUDA_VISIBLE_DEVICES=0
python run.py \
    --image_dir /root/autodl-tmp/aims_benchmarks/faithscore/images \
    --answer_path /root/msra_workspace/code/AIMS/log/faithscore/greedy/cosine/qwen2_5vl_baseline_branch_debug64/faithscore_eval_1000images_tokens_512.jsonl \
    --model_path /root/autodl-tmp/iic/ofa_visual-question-answering_pretrain_large_en \
    --openai_key <your key> \
    --openai_url <your url> \
    --vem_type ofa \
    --openai_num_workers 10 \
    --ofa_batch_size 4 \
