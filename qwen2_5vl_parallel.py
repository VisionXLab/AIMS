import argparse
import json
import os
import time
import random

import pandas as pd
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from qwen_vl_utils import process_vision_info
from qwen2_5vl.patch_attention_qsteer_adaptive_parallel import collect_recorded_weights, reset_recorded_weights
from eval_data_loader import load_json_or_jsonl

os.environ["TRANSFORMERS_NO_FLASH_ATTENTION"] = "1"
def setup_seeds():
    seed = 927

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True

def get_gen_kwargs(decode_method):
    if decode_method == "greedy":
        gen_kwargs = dict(
            do_sample=False,
            num_beams=1,
        )

    elif decode_method == "beam":
        gen_kwargs = dict(
            do_sample=False,
            num_beams=5,
        )

    elif decode_method == "nucleus":
        gen_kwargs = dict(
            do_sample=True,
            temperature=1,
            top_p=0.9,
            top_k=50,
        )
    return gen_kwargs

def run_faith(args, model, processor, base_dir, file_name):
    # ===================== save all parameters =====================
    args_dict = vars(args)
    save_path = os.path.join(base_dir, "args.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(args_dict, f, indent=4, ensure_ascii=False)

    # ===================== inference stats =====================
    total_time, total_samples, total_generated_tokens = 0.0, 0, 0
    peak_memory_records = []

    # ===================== data =====================
    all_data = load_json_or_jsonl(args.data_json)
    max_samples = len(all_data)
    if args.num_images > 0:
        max_samples = min(max_samples, args.num_images)
    if args.debug_number != -1:
        max_samples = min(max_samples, args.debug_number)
    eval_data = all_data[:max_samples]

    # ===================== batch safety =====================
    # 。
    if args.recordw and args.batch_size != 1:
        raise ValueError("--recordw currently requires --batch-size 1.")

    image_pad_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    device = next(model.parameters()).device
    output_jsonl = os.path.join(base_dir, file_name + ".jsonl")
    if os.path.exists(output_jsonl):
        os.remove(output_jsonl)

    # ===================== batched inference =====================
    num_batches = (len(eval_data) + args.batch_size - 1) // args.batch_size
    for batch_start in tqdm(range(0, len(eval_data), args.batch_size), total=num_batches, desc=f"FaithScore batch={args.batch_size}"):
        batch_items = eval_data[batch_start:batch_start + args.batch_size]
        batch_messages = []
        batch_global_indices = list(range(batch_start, batch_start + len(batch_items)))

        for item in batch_items:
            img_id = int(item["id"])
            question = item["instruction"]
            image_path = os.path.join(args.data_path, f"COCO_val2014_{img_id:012d}.jpg")
            messages = [{"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": question}]}]
            batch_messages.append(messages)
        texts = processor.apply_chat_template(batch_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,)
        image_inputs, video_inputs = process_vision_info(batch_messages)
        inputs = processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")

        # ===================== per-sample visual ranges =====================
        img_start_indices, img_end_indices = [], []
        for b in range(inputs["input_ids"].shape[0]):
            img_indices = (inputs["input_ids"][b] == image_pad_id).nonzero(as_tuple=True)[0]
            if img_indices.numel() == 0:
                raise RuntimeError(f"No <|image_pad|> token found for FaithScore sample index {batch_global_indices[b]}.")
            img_start_indices.append(img_indices[0].item())
            img_end_indices.append(img_indices[-1].item() + 1)

        prefill_attention_mask = inputs["attention_mask"].clone()

        if args.use_qsteer_adaptive:
            if args.model_name == "qwen2_5vl":
                qwen_modify_qsteer_adaptive(model, args.start_layer, args.end_layer, args.use_qsteer_adaptive, args.alpha, img_start_indices, img_end_indices, args.decode_window, args.visual_branch, args.prefill_branch, args.decode_branch, args.visual_sigma, args.prefill_sigma, args.decode_sigma, args.adaptive_mode, record_weights=args.recordw, prefill_attention_mask=prefill_attention_mask)
            elif args.model_name == "qwen35":
                qwen35_modify_qsteer_adaptive_batch(model, args.start_layer, args.end_layer, args.use_qsteer_adaptive, args.alpha, img_start_indices, img_end_indices, args.decode_window, args.visual_branch, args.prefill_branch, args.decode_branch, args.visual_sigma, args.prefill_sigma, args.decode_sigma, args.adaptive_mode, record_weights=args.recordw)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # ===================== one generate call for the whole batch =====================
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.time()
        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=args.max_tokens, **gen_kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.time() - start

        total_time += elapsed
        total_samples += len(batch_items)
        if torch.cuda.is_available():
            peak_memory_gb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            peak_memory_records.append(peak_memory_gb)

        input_seq_len = inputs["input_ids"].shape[1]
        outputs_trimmed = outputs[:, input_seq_len:]
        output_texts = processor.batch_decode(outputs_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        # ===================== save results =====================
        with open(output_jsonl, "a", encoding="utf-8") as f:
            for local_idx, (global_idx, item, output_text) in enumerate(zip(batch_global_indices, batch_items, output_texts)):
                all_data[global_idx]["model_answer"] = output_text
                generated_token_count = outputs_trimmed[local_idx].shape[0]
                total_generated_tokens += generated_token_count
                json.dump(all_data[global_idx], f, ensure_ascii=False)
                f.write("\n")

        if args.recordw and args.use_qsteer_adaptive:
            img_weights = collect_recorded_weights(model, args.start_layer, args.end_layer)
            current_img_id = int(batch_items[0]["id"])
            current_image_records = []
            for layer_idx, token_weights_list in img_weights.items():
                for timestep, w in enumerate(token_weights_list):
                    head_values = w.get("w_v_head") if w.get("w_v_head") is not None else w.get("w_p_head") if w.get("w_p_head") is not None else w.get("w_g_head")
                    if head_values is None:
                        continue
                    for head_idx in range(len(head_values)):
                        current_image_records.append({"image_id": current_img_id, "seq_len": outputs_trimmed[0].shape[0], "layer": layer_idx, "timestep": timestep, "head": head_idx, "w_v": w["w_v_head"][head_idx] if w.get("w_v_head") is not None else np.nan, "w_p": w["w_p_head"][head_idx] if w.get("w_p_head") is not None else np.nan, "w_g": w["w_g_head"][head_idx] if w.get("w_g_head") is not None else np.nan})
            df_current = pd.DataFrame(current_image_records)
            xlsx_path = os.path.join(base_dir, file_name + "_steering_weights.xlsx")
            mode = "a" if os.path.exists(xlsx_path) else "w"
            writer_kwargs = {"engine": "openpyxl", "mode": mode}
            if mode == "a":
                writer_kwargs["if_sheet_exists"] = "replace"
            with pd.ExcelWriter(xlsx_path, **writer_kwargs) as writer:
                df_current.to_excel(writer, sheet_name=f"img_{current_img_id}", index=False)
            reset_recorded_weights(model, args.start_layer, args.end_layer)

    output_json = os.path.join(base_dir, file_name + ".json")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(all_data[:max_samples], f, indent=2, ensure_ascii=False)
    print(f"Saved batched FaithScore results to {output_json} and {output_jsonl}")
    return total_samples, total_time, peak_memory_records, total_generated_tokens

def run_amber(args, model, processor, base_dir, file_name):
    # ===================== save all parameters =====================
    args_dict = vars(args)
    save_path = os.path.join(base_dir, "args.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(args_dict, f, indent=4, ensure_ascii=False)

    # ===================== inference stats =====================
    total_time, total_samples, total_generated_tokens = 0.0, 0, 0
    peak_memory_records = []

    # ===================== data =====================
    all_data = load_json_or_jsonl(args.data_json)
    max_samples = len(all_data)
    if args.num_images > 0:
        max_samples = min(max_samples, args.num_images)
    if args.debug_number != -1:
        max_samples = min(max_samples, args.debug_number)
    eval_data = all_data[:max_samples]

    # ===================== batch safety =====================
    if args.recordw and args.batch_size != 1:
        raise ValueError("--recordw currently requires --batch-size 1.")

    image_pad_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    device = next(model.parameters()).device
    output_jsonl = os.path.join(base_dir, file_name + ".jsonl")
    if os.path.exists(output_jsonl):
        os.remove(output_jsonl)

    # ===================== batched inference =====================
    num_batches = (len(eval_data) + args.batch_size - 1) // args.batch_size
    for batch_start in tqdm(range(0, len(eval_data), args.batch_size), total=num_batches, desc=f"Amber batch={args.batch_size}"):
        batch_items = eval_data[batch_start:batch_start + args.batch_size]
        batch_messages = []
        batch_global_indices = list(range(batch_start, batch_start + len(batch_items)))

        for item in batch_items:
            img_id = int(item["id"])
            question = item["query"]
            image_name = item["image"]
            image_path = os.path.join(args.data_path, image_name)
            messages = [{"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": question}]}]
            batch_messages.append(messages)

        texts = processor.apply_chat_template(batch_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,)
        image_inputs, video_inputs = process_vision_info(batch_messages)
        inputs = processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")

        # ===================== per-sample visual ranges =====================
        img_start_indices, img_end_indices = [], []
        for b in range(inputs["input_ids"].shape[0]):
            img_indices = (inputs["input_ids"][b] == image_pad_id).nonzero(as_tuple=True)[0]
            if img_indices.numel() == 0:
                raise RuntimeError(f"No <|image_pad|> token found for FaithScore sample index {batch_global_indices[b]}.")
            img_start_indices.append(img_indices[0].item())
            img_end_indices.append(img_indices[-1].item() + 1)

        prefill_attention_mask = inputs["attention_mask"].clone()
        if args.use_qsteer_adaptive:
            if args.model_name == "qwen2_5vl":
                qwen_modify_qsteer_adaptive(model, args.start_layer, args.end_layer, args.use_qsteer_adaptive, args.alpha, img_start_indices, img_end_indices, args.decode_window, args.visual_branch, args.prefill_branch, args.decode_branch, args.visual_sigma, args.prefill_sigma, args.decode_sigma, args.adaptive_mode, record_weights=args.recordw, prefill_attention_mask=prefill_attention_mask)
            elif args.model_name == "qwen35":
                qwen35_modify_qsteer_adaptive_batch(model, args.start_layer, args.end_layer, args.use_qsteer_adaptive, args.alpha, img_start_indices, img_end_indices, args.decode_window, args.visual_branch, args.prefill_branch, args.decode_branch, args.visual_sigma, args.prefill_sigma, args.decode_sigma, args.adaptive_mode, record_weights=args.recordw)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # ===================== one generate call for the whole batch =====================
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.time()
        # try:
        #     with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=args.max_tokens, **gen_kwargs)
        # except:
        #     import ipdb;ipdb.set_trace()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.time() - start

        total_time += elapsed
        total_samples += len(batch_items)
        if torch.cuda.is_available():
            peak_memory_gb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            peak_memory_records.append(peak_memory_gb)

        input_seq_len = inputs["input_ids"].shape[1]
        outputs_trimmed = outputs[:, input_seq_len:]
        output_texts = processor.batch_decode(outputs_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        # ===================== save results =====================
        with open(output_jsonl, "a", encoding="utf-8") as f:
            for local_idx, (global_idx, item, output_text) in enumerate(zip(batch_global_indices, batch_items, output_texts)):
                all_data[global_idx]["response"] = output_text
                generated_token_count = outputs_trimmed[local_idx].shape[0]
                total_generated_tokens += generated_token_count
                json.dump(all_data[global_idx], f, ensure_ascii=False)
                f.write("\n")

        if args.recordw and args.use_qsteer_adaptive:
            img_weights = collect_recorded_weights(model, args.start_layer, args.end_layer)
            current_img_id = int(batch_items[0]["id"])
            current_image_records = []
            for layer_idx, token_weights_list in img_weights.items():
                for timestep, w in enumerate(token_weights_list):
                    head_values = w.get("w_v_head") if w.get("w_v_head") is not None else w.get("w_p_head") if w.get("w_p_head") is not None else w.get("w_g_head")
                    if head_values is None:
                        continue
                    for head_idx in range(len(head_values)):
                        current_image_records.append({"image_id": current_img_id, "seq_len": outputs_trimmed[0].shape[0], "layer": layer_idx, "timestep": timestep, "head": head_idx, "w_v": w["w_v_head"][head_idx] if w.get("w_v_head") is not None else np.nan, "w_p": w["w_p_head"][head_idx] if w.get("w_p_head") is not None else np.nan, "w_g": w["w_g_head"][head_idx] if w.get("w_g_head") is not None else np.nan})
            df_current = pd.DataFrame(current_image_records)
            xlsx_path = os.path.join(base_dir, file_name + "_steering_weights.xlsx")
            mode = "a" if os.path.exists(xlsx_path) else "w"
            writer_kwargs = {"engine": "openpyxl", "mode": mode}
            if mode == "a":
                writer_kwargs["if_sheet_exists"] = "replace"
            with pd.ExcelWriter(xlsx_path, **writer_kwargs) as writer:
                df_current.to_excel(writer, sheet_name=f"img_{current_img_id}", index=False)
            reset_recorded_weights(model, args.start_layer, args.end_layer)

    output_json = os.path.join(base_dir, file_name + ".json")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(all_data[:max_samples], f, indent=2, ensure_ascii=False)
    print(f"Saved batched AMBER results to {output_json} and {output_jsonl}")
    return total_samples, total_time, peak_memory_records, total_generated_tokens
# ===================== args =====================
parser = argparse.ArgumentParser()

parser.add_argument("--exp-tag", type=str, help="Additional information about the experiment name")
parser.add_argument("--model-path", type=str, required=True)
parser.add_argument("--data-path", type=str, required=True)
parser.add_argument("--model-name", type=str, required=True, choices=["qwen2_5vl", "qwen35"])
parser.add_argument("--batch-size", type=int, default=1)
parser.add_argument("--max-tokens", type=int, default=512)
parser.add_argument("--num-images", type=int, default=500)
parser.add_argument("--beam", type=int, default=1)
parser.add_argument("--sample", action="store_true")
parser.add_argument("--alpha", type=float, default=0.2)
parser.add_argument("--start-layer", type=int, default=2)
parser.add_argument("--end-layer", type=int, default=32)
parser.add_argument("--debug-number", type=int, default=32)
## llava1.5-7b 32decoder layers, [startlayer, endlayer] = [2, 32]
## qwen2.5vl-3b 35decoder layers, [startlayer, endlayer] = [2, 35]
## qwen3vl-2b 27decoder layers, [startlayer, endlayer] = [2, 27]

# q steer
parser.add_argument("--use-qsteer-adaptive", action="store_true")
parser.add_argument("--decode-window", type=int, default=-1)
parser.add_argument("--ablation", type=str, default=None)

parser.add_argument("--visual-sigma", type=float, default=1.)
parser.add_argument("--prefill-sigma", type=float, default=1.)
parser.add_argument("--decode-sigma", type=float, default=1.)

parser.add_argument("--visual-branch", action="store_true")
parser.add_argument("--prefill-branch", action="store_true")
parser.add_argument("--decode-branch", action="store_true")
parser.add_argument("--adaptive-mode", type=str, default="cosine",
                    choices=["gaussian", "cosine", "mutual_info", "kl", "fixed"],
                    help="Adaptive weighting mode: gaussian(RBF), cosine, mutual_info, kl")
parser.add_argument("--recordw", action="store_true",
                    help="Record steering weights for each branch (v/p/d) per layer per token, save to xlsx")

# bench type 
parser.add_argument("--bench-type", type=str, required=True, choices=['chair', 'mmhal', 'faithscore', "amber"])
parser.add_argument("--data-json", type=str, default="")
args = parser.parse_args()

# FaithScore + batch-aware adaptive QSteer 支持 batch_size > 1；CHAIR/MMHAL 以及其他旧 patch 仍建议 batch_size=1。
setup_seeds()
# ===================== log =====================
if args.sample:
    decode_method = "nucleus"
elif args.beam == 1:
    decode_method = "greedy"
else:
    decode_method = "beam"

exp_suffix = [
    f"_start{args.start_layer}_end_{args.end_layer}_alpha{args.alpha}" if args.use_qsteer_adaptive else "",
    f"_{args.exp_tag}",
    f"_{args.adaptive_mode}" if args.use_qsteer_adaptive else "",
    f"_branch",
    f"_v_sgm{args.visual_sigma}" if args.use_qsteer_adaptive and args.visual_branch and args.adaptive_mode != "fixed" else "",
    f"_v" if args.use_qsteer_adaptive and args.visual_branch and args.adaptive_mode == "fixed" else "",
    f"_p_sgm{args.prefill_sigma}" if args.use_qsteer_adaptive and args.prefill_branch and args.adaptive_mode != "fixed" else "",
    f"_p" if args.use_qsteer_adaptive and args.prefill_branch and args.adaptive_mode == "fixed" else "",
    f"_dw{args.decode_window}_sgm{args.decode_sigma}" if args.use_qsteer_adaptive and args.decode_branch and args.adaptive_mode != "fixed" else "",
    f"_dw{args.decode_window}" if args.use_qsteer_adaptive and args.decode_branch and args.adaptive_mode == "fixed" else "",
    f"_debug{args.debug_number}" if args.debug_number != -1 else ""
]
base_dir = f"./log/{args.bench_type}/{decode_method}/"
args.ablation = args.adaptive_mode
if args.ablation is not None:
    base_dir = os.path.join(base_dir, args.ablation)
base_dir = os.path.join(base_dir, args.model_name + "".join(exp_suffix))
os.makedirs(base_dir, exist_ok=True)

file_name = f"{args.bench_type}_eval_{args.num_images}images_tokens_{args.max_tokens}"
# ===================== model =====================
print(f"Loading model: {args.model_path}")
if args.model_name == "qwen2_5vl":
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen2_5vl.patch_attention_qsteer_adaptive_parallel import qwen_modify_qsteer_adaptive
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    print(f"Use Qwen2.5vl Model")

    processor = AutoProcessor.from_pretrained(args.model_path)
    processor.tokenizer.padding_side = "left"
elif args.model_name == "qwen35":
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from qwen3_5.patch_attention_qsteer import qwen35_modify_qsteer_adaptive
    from qwen3_5.patch_attention_qsteer_batch import qwen35_modify_qsteer_adaptive_batch
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        # device_map="auto",
        trust_remote_code=True,
    )
    print("Use Qwen3.5 Model")
    model.to('cuda')
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    processor.tokenizer.padding_side = "left"
# sample strategies
gen_kwargs = get_gen_kwargs(decode_method)
print(f"Decoding config: {decode_method}\n{gen_kwargs}")
if args.bench_type == "faithscore":
    total_samples, total_time, peak_memory_records, total_generated_tokens = run_faith(args, model, processor, base_dir, file_name)
elif args.bench_type == "amber":
    total_samples, total_time, peak_memory_records, total_generated_tokens = run_amber(args, model, processor, base_dir, file_name)
# ===================== time =====================
with open(os.path.join(base_dir, "time_" + file_name + ".txt"), "w") as f:
    f.write(f"Total inference time: {total_time:.4f} s\n")
    f.write(f"Total samples: {total_samples}\n")
    f.write(f"Avg time per image: {total_time / total_samples:.6f} s\n")
    f.write(f"Throughput: {total_samples / total_time:.6f} samples/s\n")
    f.write(f"Total generated tokens: {total_generated_tokens}\n")
    

with open(os.path.join(base_dir, "memory_" + file_name + ".txt"), "w") as f:
    f.write(f"Total samples: {len(peak_memory_records)}\n")
    f.write(f"Avg peak memory: {np.mean(peak_memory_records):.6f} MB\n")
    f.write(f"Max peak memory: {np.max(peak_memory_records):.6f} MB\n")
    f.write(f"Std peak memory: {np.std(peak_memory_records):.6f} MB\n")

