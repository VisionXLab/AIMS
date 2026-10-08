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

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from qwen2_5vl.patch_attention_qsteer_adaptive import qwen_modify_qsteer_adaptive, collect_recorded_weights, reset_recorded_weights
from eval_data_loader import COCODataSet, MMEDataSet, load_json_or_jsonl

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

def run_chair(args, model, processor, base_dir, file_name):
    # ===================== data =====================
    coco_dataset = COCODataSet(
        data_path=args.data_path,
        trans=processor.image_processor,
        debug_number=args.debug_number,
    )

    coco_loader = torch.utils.data.DataLoader(
        coco_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=32
    )
    # ===================== save all parameters ===================== 
    args_dict = vars(args)
    save_path = os.path.join(base_dir, "args.json")
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(args_dict, f, indent=4, ensure_ascii=False)
    # ===================== inference =====================
    total_time, total_samples, total_generated_tokens = 0.0, 0, 0
    peak_memory_records = []
    # recordw: 收集所有图片的 steering weights
    # 结构: all_weights_records[layer_idx] -> list of per-token weight dicts (across all images)
    if args.recordw:
        xlsx_path = os.path.join(
            base_dir,
            file_name + "_steering_weights.xlsx"
        )

        # 防止重复实验时追加到旧文件
        if os.path.exists(xlsx_path):
            os.remove(xlsx_path)

        xlsx_initialized = False

    for batch_id, data in tqdm(enumerate(coco_loader), total=len(coco_loader)):

        if batch_id == args.num_images:
            break

        img_id = data["img_id"]
        image_path = data["image_path"]
        # image = data["image"]
        # ===== 构造 prompt =====
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path[0]},  # tensor
                    {"type": "text", "text": "Please describe the image in detail."},
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        # get single image token id

        input_ids = inputs["input_ids"][0]
        image_pad_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        img_mask = (input_ids == image_pad_id)
        img_indices = img_mask.nonzero(as_tuple=True)[0]
        img_start_idx = img_indices[0].item()
        img_end_idx = img_indices[-1].item() + 1

        # 
        if args.use_qsteer_adaptive:
            qwen_modify_qsteer_adaptive(
                model, 
                args.start_layer, 
                args.end_layer, 
                args.use_qsteer_adaptive, 
                args.alpha, 
                img_start_idx, 
                img_end_idx,
                args.decode_window,
                args.visual_branch,
                args.prefill_branch,
                args.decode_branch,
                args.visual_sigma,
                args.prefill_sigma,
                args.decode_sigma,
                args.adaptive_mode,
                record_weights=args.recordw
                )

        # ===== reset peak memory for current sample =====
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # ===== 推理 =====
        start = time.time()

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                **gen_kwargs,
            )

        end = time.time()
        total_time += (end - start)
        total_samples += 1
        if torch.cuda.is_available():
            peak_memory_gb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            peak_memory_records.append(peak_memory_gb)
        # ===== decode =====
        outputs_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs["input_ids"], outputs)
        ]

        output_text = processor.batch_decode(
            outputs_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        num_generated_tokens = outputs_trimmed[0].shape[0]
        total_generated_tokens += num_generated_tokens
        # ===== 保存 =====
        with open(os.path.join(base_dir, file_name + ".jsonl"), "a") as f:
            json.dump(
                {
                    "image_id": int(img_id[0]),
                    "caption": output_text
                },
                f
            )
            f.write("\n")
        # ===================== recordw: 每张图立即写入 xlsx =====================
        if args.recordw and args.use_qsteer_adaptive:

            img_weights = collect_recorded_weights(
                model,
                args.start_layer,
                args.end_layer
            )

            current_img_id = int(img_id[0])

            # 当前图片的数据
            current_image_records = []

            for layer_idx, token_weights_list in img_weights.items():

                for timestep, w in enumerate(token_weights_list):

                    # 找一个存在的 branch，确定 head 数量
                    head_values = None

                    if w.get("w_v_head") is not None:
                        head_values = w["w_v_head"]
                    elif w.get("w_p_head") is not None:
                        head_values = w["w_p_head"]
                    elif w.get("w_g_head") is not None:
                        head_values = w["w_g_head"]

                    if head_values is None:
                        continue

                    num_heads = len(head_values)

                    for head_idx in range(num_heads):

                        current_image_records.append({
                            "image_id": current_img_id,
                            "seq_len": num_generated_tokens,
                            "layer": layer_idx,
                            "timestep": timestep,
                            "head": head_idx,

                            "w_v": (
                                w["w_v_head"][head_idx]
                                if w.get("w_v_head") is not None
                                else np.nan
                            ),

                            "w_p": (
                                w["w_p_head"][head_idx]
                                if w.get("w_p_head") is not None
                                else np.nan
                            ),

                            "w_g": (
                                w["w_g_head"][head_idx]
                                if w.get("w_g_head") is not None
                                else np.nan
                            ),
                        })

            # 当前图片转 dataframe
            df_current = pd.DataFrame(current_image_records)

            # ==========================================================
            # 每张图片一个 sheet
            # ==========================================================
            sheet_name = f"img_{current_img_id}"

            if not xlsx_initialized:
                # 第一张图：创建 xlsx
                with pd.ExcelWriter(
                    xlsx_path,
                    engine="openpyxl",
                    mode="w"
                ) as writer:
                    df_current.to_excel(
                        writer,
                        sheet_name=sheet_name,
                        index=False
                    )

                xlsx_initialized = True

            else:
                # 后续图片：追加 sheet
                with pd.ExcelWriter(
                    xlsx_path,
                    engine="openpyxl",
                    mode="a",
                    if_sheet_exists="replace"
                ) as writer:
                    df_current.to_excel(
                        writer,
                        sheet_name=sheet_name,
                        index=False
                    )

            print(
                f"\n[recordw] image={current_img_id}, "
                f"seq_len={num_generated_tokens}, "
                f"rows={len(df_current)} "
                f"saved to sheet={sheet_name}"
            )

            # 清空模型中当前图片的记录
            reset_recorded_weights(
                model,
                args.start_layer,
                args.end_layer
            )

            del current_image_records
            del df_current
    return total_samples, total_time, peak_memory_records, total_generated_tokens

def run_mme(args, model, processor, base_dir, file_name):
    result_txt_path = os.path.join(base_dir, "results_txt")
    os.makedirs(result_txt_path, exist_ok=True)
    # data
    mme_dataset = MMEDataSet(
        data_path=args.data_path,
        trans=processor.image_processor,
        debug_number=args.debug_number,
    )
    mme_loader = torch.utils.data.DataLoader(
        mme_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=32
    )
    # ===================== save all parameters ===================== 
    args_dict = vars(args)
    save_path = os.path.join(base_dir, "args.json")
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(args_dict, f, indent=4, ensure_ascii=False)
    # ===================== inference =====================
    total_time, total_samples = 0.0, 0

    for batch_id, data in tqdm(enumerate(mme_loader), total=len(mme_loader)):

        image_path = data["image_path"][0]
        img_name = data["img_name"][0],
        dimension_name = data["dimension"][0],
        questions = data["questions"],   # usually 2
        answers = data["answers"],       # 
        

        for ques, ans in zip(questions[0], answers[0]):
            # image = data["image"]
            # =====  prompt =====
            ques, ans = ques[0], ans[0]

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},  # tensor
                        {"type": "text", "text": f"{ques}"},
                    ],
                }
            ]

            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            image_inputs, video_inputs = process_vision_info(messages)

            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            # get single image token id

            input_ids = inputs["input_ids"][0]
            image_pad_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
            img_mask = (input_ids == image_pad_id)
            img_indices = img_mask.nonzero(as_tuple=True)[0]
            img_start_idx = img_indices[0].item()
            img_end_idx = img_indices[-1].item() + 1
            # 
            if args.use_qsteer_adaptive:
                qwen_modify_qsteer_adaptive(
                    model, 
                    args.start_layer, 
                    args.end_layer, 
                    args.use_qsteer_adaptive, 
                    args.alpha, 
                    img_start_idx, 
                    img_end_idx,
                    args.decode_window,
                    args.visual_branch,
                    args.prefill_branch,
                    args.decode_branch,
                    args.visual_sigma,
                    args.prefill_sigma,
                    args.decode_sigma,
                    args.adaptive_mode,
                    record_weights=args.recordw
                    )

            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # ===== inference =====
            start = time.time()

            with torch.inference_mode():
                try:
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=args.max_tokens,
                        **gen_kwargs,
                    )
                except:
                    import ipdb;ipdb.set_trace() # 

            end = time.time()
            total_time += (end - start)
            total_samples += 1

            # ===== decode =====
            outputs_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs["input_ids"], outputs)
            ]

            output_text = processor.batch_decode(
                outputs_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            # ===== save =====
            with open(os.path.join(result_txt_path, dimension_name[0] + ".txt"), "a") as f_res:
                line = "\t".join([
                    img_name[0],
                    ques,
                    ans,
                    output_text.replace("\n", " ")
                ])
                f_res.write(line + "\n")
    return total_samples, total_time, total_generated_tokens
# ===================== args =====================
parser = argparse.ArgumentParser()

parser.add_argument("--exp-tag", type=str, help="Additional information about the experiment name")
parser.add_argument("--model-path", type=str, required=True)
parser.add_argument("--data-path", type=str, required=True)
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
parser.add_argument("--bench-type", type=str, required=True, choices=['chair', 'mme'])
parser.add_argument("--data-json", type=str, default="")
args = parser.parse_args() 
assert args.batch_size == 1  # Qwen VL 必须=1
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
base_dir = os.path.join(base_dir, "qwen2_5vl" + "".join(exp_suffix))
os.makedirs(base_dir, exist_ok=True)

file_name = f"{args.bench_type}_eval_{args.num_images}images_tokens_{args.max_tokens}"
# ===================== model =====================
print(f"Loading model: {args.model_path}")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    args.model_path,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    # device_map="auto",
)
model.to("cuda")
print(f"Use Qwen2.5vl Model")


processor = AutoProcessor.from_pretrained(args.model_path)
# sample strategies
gen_kwargs = get_gen_kwargs(decode_method)
print(f"Decoding config: {decode_method}\n{gen_kwargs}")
if args.bench_type == "chair":
    total_samples, total_time, peak_memory_records, total_generated_tokens = run_chair(args, model, processor, base_dir, file_name)
    # record
    with open(os.path.join(base_dir, "memory_" + file_name + ".txt"), "w") as f:
        f.write(f"Total samples: {len(peak_memory_records)}\n")
        f.write(f"Avg peak memory: {np.mean(peak_memory_records):.6f} MB\n")
        f.write(f"Max peak memory: {np.max(peak_memory_records):.6f} MB\n")
        f.write(f"Std peak memory: {np.std(peak_memory_records):.6f} MB\n")

elif args.bench_type == "mme":
    total_samples, total_time, total_generated_tokens = run_mme(args, model, processor, base_dir, file_name)
# ===================== time =====================
with open(os.path.join(base_dir, "time_" + file_name + ".txt"), "w") as f:
    f.write(f"Total inference time: {total_time:.4f} s\n")
    f.write(f"Total samples: {total_samples}\n")
    f.write(f"Avg time per image: {total_time / total_samples:.6f} s\n")
    f.write(f"Total generated tokens: {total_generated_tokens}\n")
    


