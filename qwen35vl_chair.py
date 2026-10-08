import argparse
import json
import os
import time
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm

# from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, AutoModelForImageTextToText
from qwen3_5.patch_attention_qsteer import qwen35_modify_qsteer_adaptive
from eval_data_loader import COCODataSet, MMEDataSet

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

def run_chair(args, processor, model):
    # ===================== data =====================
    coco_dataset = COCODataSet(
        data_path=args.data_path,
        trans=processor.image_processor,
        debug_number=args.debug_number
    )

    coco_loader = torch.utils.data.DataLoader(
        coco_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=32
    )

    # ===================== inference =====================
    total_time, total_samples = 0.0, 0

    for batch_id, data in tqdm(enumerate(coco_loader), total=len(coco_loader)):

        img_id = data["img_id"]
        image_path = data["image_path"]
        # image = data["image"]
        # ===== 构造 prompt =====
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path[0]},  # tensor
                    {"type": "text", "text": "Please describe the image in detail. Only output the final image description. Do not include reasoning steps."},
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False # 关闭 thinking
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

        if args.use_qsteer_adaptive:
            qwen35_modify_qsteer_adaptive(
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
    return total_time, total_samples

def run_mme(args, processor, model, base_dir):
    result_txt_path = os.path.join(base_dir, "results_txt")
    os.makedirs(result_txt_path, exist_ok=True)
    # ===================== data =====================
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
    # ===================== inference =====================
    total_time, total_samples = 0.0, 0

    for batch_id, data in tqdm(enumerate(mme_loader), total=len(mme_loader)):

        image_path = data["image_path"][0]
        img_name = data["img_name"][0],
        dimension_name = data["dimension"][0],
        questions = data["questions"],   # 长度一般为2
        answers = data["answers"],       # 对应GT
        

        for ques, ans in zip(questions[0], answers[0]):
            # image = data["image"]
            # ===== 构造 prompt =====
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
                enable_thinking=False
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
                qwen35_modify_qsteer_adaptive(
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

            # ===== 保存 =====
            with open(os.path.join(result_txt_path, dimension_name[0] + ".txt"), "a") as f_res:
                line = "\t".join([
                    img_name[0],
                    ques,
                    ans,
                    output_text.replace("\n", " ")
                ])
                f_res.write(line + "\n")
    return total_time, total_samples
# ===================== args =====================
parser = argparse.ArgumentParser()

parser.add_argument("--exp-tag", type=str, help="Additional information about the experiment name")
parser.add_argument("--model-path", type=str, required=True)
parser.add_argument("--data-path", type=str, required=True)
parser.add_argument("--batch-size", type=int, default=1)
parser.add_argument("--max-tokens", type=int, default=512)
parser.add_argument("--debug-number", type=int, default=-1)
parser.add_argument("--beam", type=int, default=1)
parser.add_argument("--sample", action="store_true")
parser.add_argument("--alpha", type=float, default=0.2)
parser.add_argument("--start-layer", type=int, default=2)
parser.add_argument("--end-layer", type=int, default=32)
## llava1.5-7b 32decoder layers, [startlayer, endlayer] = [2, 32]
## qwen2.5vl-3b 35decoder layers, [startlayer, endlayer] = [2, 35]
## qwen3vl-2b 27decoder layers, [startlayer, endlayer] = [2, 27]

# q steer
parser.add_argument("--use-qsteer-adaptive", action="store_true")
parser.add_argument("--decode-window", type=int, default=-1)
parser.add_argument("--ablation", type=str, default=None)
parser.add_argument("--bench-type", type=str, required=True, choices=["chair", "mme"])

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
    f"_branch" if args.use_qsteer_adaptive else "",
    f"_v_sgm{args.visual_sigma}" if args.use_qsteer_adaptive and args.visual_branch and args.adaptive_mode != "fixed" else "",
    f"_v" if args.use_qsteer_adaptive and args.visual_branch and args.adaptive_mode == "fixed" else "",
    f"_p_sgm{args.prefill_sigma}" if args.use_qsteer_adaptive and args.prefill_branch and args.adaptive_mode != "fixed" else "",
    f"_p" if args.use_qsteer_adaptive and args.prefill_branch and args.adaptive_mode == "fixed" else "",
    f"_dw{args.decode_window}_sgm{args.decode_sigma}" if args.use_qsteer_adaptive and args.decode_branch and args.adaptive_mode != "fixed" else "",
    f"_dw{args.decode_window}" if args.use_qsteer_adaptive and args.decode_branch and args.adaptive_mode == "fixed" else "",
    f"_debug{args.debug_number}" if args.debug_number != -1 else ""
]

qsteer_adaptive_type = f"{args.adaptive_mode}/" if args.use_qsteer_adaptive else ""
base_dir = f"./log_qwen35/{args.bench_type}/{decode_method}/{qsteer_adaptive_type}" + f"qwen35vl_{decode_method}" + "".join(exp_suffix)
os.makedirs(base_dir, exist_ok=True)

file_name = f"{args.bench_type}_eval_images_tokens_{args.max_tokens}"
# ===================== model =====================
print(f"Loading model: {args.model_path}")
model = AutoModelForImageTextToText.from_pretrained(
    args.model_path,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    # device_map="auto",
    trust_remote_code=True,
)
model.to("cuda")
print("Use Qwen3.5 Model")

processor = AutoProcessor.from_pretrained(
    args.model_path,
    trust_remote_code=True,
)
# sample strategies
gen_kwargs = get_gen_kwargs(decode_method)
print(f"Decoding config: {decode_method}\n{gen_kwargs}")

# ===================== save all parameters ===================== 
args_dict = vars(args)
save_path = os.path.join(base_dir, "args.json")
with open(save_path, 'w', encoding='utf-8') as f:
    json.dump(args_dict, f, indent=4, ensure_ascii=False)

if args.bench_type == "chair":
    total_time, total_samples = run_chair(args, processor, model)
elif args.bench_type == "mme":
    total_time, total_samples = run_mme(args, processor, model, base_dir)


# ===================== time =====================
with open(os.path.join(base_dir, "time_" + file_name + ".txt"), "w") as f:
    f.write(f"Total inference time: {total_time:.4f} s\n")
    f.write(f"Total samples: {total_samples}\n")
    f.write(f"Avg time per image: {total_time / total_samples:.6f} s\n")