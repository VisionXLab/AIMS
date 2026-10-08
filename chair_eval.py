import argparse
import json
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from llava.attention_qsteer_adaptive import llama_modify_qsteer_adaptive
from constants import INSTRUCTION_TEMPLATE, SYSTEM_MESSAGE
from eval_data_loader import COCODataSet
from llava.utils import disable_torch_init
from model_loader import ModelLoader
from tqdm import tqdm
from transformers.generation.logits_process import LogitsProcessorList

import time

def setup_seeds():
    seed = 927

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


parser = argparse.ArgumentParser(description="CHAIR evaluation on LVLMs.")
parser.add_argument("--model", type=str, help="model")
parser.add_argument(
    "--options",
    nargs="+",
    help="override some settings in the used config, the key-value pair "
    "in xxx=yyy format will be merged into config file (deprecate), "
    "change to --cfg-options instead.",
)
# TODO
parser.add_argument(
    "--data-path",
    type=str,
    default="/path/to/coco/val2014/",
    help="data path",
)
parser.add_argument("--batch-size", type=int, default=1)

parser.add_argument("--beam", type=int, default=1)
parser.add_argument("--sample", action="store_true")
parser.add_argument("--alpha", type=float, default=0.2)
parser.add_argument("--start-layer", type=int, default=2)
parser.add_argument("--end-layer", type=int, default=32)
parser.add_argument("--max-tokens", type=int, default=512)
parser.add_argument("--exp-tag", type=str, default="")
parser.add_argument("--debug-number", type=int, default=-1)
# qsteer adaptive
parser.add_argument("--use-qsteer-adaptive", action="store_true")

parser.add_argument("--ablation", type=str, default=None, help="set `adaptive-mode` to ablation on qsteer adaptive mode")

parser.add_argument("--decode-window", type=int, default=-1)

parser.add_argument("--visual-sigma", type=float, default=1.)
parser.add_argument("--prefill-sigma", type=float, default=1.)
parser.add_argument("--decode-sigma", type=float, default=1.)

parser.add_argument("--visual-branch", action="store_true")
parser.add_argument("--prefill-branch", action="store_true")
parser.add_argument("--decode-branch", action="store_true")
parser.add_argument("--adaptive-mode", type=str, default="cosine",
                    choices=["gaussian", "cosine", "mutual_info", "kl", "fixed"],
                    help="Adaptive weighting mode: gaussian(RBF), cosine, mutual_info, kl")
parser.add_argument("--recordw", action="store_true")
args = parser.parse_known_args()[0]

setup_seeds()

disable_torch_init()

model_loader = ModelLoader(args.model)

if args.sample:
    decode_method = "nucleus"
elif args.beam == 1:
    decode_method = "greedy"
else:
    decode_method = "beam"

base_dir_suffix_parts = [
    args.model,
    "_qsteer_adaptive" if args.use_qsteer_adaptive else "",
    # 
    f"_alpha_{args.alpha}" if args.use_qsteer_adaptive else "",
    f"_v_sgm{args.visual_sigma}" if args.use_qsteer_adaptive and args.visual_branch and args.adaptive_mode != "fixed" else "",
    f"_v" if args.use_qsteer_adaptive and args.visual_branch and args.adaptive_mode == "fixed" else "",
    f"_p_sgm{args.prefill_sigma}" if args.use_qsteer_adaptive and args.prefill_branch and args.adaptive_mode != "fixed" else "",
    f"_p" if args.use_qsteer_adaptive and args.prefill_branch and args.adaptive_mode == "fixed" else "",
    f"_dw{args.decode_window}_sgm{args.decode_sigma}" if args.use_qsteer_adaptive and args.decode_branch and args.adaptive_mode != "fixed" else "",
    f"_dw{args.decode_window}" if args.use_qsteer_adaptive and args.decode_branch and args.adaptive_mode == "fixed" else "",
    f"_{args.exp_tag}" if args.exp_tag else "",

]
base_dir_suffix = "".join(base_dir_suffix_parts)
if args.ablation == "adaptive_mode":
    args.ablation = args.adaptive_mode
if args.ablation:
    base_dir = f"./log_{args.model}/chair/{decode_method}/{args.model}/{args.ablation}/" + base_dir_suffix
else:
    base_dir = f"./msra_{args.model}/chair/{decode_method}/{args.model}/" + base_dir_suffix
print(base_dir)
if not os.path.exists(base_dir):
    os.makedirs(base_dir, exist_ok=True)

coco_dataset = COCODataSet(data_path=args.data_path, trans=model_loader.image_processor, debug_number=args.debug_number)
coco_loader = torch.utils.data.DataLoader(
    coco_dataset, batch_size=args.batch_size, shuffle=False, num_workers=32
) 

file_parts = [
    f"chair_eval_layers_{args.start_layer}-{args.end_layer}_tokens_{args.max_tokens}_bs_{args.batch_size}",
    "_sample" if args.sample else "",
    f"_beams_{args.beam}" if args.beam != 1 else "",
]

file_name = "".join(file_parts)
template = INSTRUCTION_TEMPLATE[args.model]
if args.model == "llava-1.5" or args.model == "shikra":
    template = SYSTEM_MESSAGE + template

# ===================== save all parameters ===================== 
args_dict = vars(args)
save_path = os.path.join(base_dir, "args.json")
with open(save_path, 'w', encoding='utf-8') as f:
    json.dump(args_dict, f, indent=4, ensure_ascii=False)

# Inference
# record_time
total_time, total_samples = 0.0, 0


for batch_id, data in tqdm(enumerate(coco_loader), total=len(coco_loader)):
    img_id = data["img_id"]
    image = data["image"]

    batch_size = img_id.shape[0]
    query = ["Please help me describe the image in detail."] * batch_size
    questions, kwargs = model_loader.prepare_inputs_for_model(template, query, image)
    if args.use_qsteer_adaptive:
        llama_modify_qsteer_adaptive(
            model_loader.llm_model,
            args.start_layer, 
            args.end_layer, 
            args.use_qsteer_adaptive, 
            args.alpha, 
            model_loader.img_start_idx,
            model_loader.img_end_idx,
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
    start = time.time()
    with torch.inference_mode():
        outputs = model_loader.llm_model.generate(
            do_sample=args.sample,
            max_new_tokens=args.max_tokens,
            use_cache=True,
            num_beams=args.beam,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
            **kwargs,
        )

    output_text = model_loader.decode(outputs)
    end = time.time()
    elapsed = end - start

    total_time += elapsed
    total_samples += args.batch_size

    for i in range(len(output_text)):
        with open(os.path.join(base_dir, file_name + ".jsonl"), "a") as f:
            json.dump({"image_id": int(img_id[i]), "caption": output_text[i]}, f)
            f.write("\n")

with open(os.path.join(base_dir, "time_" + file_name + ".txt"), "a") as f:
    f.write(f"Total inference time: {total_time:.4f} s\n")
    f.write(f"Total samples: {total_samples}\n")
    f.write(f"Avg time per image: {total_time / total_samples:.6f} s\n")