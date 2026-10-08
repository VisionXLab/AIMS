'''
分段版 CHAIR 评估脚本

将每条生成的 caption 按 token 比例拆分为前段和后段，分别计算 CHAIR/Recall/F1/Len 等指标。

用法:
    python chair_segment.py --cap_file /path/to/captions.jsonl --split_ratio 0.6
    python chair_segment.py --cap_file /path/to/captions.jsonl --split_ratio 0.5 --save_path ./result.json

输出:
    前段 (front 0~ratio):   CHAIRs, CHAIRi, Recall, Precision, F1, Len
    后段 (back ratio~1.0):  CHAIRs, CHAIRi, Recall, Precision, F1, Len
    整体 (overall):         CHAIRs, CHAIRi, Recall, Precision, F1, Len
'''

import re
import os
import sys
import json
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

NLTK_DATA = "/output/liyan/msr/data/coco2014/msr/nltk_data"
if NLTK_DATA not in nltk.data.path:
    nltk.data.path.insert(0, NLTK_DATA)

_lemmatizer = WordNetLemmatizer()

import argparse
import tqdm
import pickle
from collections import defaultdict

# 导入原始 CHAIR 的基础设施
from chair import CHAIR, load_generated_captions, synonyms_txt


def split_caption_by_ratio(caption: str, ratio: float) -> tuple:
    """
    按 token 比例将 caption 拆分为前段和后段。
    
    Args:
        caption: 原始 caption 文本
        ratio: 分割比例，0~1 之间，前段占比
    
    Returns:
        (front_caption, back_caption): 前段文本和后段文本
    """
    words = nltk.word_tokenize(caption)
    if len(words) == 0:
        return "", ""
    
    split_idx = max(1, int(len(words) * ratio))  # 至少保留 1 个 token 在前段
    
    front_words = words[:split_idx]
    back_words = words[split_idx:]
    
    # 重新组装为文本（nltk tokenize 后用空格连接，标点前不加空格）
    front_caption = _detokenize(front_words)
    back_caption = _detokenize(back_words)
    
    return front_caption, back_caption


def _detokenize(tokens: list) -> str:
    """简单的 detokenize：将 token 列表重新组合为文本。"""
    if not tokens:
        return ""
    text = tokens[0]
    for t in tokens[1:]:
        # 标点符号前不加空格
        if t in {'.', ',', '!', '?', ';', ':', "'", '"', ')', ']', '}', "'s", "n't", "'re", "'ve", "'ll", "'m", "'d"}:
            text += t
        elif text and text[-1] in {'(', '[', '{', '"', "'"}:
            text += t
        else:
            text += ' ' + t
    return text


def compute_chair_segment(evaluator: CHAIR, cap_file: str, split_ratio: float,
                          image_id_key: str = "image_id", caption_key: str = "caption"):
    """
    分段计算 CHAIR 指标。
    
    对于每条 caption：
    1. 按 token 比例分割为 front / back 两段
    2. 分别对 front 和 back 计算 hallucination 指标
    3. 汇总报告
    """
    # 加载 captions
    caps, eval_imids = load_generated_captions(cap_file, image_id_key, caption_key)
    imid_to_objects = evaluator.imid_to_objects
    
    # 统计容器
    stats = {
        'front': {
            'num_caps': 0, 'num_hallucinated_caps': 0,
            'hallucinated_word_count': 0, 'coco_word_count': 0,
            'len_caps': 0,
            'num_recall_gt_objects': 0, 'num_gt_objects': 0, 'num_generated_objects': 0,
        },
        'back': {
            'num_caps': 0, 'num_hallucinated_caps': 0,
            'hallucinated_word_count': 0, 'coco_word_count': 0,
            'len_caps': 0,
            'num_recall_gt_objects': 0, 'num_gt_objects': 0, 'num_generated_objects': 0,
        },
        'overall': {
            'num_caps': 0, 'num_hallucinated_caps': 0,
            'hallucinated_word_count': 0, 'coco_word_count': 0,
            'len_caps': 0,
            'num_recall_gt_objects': 0, 'num_gt_objects': 0, 'num_generated_objects': 0,
        },
    }
    
    output = {'sentences': []}
    
    for i in tqdm.trange(len(caps), desc="Computing segmented CHAIR"):
        cap = caps[i]
        imid = eval_imids[i]
        gt_objects = imid_to_objects[imid]
        
        # 分割 caption
        front_cap, back_cap = split_caption_by_ratio(cap, split_ratio)
        
        # 对每个段落计算
        sentence_result = {
            'image_id': imid,
            'caption': cap,
            'front_caption': front_cap,
            'back_caption': back_cap,
            'front_metrics': {},
            'back_metrics': {},
            'overall_metrics': {},
        }
        
        for segment_name, segment_cap in [('front', front_cap), ('back', back_cap), ('overall', cap)]:
            if not segment_cap.strip():
                sentence_result[f'{segment_name}_metrics'] = {
                    'CHAIRs': 0, 'CHAIRi': 0, 'Recall': 0, 'Precision': 0, 'F1': 0, 'Len': 0
                }
                continue
            
            words, node_words, idxs, raw_words = evaluator.caption_to_words(segment_cap)
            
            s = stats[segment_name]
            s['coco_word_count'] += len(node_words)
            s['num_caps'] += 1
            s['len_caps'] += len(raw_words)
            s['num_gt_objects'] += len(gt_objects)
            s['num_generated_objects'] += len(set(node_words))
            
            hallucinated = False
            recall_gt_objects = set()
            hallucinated_words = []
            
            for word, node_word, idx in zip(words, node_words, idxs):
                if node_word not in gt_objects:
                    s['hallucinated_word_count'] += 1
                    hallucinated_words.append((word, node_word))
                    hallucinated = True
                else:
                    recall_gt_objects.add(node_word)
            
            if hallucinated:
                s['num_hallucinated_caps'] += 1
            
            s['num_recall_gt_objects'] += len(recall_gt_objects)
            
            # 单条指标
            chairs_i = len(hallucinated_words) / len(words) if len(words) > 0 else 0
            recall_i = len(recall_gt_objects) / len(gt_objects) if len(gt_objects) > 0 else 0
            precision_i = len(recall_gt_objects) / len(set(node_words)) if len(set(node_words)) > 0 else 0
            f1_i = 2 * recall_i * precision_i / (recall_i + precision_i) if (recall_i + precision_i) > 0 else 0
            
            sentence_result[f'{segment_name}_metrics'] = {
                'CHAIRs': int(hallucinated),
                'CHAIRi': chairs_i,
                'Recall': recall_i,
                'Precision': precision_i,
                'F1': f1_i,
                'Len': len(raw_words),
                'hallucinated_words': hallucinated_words,
                'generated_objects': list(set(node_words)),
            }
        
        output['sentences'].append(sentence_result)
    
    # 计算整体指标
    output['segment_overall_metrics'] = {}
    for segment_name in ['front', 'back', 'overall']:
        s = stats[segment_name]
        if s['num_caps'] == 0:
            output['segment_overall_metrics'][segment_name] = {
                'CHAIRs': 0, 'CHAIRi': 0, 'Recall': 0, 'Precision': 0, 'F1': 0, 'Len': 0, 'ObjectWordsNum': 0, 'HObjectWordsNum': 0
            }
            continue
        
        chair_s = s['num_hallucinated_caps'] / s['num_caps']
        chair_i = s['hallucinated_word_count'] / s['coco_word_count'] if s['coco_word_count'] > 0 else 0
        recall = s['num_recall_gt_objects'] / s['num_gt_objects'] if s['num_gt_objects'] > 0 else 0
        precision = s['num_recall_gt_objects'] / s['num_generated_objects'] if s['num_generated_objects'] > 0 else 0
        f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0
        avg_len = s['len_caps'] / s['num_caps']
        
        output['segment_overall_metrics'][segment_name] = {
            'CHAIRs': chair_s,
            'CHAIRi': chair_i,
            'Recall': recall,
            'Precision': precision,
            'F1': f1,
            'Len': avg_len,
            'ObjectWordsNum': s['coco_word_count'] / (100 * s['num_caps']),
            "HObjectWordsNum": s['hallucinated_word_count'] / (100 * s['num_caps'])
        }
    
    return output


def print_segment_metrics(output: dict, split_ratio: float):
    """打印分段指标。"""
    metrics = output['segment_overall_metrics']
    
    print(f"\n{'='*70}")
    print(f"  Segmented CHAIR Evaluation (split_ratio = {split_ratio})")
    print(f"{'='*70}")
    
    header = f"{'Metric':<12} {'Front (0~'+str(split_ratio)+')':<20} {'Back ('+str(split_ratio)+'~1.0)':<20} {'Overall':<20}"
    # 格式化 header
    front_label = f"Front (0~{split_ratio})"
    back_label = f"Back ({split_ratio}~1.0)"
    print(f"\n{'Metric':<12} {front_label:<22} {back_label:<22} {'Overall':<22}")
    print(f"{'-'*12} {'-'*22} {'-'*22} {'-'*22}")
    
    metric_names = ['CHAIRs', 'CHAIRi', 'Recall', 'Precision', 'F1', 'Len', 'ObjectWordsNum', 'HObjectWordsNum']
    for m in metric_names:
        front_val = metrics['front'].get(m, 0)
        back_val = metrics['back'].get(m, 0)
        overall_val = metrics['overall'].get(m, 0)
        
        if m == 'Len':
            print(f"{m:<12} {front_val:<22.2f} {back_val:<22.2f} {overall_val:<22.2f}")
        else:
            print(f"{m:<12} {front_val*100:<22.4f} {back_val*100:<22.4f} {overall_val*100:<22.4f}")
    
    print(f"{'='*70}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="分段版 CHAIR 评估")
    
    parser.add_argument("--cap_file", type=str, required=True,
                        help="path towards json or jsonl saving image ids and their captions.")
    parser.add_argument("--split_ratio", type=float, required=True,
                        help="分割比例 (0~1)，前段占比。例如 0.6 表示前 60%% tokens 为前段。")
    parser.add_argument("--image_id_key", type=str, default="image_id",
                        help="in each dict of cap_file, which key stores image id of coco.")
    parser.add_argument("--caption_key", type=str, default="caption",
                        help="in each dict of cap_file, which key stores caption of the image.")
    parser.add_argument("--cache", type=str, default="chair.pkl",
                        help="pre inited CHAIR evaluator object, for fast loading.")
    parser.add_argument("--coco_path", type=str, default='/root/jzq/Benchmarks/COCO/annotations',
                        help="only use for regenerating CHAIR evaluator object.")
    parser.add_argument("--save_path", type=str, default=None,
                        help="保存详细结果的路径 (json)。默认保存在 cap_file 同目录下。")
    args = parser.parse_args()
    
    assert 0 < args.split_ratio < 1, f"split_ratio must be in (0, 1), got {args.split_ratio}"
    
    # 加载 evaluator
    if args.cache and os.path.exists(args.cache):
        evaluator = pickle.load(open(args.cache, 'rb'))
        print(f"Loaded evaluator from cache: {args.cache}")
    else:
        print(f"Cache not set or not exist, building from scratch...")
        evaluator = CHAIR(args.coco_path)
        pickle.dump(evaluator, open(args.cache, 'wb'))
        print(f"Cached evaluator to: {args.cache}")
    
    # 计算分段指标
    output = compute_chair_segment(
        evaluator, args.cap_file, args.split_ratio,
        args.image_id_key, args.caption_key
    )
    
    # 打印结果
    print_segment_metrics(output, args.split_ratio)
    
    # 保存结果
    if args.save_path is None:
        args.save_path = os.path.join(
            os.path.dirname(args.cap_file),
            f"chair_segment_{args.split_ratio}.json"
        )
    
    # 保存时去掉 hallucinated_words 中的 tuple（JSON 不支持）
    for sent in output['sentences']:
        for key in ['front_metrics', 'back_metrics', 'overall_metrics']:
            if 'hallucinated_words' in sent.get(key, {}):
                sent[key]['hallucinated_words'] = [
                    list(w) for w in sent[key]['hallucinated_words']
                ]
    
    with open(args.save_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Detailed results saved to: {args.save_path}")
