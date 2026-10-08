import math
import types
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
# 确保导入 repeat_kv 
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import repeat_kv, apply_multimodal_rotary_pos_emb
import torch.nn.functional as F

def qwen2_5_vl_new_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[any] = None, 
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    # 1. 投影 QKV
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    # 2. 应用 RoPE
    if position_embeddings is None:
        # 兼容性处理：如果外部没传，内部算一下
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings

    query_states, key_states = apply_multimodal_rotary_pos_emb(
        query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
    )

    # 3. 更新 KV Cache (只更新一次！)
    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # 4. GQA 处理：把 Key/Value 展开到跟 Query 一样的 Head 数
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    # 5. 计算 Attention Weights
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling

    if attention_mask is not None:
        # 注意维度对齐，Qwen 的 mask 处理有时比较严苛
        if attention_mask.size()[-1] == attn_weights.size()[-1]:
            attn_weights = attn_weights + attention_mask

    # 6. Softmax & Output
    attn_probs = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

    #### 统计上下文域 attention weights
    if getattr(self, "use_qsteer", False):
        # attn_probs: [bsz, num_heads, q_len, seq_len]
        # 取最后一个 query token 对所有 key 的 attention 分布
        # aw_last: [bsz, num_heads, seq_len]
        aw_last = attn_probs[:, :, -1, :]

        if q_len > 1:  # prefill 阶段，记录 prefill_len
            self.prefill_len = q_len
        # --- Visual attention weights ---
        # 对 vision token 范围内的 attention weights
        num_vision_tokens = self.img_end_idx - self.img_start_idx
        aw_vision_slice = aw_last[:, :, self.img_start_idx:self.img_end_idx]  # [bsz, num_heads, num_vision_tokens]
        # token sum: 该区域获得的总注意力占比 [bsz, num_heads]
        aw_vision_heads_sum = aw_vision_slice.sum(dim=-1)
        # 记录 head mean of token sum [bsz, 1] → 该区域获得的总注意力占比的 head 均值
        self.history_aw_mean.append(aw_vision_heads_sum.mean(dim=1, keepdim=True).detach().cpu())

        # --- Prefill text attention weights ---
        # prefill 中非视觉部分的索引
        prefilltxt_indices = list(range(0, self.img_start_idx)) + list(range(self.img_end_idx, self.prefill_len))
        num_prefill_tokens = len(prefilltxt_indices)
        if num_prefill_tokens > 0:
            aw_prefill_slice = aw_last[:, :, prefilltxt_indices]  # [bsz, num_heads, num_prefill_tokens]
            aw_prefill_heads_sum = aw_prefill_slice.sum(dim=-1)   # [bsz, num_heads]
            self.prefilltxt_aw_mean.append(aw_prefill_heads_sum.mean(dim=1, keepdim=True).detach().cpu())
        else:
            self.prefilltxt_aw_mean.append(torch.zeros(bsz, 1).cpu())

        # --- Last decoded token attention weight ---
        # 当前 q 对上一个生成的 token 的 attention weight
        # 在 KV cache 模式下，key 序列中 seq_len-1 是当前 token 自身，seq_len-2 是上一个生成的 token
        seq_len = aw_last.size(-1)
        if q_len == 1 and seq_len > self.prefill_len:
            # seq_len-2 即上一个 decoded token 的位置
            aw_last_decoded_heads = aw_last[:, :, -1]  # [bsz, num_heads]
            self.lasttoken_aw_mean.append(aw_last_decoded_heads.mean(dim=1, keepdim=True).detach().cpu())
            aw_allgenerated_decoded_heads_sum = aw_last[:, :, self.prefill_len:].sum(dim=-1)  # [bsz, num_heads]
            self.allgenerated_aw_mean.append(aw_allgenerated_decoded_heads_sum.mean(dim=1, keepdim=True).detach().cpu())
        else:
            # Prefill 阶段或第一个生成 token（还没有上一个 decoded token）
            self.lasttoken_aw_mean.append(torch.zeros(bsz, 1).cpu())
            self.allgenerated_aw_mean.append(torch.zeros(bsz, 1).cpu())

    ####
    attn_output = torch.matmul(attn_probs, value_states)

    # 7. 维度恢复
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)

    return attn_output, (attn_probs if output_attentions else None)

def qwen_timestep(model, start_layer, end_layer, use_qsteer, 
                     img_start_idx, img_end_idx):
    # Qwen2.5-VL 的模型结构是 model.layers
    for i in range(start_layer, end_layer):
        target_layer = model.language_model.layers[i].self_attn
        
        # 注入参数
        target_layer.use_qsteer = use_qsteer
        target_layer.img_start_idx = img_start_idx
        target_layer.img_end_idx = img_end_idx
        
        # 注入必要属性（防止 new_forward 找不到变量）
        if not hasattr(target_layer, "num_key_value_groups"):
            target_layer.num_key_value_groups = target_layer.num_heads // target_layer.num_key_value_heads
        
        # 统计容器：attention weights
        # --- token sum 系列 (区域获得的总注意力占比) ---
        target_layer.history_aw_mean = []        # 所有 head 均值，对 vision tokens 的 token sum [bsz, 1] per step
        target_layer.prefilltxt_aw_mean = []     # 所有 head 均值，对 prefill text tokens 的 token sum [bsz, 1] per step
        target_layer.lasttoken_aw_mean = []      # 所有 head 均值，对已 decoded tokens 的 token sum [bsz, 1] per step
        target_layer.allgenerated_aw_mean = []      # 所有 head 均值，对已 decoded tokens 的 token sum [bsz, 1] per step
        # 替换方法
        target_layer.forward = types.MethodType(qwen2_5_vl_new_forward, target_layer)