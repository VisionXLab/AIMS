import math
import types
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
# ensure import repeat_kv 
from transformers.models.qwen3_5.modeling_qwen3_5 import repeat_kv, apply_rotary_pos_emb, eager_attention_forward
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.cache_utils import Cache, DynamicCache
from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
import torch.nn.functional as F
# Qwen3VLTextAttention

def _compute_weight_gaussian(q_current, target, sigma, head_dim):
    """
    RBF / Gaussian kernel: w = exp(-||q - target||^2 / (2 * sigma^2 * d_k))
    """
    dist = torch.sum((q_current - target) ** 2, dim=-1, keepdim=True)
    scale = 2 * (sigma ** 2) * head_dim
    w = torch.exp(-dist / scale)
    return w


def _compute_weight_cosine(q_current, target, sigma, head_dim):
    """
    Cosine similarity: w = (cos_sim + 1) / 2  to [0, 1]
    sigma : w = ((cos_sim + 1) / 2) ^ (1/sigma)
    """
    # q_current: [bsz, num_heads, 1, head_dim]
    # target:    [bsz, num_heads, 1, head_dim]
    cos_sim = F.cosine_similarity(q_current, target, dim=-1).unsqueeze(-1)  # [bsz, heads, 1, 1]
    w = (cos_sim + 1.0) / 2.0
    w = torch.pow(w, 1.0 / sigma)
    return w


def _compute_weight_mutual_info(q_current, target, sigma, head_dim):
    q_norm = F.normalize(q_current, dim=-1)
    t_norm = F.normalize(target, dim=-1)

    dot = torch.sum(q_norm * t_norm, dim=-1, keepdim=True)  # [bsz, heads, 1, 1]

    mi_approx = F.softplus(dot / sigma)
    w = torch.sigmoid(mi_approx)
    return w


def _compute_weight_kl(q_current, target, sigma, head_dim):

    q_dist = F.softmax(q_current / sigma, dim=-1)  # temperature scaling
    t_dist = F.softmax(target / sigma, dim=-1)
    # KL(q || t) = sum(q * log(q / t))
    eps = 1e-8
    kl = torch.sum(q_dist * torch.log((q_dist + eps) / (t_dist + eps)), dim=-1, keepdim=True)
    w = torch.exp(-kl)
    return w

def _compute_weight_fixed(q_current, target, sigma, head_dim):
    return 1.

def get_adaptive_qsteer(query_states, v_target, p_target, all_gen_keys, 
                         decode_window=-1, vb=True, pb=True, db=True, 
                         v_sgm=1.0, p_sgm=1.0, d_sgm=1.0,
                         adaptive_mode="gaussian", record_weights=False):
    """
    General Query Steering function supporting multiple adaptive weighting modes.

    Args:
        query_states: [bsz, num_heads, q_len, head_dim]
        v_target: [bsz, num_heads, 1, head_dim], mean of visual keys
        p_target: [bsz, num_heads, 1, head_dim], mean of prefill text keys
        all_gen_keys: [bsz, num_heads, num_g_tokens, head_dim]
        decode_window: Size of the decoding window; -1 indicates using all previously generated tokens
        vb/pb/db: Whether to enable the visual/prefill/decode branch
        v_sgm/p_sgm/d_sgm: Sigma/temperature parameter for each branch
        adaptive_mode: "gaussian" | "cosine" | "mutual_info" | "kl"

    Returns:
        steer_vec: [bsz, num_heads, 1, head_dim]
    """
    # import ipdb;ipdb.set_trace()
    q_current = query_states[:, :, -1:, :]
    head_dim = q_current.size(-1)

    weight_fn = ADAPTIVE_WEIGHT_FN[adaptive_mode]

    steer_vec = torch.zeros_like(q_current)
    w_v_val, w_p_val, w_g_val = 0. ,0., 0.
    if vb:
        
        w_v = weight_fn(q_current, v_target, v_sgm, head_dim)
        # w_v = 1. - w_v
        steer_vec += w_v * v_target
        if record_weights:
            w_v_val = w_v.mean().item()  # head average

    if pb:
        w_p = weight_fn(q_current, p_target, p_sgm, head_dim)
        steer_vec += w_p * p_target
        if record_weights:
            w_p_val = w_p.mean().item()  # head average

    if db and all_gen_keys.size(2) > 0:
        if decode_window == -1:
            g_target = all_gen_keys.mean(dim=2, keepdim=True)
        else:
            g_target = all_gen_keys[:, :, -decode_window:, :].mean(dim=2, keepdim=True)
        w_g = weight_fn(q_current, g_target, d_sgm, head_dim)
        steer_vec += w_g * g_target
        if record_weights:
            w_g_val = w_g.mean().item()  # head average
    if record_weights:
        return steer_vec, {"w_v": w_v_val, "w_p": w_p_val, "w_g": w_g_val}
    return steer_vec
# register all modes
ADAPTIVE_WEIGHT_FN = {
    "gaussian": _compute_weight_gaussian,
    "cosine": _compute_weight_cosine,
    "mutual_info": _compute_weight_mutual_info,
    "kl": _compute_weight_kl,
    "fixed": _compute_weight_fixed,
}

def qwen35_self_attn_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values: Cache | None = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    bsz, q_len, _ = hidden_states.size()
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states, gate = torch.chunk(
        self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
    )
    gate = gate.reshape(*input_shape, -1)

    query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    # Use RoPE
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    # Update KV Cache (only Once ! )
    if past_key_values is not None:
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
    # Process GQA ：Flatten Key/Value to Same Head number as Query.
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    #### qsteer
    if getattr(self, "use_qsteer_adaptive", False):
        # get prefill length
        if q_len > 1: 
            self.prefill_len = q_len
            # visual/prefill targets are calculated during prefilling
            #### Prefill visual keys
            # self.img_start_idx, self.img_end_idx has been added
            visual_keys = key_states[:, :, self.img_start_idx : self.img_end_idx, :]

            #### Prefill txt keys
            prefilltxt_indices = list(range(0, self.img_start_idx)) + list(range(self.img_end_idx, self.prefill_len))
            prefill_txt_keys = key_states[:, :, prefilltxt_indices, :]

            self.v_target = visual_keys.mean(dim=2, keepdim=True)
            self.p_target = prefill_txt_keys.mean(dim=2, keepdim=True)
        # Shape key_states [bsz, 16, seq_len, head_dim]

        #### All Generated keys
        all_gen_keys = key_states[:, :, self.prefill_len:-1, :]

        # Updata Query：Steering Q to Current visual tokens
        # alpha: stride
        _record = getattr(self, "record_weights", False)
        result = get_adaptive_qsteer(
            query_states, 
            self.v_target, 
            self.p_target, 
            all_gen_keys, 
            ## choose branch
            # v
            vb=self.visual_branch, 
            v_sgm=self.visual_sigma,
            # p
            pb=self.prefill_branch, 
            p_sgm=self.prefill_sigma,
            # d
            db=self.decode_branch,
            d_sgm=self.decode_sigma,
            decode_window=self.decode_window,
            # adaptive mode
            adaptive_mode=self.adaptive_mode,
            # record
            record_weights=_record,
            )
        if _record:
            steer_vec, weights_dict = result
            # layer--list
            if not hasattr(self, "_recorded_weights"):
                self._recorded_weights = []
            self._recorded_weights.append(weights_dict)
        else:
            steer_vec = result
        # steer query
        query_states[:, :, -1:, :] = (1. - self.alpha) * query_states[:, :, -1:, :] + self.alpha * steer_vec


    attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, eager_attention_forward
    )
    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = attn_output * torch.sigmoid(gate)

    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights

import inspect

def qwen35_modify_qsteer_adaptive(model, start_layer, end_layer, use_qsteer_adaptive, alpha,
                    img_start_idx, img_end_idx, decode_window, visual_branch, prefill_branch, decode_branch,
                    visual_sigma, prefill_sigma, decode_sigma, adaptive_mode="gaussian", record_weights=False):
    
    assert adaptive_mode in ADAPTIVE_WEIGHT_FN, \
        f"adaptive_mode must be one of {list(ADAPTIVE_WEIGHT_FN.keys())}, got '{adaptive_mode}'"

    for i in range(start_layer, end_layer):
        
        try:
            target_layer = model.model.language_model.layers[i].self_attn
            # inspect.getsourcefile(target_layer.__class__)
            # import ipdb;ipdb.set_trace()
            target_layer = model.model.language_model.layers[i].self_attn
            target_layer.use_qsteer_adaptive = use_qsteer_adaptive
            target_layer.alpha = alpha
            target_layer.img_start_idx = img_start_idx
            target_layer.img_end_idx = img_end_idx
            target_layer.decode_window = decode_window

            target_layer.visual_branch, target_layer.prefill_branch, target_layer.decode_branch = visual_branch, prefill_branch, decode_branch
            target_layer.visual_sigma, target_layer.prefill_sigma, target_layer.decode_sigma = visual_sigma, prefill_sigma, decode_sigma
            target_layer.adaptive_mode = adaptive_mode
            target_layer.record_weights = record_weights
            
            # option
            target_layer.forward = types.MethodType(qwen35_self_attn_forward, target_layer)
        except:
            target_layer = model.model.language_model.layers[i].linear_attn