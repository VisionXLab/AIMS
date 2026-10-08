import math
import types
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
# 确保导入 repeat_kv 
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import repeat_kv, apply_multimodal_rotary_pos_emb, eager_attention_forward
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

# ============================================================
# Adaptive Weight Computation: 4 modes
# ============================================================

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
    Cosine similarity: w = (cos_sim + 1) / 2
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

    eps = 1e-8
    kl = torch.sum(q_dist * torch.log((q_dist + eps) / (t_dist + eps)), dim=-1, keepdim=True)

    w = torch.exp(-kl)
    return w

def _compute_weight_fixed(q_current, target, sigma, head_dim):
    return 1.


# 
ADAPTIVE_WEIGHT_FN = {
    "gaussian": _compute_weight_gaussian,
    "cosine": _compute_weight_cosine,
    "mutual_info": _compute_weight_mutual_info,
    "kl": _compute_weight_kl,
    "fixed": _compute_weight_fixed,
}


def get_adaptive_qsteer(query_states, v_target, p_target, all_gen_keys, 
                         decode_window=-1, vb=True, pb=True, db=True, 
                         v_sgm=1.0, p_sgm=1.0, d_sgm=1.0,
                         adaptive_mode="gaussian", record_weights=False):

    q_current = query_states[:, :, -1:, :]
    head_dim = q_current.size(-1)

    weight_fn = ADAPTIVE_WEIGHT_FN[adaptive_mode]

    steer_vec = torch.zeros_like(q_current)
    w_v_val, w_p_val, w_g_val = 0. ,0., 0.
    w_v_head, w_p_head, w_g_head = None, None, None # head-timestep
    if vb:
        w_v = weight_fn(q_current, v_target, v_sgm, head_dim)
        # w_v = 1. - w_v
        steer_vec += w_v * v_target
        if record_weights:
            w_v_val = w_v.mean().item()  # head 
            w_v_head = (w_v[0, :, 0, 0].detach().float().cpu().tolist())

    if pb:
        w_p = weight_fn(q_current, p_target, p_sgm, head_dim)
        steer_vec += w_p * p_target
        if record_weights:
            w_p_val = w_p.mean().item()  # head 
            w_p_head = (w_p[0, :, 0, 0].detach().float().cpu().tolist())

    if db and all_gen_keys.size(2) > 0:
        if decode_window == -1:
            g_target = all_gen_keys.mean(dim=2, keepdim=True)
        else:
            g_target = all_gen_keys[:, :, -decode_window:, :].mean(dim=2, keepdim=True)
        w_g = weight_fn(q_current, g_target, d_sgm, head_dim)
        steer_vec += w_g * g_target
        if record_weights:
            w_g_val = w_g.mean().item()  # head 
            w_g_head = (w_g[0, :, 0, 0].detach().float().cpu().tolist())
    if record_weights:
        return steer_vec, {
            "w_v": w_v_val,
            "w_p": w_p_val,
            "w_g": w_g_val,

            "w_v_head": w_v_head,
            "w_p_head": w_p_head,
            "w_g_head": w_g_head,
        }

    return steer_vec

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
    input_shape = hidden_states.shape[:-1]
    # project QKV
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    # Use RoPE
    if position_embeddings is None:
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings

    query_states, key_states = apply_multimodal_rotary_pos_emb(
        query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
    )

    # Update KV Cache (only Once ! )
    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
    # ============================================================
    # Keep original K/V for Hugging Face attention interface.
    # HF sdpa_attention_forward will perform repeat_kv internally
    # when attention_mask is present.
    # ============================================================
    key_states_for_attn = key_states
    value_states_for_attn = value_states

    # ============================================================
    # Expanded K/V are only used by QSteer/AIMS to construct
    # head-wise prototypes aligned with the 16 query heads.
    # ============================================================
    key_states_for_steer = repeat_kv(key_states, self.num_key_value_groups)
    value_states_for_steer = repeat_kv(value_states, self.num_key_value_groups)

    # # Process GQA ：Flatten Key/Value to Same Head number as Query.
    # key_states = repeat_kv(key_states, self.num_key_value_groups)
    # value_states = repeat_kv(value_states, self.num_key_value_groups)

    # ============================================================
    # Batch-aware Adaptive QSteer
    # ============================================================
    if getattr(self, "use_qsteer_adaptive", False):
        if q_len > 1:
            # During the prefill process, q_len represents the unified sequence length after padding. The visual interval and the valid text interval must be calculated sample by sample, and the scalar index of batch[0] cannot be reused.
            self.prefill_len = q_len

            img_start_indices = self.img_start_idx if isinstance(self.img_start_idx, (list, tuple)) else [self.img_start_idx]
            img_end_indices = self.img_end_idx if isinstance(self.img_end_idx, (list, tuple)) else [self.img_end_idx]

            # In cases such as beam search, "generate" may expand the batch; here, the visual boundaries of each sample are replicated according to the expansion factor. In greedy/nucleus mode, "expand_factor" is set to 1.
            if len(img_start_indices) != bsz:
                if bsz % len(img_start_indices) != 0:
                    raise RuntimeError(f"Cannot align image ranges with attention batch: ranges={len(img_start_indices)}, bsz={bsz}")
                expand_factor = bsz // len(img_start_indices)
                img_start_indices = [x for x in img_start_indices for _ in range(expand_factor)]
                img_end_indices = [x for x in img_end_indices for _ in range(expand_factor)]

            visual_mask = torch.zeros((bsz, 1, q_len, 1), dtype=key_states_for_steer.dtype, device=key_states.device)
            for b, (img_start, img_end) in enumerate(zip(img_start_indices, img_end_indices)):
                img_start = max(0, min(int(img_start), q_len))
                img_end = max(img_start, min(int(img_end), q_len))
                visual_mask[b, :, img_start:img_end, :] = 1.0

            # The original 2D attention mask is saved by the main function after the processor and is used to exclude the left-padding token from the P prototype.
            prefill_attention_mask = getattr(self, "prefill_attention_mask", None)
            if prefill_attention_mask is None:
                valid_mask = torch.ones((bsz, 1, q_len, 1), dtype=key_states_for_steer.dtype, device=key_states_for_steer.device)
            else:
                valid_2d = prefill_attention_mask.to(device=key_states_for_steer.device)
                if valid_2d.shape[0] != bsz:
                    if bsz % valid_2d.shape[0] != 0:
                        raise RuntimeError(f"Cannot align prefill attention mask with batch: mask_bsz={valid_2d.shape[0]}, bsz={bsz}")
                    valid_2d = valid_2d.repeat_interleave(bsz // valid_2d.shape[0], dim=0)
                valid_mask = valid_2d[:, None, :q_len, None].to(dtype=key_states_for_steer.dtype)

            # V prototype: Each sample only averages its own image tokens.
            visual_denominator = visual_mask.sum(dim=2, keepdim=True).clamp_min(1.0)
            self.v_target = (key_states_for_steer[:, :, :q_len, :] * visual_mask).sum(dim=2, keepdim=True) / visual_denominator

            # P prototype: Average all valid non-visual prefill tokens and explicitly exclude batch padding.
            prefill_text_mask = valid_mask * (1.0 - visual_mask)
            prefill_text_denominator = prefill_text_mask.sum(dim=2, keepdim=True).clamp_min(1.0)
            self.p_target = (key_states_for_steer[:, :, :q_len, :] * prefill_text_mask).sum(dim=2, keepdim=True) / prefill_text_denominator

        # 
        all_gen_keys = key_states_for_steer[:, :, self.prefill_len:-1, :]
        _record = getattr(self, "record_weights", False)
        result = get_adaptive_qsteer(query_states, self.v_target, self.p_target, all_gen_keys, vb=self.visual_branch, v_sgm=self.visual_sigma, pb=self.prefill_branch, p_sgm=self.prefill_sigma, db=self.decode_branch, d_sgm=self.decode_sigma, decode_window=self.decode_window, adaptive_mode=self.adaptive_mode, record_weights=_record)

        if _record:
            steer_vec, weights_dict = result
            if not hasattr(self, "_recorded_weights"):
                self._recorded_weights = []
            self._recorded_weights.append(weights_dict)
        else:
            steer_vec = result

        query_states[:, :, -1:, :] = (1.0 - self.alpha) * query_states[:, :, -1:, :] + self.alpha * steer_vec

    attention_interface: Callable = ALL_ATTENTION_FUNCTIONS._global_mapping.get(
        self.config._attn_implementation, eager_attention_forward
    )
    # print("attn impl:", self.config._attn_implementation)
    # print("attention interface:", attention_interface)
    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states_for_attn,
        value_states_for_attn,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=self.sliding_window,
        position_ids=position_ids,  # pass positions for FA2
        **kwargs,
    )

    attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def qwen_modify_qsteer_adaptive(model, start_layer, end_layer, use_qsteer_adaptive, alpha,
                    img_start_idx, img_end_idx, decode_window, visual_branch, prefill_branch, decode_branch,
                    visual_sigma, prefill_sigma, decode_sigma, adaptive_mode="gaussian", record_weights=False, prefill_attention_mask=None):

    assert adaptive_mode in ADAPTIVE_WEIGHT_FN, \
        f"adaptive_mode must be one of {list(ADAPTIVE_WEIGHT_FN.keys())}, got '{adaptive_mode}'"
    # Qwen2.5-VL: model.layers
    for i in range(start_layer, end_layer):
        target_layer = model.language_model.layers[i].self_attn
        
        # 
        target_layer.use_qsteer_adaptive = use_qsteer_adaptive
        target_layer.alpha = alpha
        target_layer.img_start_idx = img_start_idx
        target_layer.img_end_idx = img_end_idx
        target_layer.decode_window = decode_window

        target_layer.visual_branch, target_layer.prefill_branch, target_layer.decode_branch = visual_branch, prefill_branch, decode_branch
        target_layer.visual_sigma, target_layer.prefill_sigma, target_layer.decode_sigma = visual_sigma, prefill_sigma, decode_sigma
        target_layer.adaptive_mode = adaptive_mode
        target_layer.record_weights = record_weights
        # Batch-aware prefill mask: [batch, padded_prefill_len]， P prototype 时排除 padding。
        target_layer.prefill_attention_mask = prefill_attention_mask
        
        # 
        if not hasattr(target_layer, "num_key_value_groups"):
            target_layer.num_key_value_groups = target_layer.num_heads // target_layer.num_key_value_heads
        
        # 
        target_layer.forward = types.MethodType(qwen2_5_vl_new_forward, target_layer)

def collect_recorded_weights(model, start_layer, end_layer):
    """
    reset recorded layer weights.
    
    Returns:
        dict: {layer_idx: [{"w_v": ..., "w_p": ..., "w_d": ...}, ...]}  per token
    """
    all_weights = {}
    for i in range(start_layer, end_layer):
        target_layer = model.language_model.layers[i].self_attn
        if hasattr(target_layer, "_recorded_weights") and target_layer._recorded_weights:
            all_weights[i] = target_layer._recorded_weights
    return all_weights


def reset_recorded_weights(model, start_layer, end_layer):
    """
    reset recorded layer weights.
    """
    for i in range(start_layer, end_layer):
        target_layer = model.language_model.layers[i].self_attn
        if hasattr(target_layer, "_recorded_weights"):
            target_layer._recorded_weights = []