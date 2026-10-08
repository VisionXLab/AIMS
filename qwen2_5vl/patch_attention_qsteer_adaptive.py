import math
import types
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
# Make sure to import repeat_kv
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import repeat_kv, apply_multimodal_rotary_pos_emb, eager_attention_forward
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

# ============================================================
# Adaptive Weight Computation: 4 modes
# ============================================================

def _compute_weight_gaussian(q_current, target, sigma, head_dim):
    """
    RBF / Gaussian kernel: w = exp(-||q - target||^2 / (2 * sigma^2 * d_k))
    Larger sigma -> more uniform weights (approaching 1); smaller sigma -> effective only at short distances
    """
    dist = torch.sum((q_current - target) ** 2, dim=-1, keepdim=True)
    scale = 2 * (sigma ** 2) * head_dim
    w = torch.exp(-dist / scale)
    return w


def _compute_weight_cosine(q_current, target, sigma, head_dim):
    """
    Cosine similarity: w = (cos_sim + 1) / 2, mapped to [0, 1]
    Then apply temperature scaling using sigma: w = ((cos_sim + 1) / 2) ^ (1/sigma)
    Larger sigma -> more uniform weights; smaller sigma -> significant weights only for high similarity
    """
    # q_current: [bsz, num_heads, 1, head_dim]
    # target:    [bsz, num_heads, 1, head_dim]
    cos_sim = F.cosine_similarity(q_current, target, dim=-1).unsqueeze(-1)  # [bsz, heads, 1, 1]
    # Map to [0, 1]
    w = (cos_sim + 1.0) / 2.0
    # Temperature scaling: sigma is used as the temperature parameter
    w = torch.pow(w, 1.0 / sigma)
    return w


def _compute_weight_mutual_info(q_current, target, sigma, head_dim):
    """
    Mutual-information-based approximation: treat q and target as high-dimensional vectors,
    and use their normalized inner product as a proxy for statistical dependence.
    
    Specifically:
    1. Apply L2 normalization to q_current and target
    2. Compute the dot product (i.e., cosine similarity)
    3. Use softplus to ensure non-negativity: MI_approx = softplus(dot / sigma)
    4. Apply sigmoid to map the value to [0, 1]
    
    Larger sigma -> smoother/more uniform weights; smaller sigma -> more sensitive to the degree of dependence
    """
    q_norm = F.normalize(q_current, dim=-1)
    t_norm = F.normalize(target, dim=-1)
    # Dot product -> proxy for statistical dependence
    dot = torch.sum(q_norm * t_norm, dim=-1, keepdim=True)  # [bsz, heads, 1, 1]
    # softplus + sigmoid mapping
    mi_approx = F.softplus(dot / sigma)
    w = torch.sigmoid(mi_approx)
    return w


def _compute_weight_kl(q_current, target, sigma, head_dim):
    """
    KL-divergence-based weighting: treat q and target as normalized distributions,
    and compute KL(q || target), where a smaller divergence yields a larger weight.
    
    Specifically:
    1. Apply softmax to q_current and target along head_dim -> treat them as probability distributions
    2. Compute KL divergence (element-wise)
    3. w = exp(-KL / sigma)
    
    Larger sigma -> greater tolerance for KL divergence; smaller sigma -> non-negligible weights only when the distributions are close
    """
    # Normalize into probability distributions using softmax
    q_dist = F.softmax(q_current / sigma, dim=-1)  # temperature scaling
    t_dist = F.softmax(target / sigma, dim=-1)
    # KL(q || t) = sum(q * log(q / t))
    # Add eps to prevent log(0)
    eps = 1e-8
    kl = torch.sum(q_dist * torch.log((q_dist + eps) / (t_dist + eps)), dim=-1, keepdim=True)
    # KL >= 0; a smaller value indicates higher similarity
    w = torch.exp(-kl)
    return w

def _compute_weight_fixed(q_current, target, sigma, head_dim):
    return 1.


# Register all modes
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
    q_current = query_states[:, :, -1:, :]
    head_dim = q_current.size(-1)

    weight_fn = ADAPTIVE_WEIGHT_FN[adaptive_mode]

    steer_vec = torch.zeros_like(q_current)
    w_v_val, w_p_val, w_g_val = 0. ,0., 0.
    w_v_head, w_p_head, w_g_head = None, None, None # Record per-head, per-timestep weights
    if vb:
        w_v = weight_fn(q_current, v_target, v_sgm, head_dim)
        # w_v = 1. - w_v
        steer_vec += w_v * v_target
        if record_weights:
            w_v_val = w_v.mean().item()  # Mean across heads
            w_v_head = (w_v[0, :, 0, 0].detach().float().cpu().tolist())

    if pb:
        w_p = weight_fn(q_current, p_target, p_sgm, head_dim)
        steer_vec += w_p * p_target
        if record_weights:
            w_p_val = w_p.mean().item()  # Mean across heads
            w_p_head = (w_p[0, :, 0, 0].detach().float().cpu().tolist())

    if db and all_gen_keys.size(2) > 0:
        if decode_window == -1:
            g_target = all_gen_keys.mean(dim=2, keepdim=True)
        else:
            g_target = all_gen_keys[:, :, -decode_window:, :].mean(dim=2, keepdim=True)
        w_g = weight_fn(q_current, g_target, d_sgm, head_dim)
        steer_vec += w_g * g_target
        if record_weights:
            w_g_val = w_g.mean().item()  # Mean across heads
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
    # Project QKV
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

    # Process GQA: Flatten Key/Value to Same Head number as Query.
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    #### qsteer
    if getattr(self, "use_qsteer_adaptive", False):
        # Get prefill length
        if q_len > 1: 
            self.prefill_len = q_len
            # Visual/prefill targets are calculated during prefilling
            #### Prefill visual keys
            # self.img_start_idx, self.img_end_idx have been added
            visual_keys = key_states[:, :, self.img_start_idx : self.img_end_idx, :]

            #### Prefill text keys
            prefilltxt_indices = list(range(0, self.img_start_idx)) + list(range(self.img_end_idx, self.prefill_len))
            prefill_txt_keys = key_states[:, :, prefilltxt_indices, :]

            self.v_target = visual_keys.mean(dim=2, keepdim=True)
            self.p_target = prefill_txt_keys.mean(dim=2, keepdim=True)
        # Shape key_states [bsz, 16, seq_len, head_dim]

        #### All Generated keys
        all_gen_keys = key_states[:, :, self.prefill_len:-1, :]

        # Update Query: Steering Q to Current visual tokens
        # alpha: stride
        _record = getattr(self, "record_weights", False)
        result = get_adaptive_qsteer(
            query_states, 
            self.v_target, 
            self.p_target, 
            all_gen_keys, 
            ## Choose branch
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
            # Adaptive mode
            adaptive_mode=self.adaptive_mode,
            # Record
            record_weights=_record,
            )
        if _record:
            steer_vec, weights_dict = result
            # Record into the layer-level list
            if not hasattr(self, "_recorded_weights"):
                self._recorded_weights = []
            self._recorded_weights.append(weights_dict)
        else:
            steer_vec = result
        query_states[:, :, -1:, :] = (1. - self.alpha) * query_states[:, :, -1:, :] + self.alpha * steer_vec
    attention_interface: Callable = ALL_ATTENTION_FUNCTIONS._global_mapping.get(
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
        sliding_window=self.sliding_window,
        position_ids=position_ids,  # Pass positions for FA2
        **kwargs,
    )

    attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights

    # # 5. Compute Attention Weights
    # attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling

    # if attention_mask is not None:
    #     # Pay attention to dimension alignment; Qwen's mask handling can sometimes be strict
    #     if attention_mask.size()[-1] == attn_weights.size()[-1]:
    #         attn_weights = attn_weights + attention_mask

    # # 6. Softmax & Output
    # attn_probs = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    # attn_output = torch.matmul(attn_probs, value_states)

    # # 7. Restore dimensions
    # attn_output = attn_output.transpose(1, 2).contiguous()
    # attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    # attn_output = self.o_proj(attn_output)

    # return attn_output, (attn_probs if output_attentions else None)

def qwen_modify_qsteer_adaptive(model, start_layer, end_layer, use_qsteer_adaptive, alpha,
                    img_start_idx, img_end_idx, decode_window, visual_branch, prefill_branch, decode_branch,
                    visual_sigma, prefill_sigma, decode_sigma, adaptive_mode="gaussian", record_weights=False):
    """
    Inject QSteer Adaptive into the self_attn modules of the specified decoder layers.
    
    Args:
        adaptive_mode: Adaptive weighting mode, supporting:
            - "gaussian": RBF Gaussian kernel (default, original version)
            - "cosine": Cosine similarity + temperature scaling
            - "mutual_info": Mutual-information-approximation-based weighting
            - "kl": KL-divergence-based weighting
    """
    assert adaptive_mode in ADAPTIVE_WEIGHT_FN, \
        f"adaptive_mode must be one of {list(ADAPTIVE_WEIGHT_FN.keys())}, got '{adaptive_mode}'"
    
    # The Qwen2.5-VL model structure is model.layers
    for i in range(start_layer, end_layer):
        target_layer = model.language_model.layers[i].self_attn
        
        # Inject parameters
        target_layer.use_qsteer_adaptive = use_qsteer_adaptive
        target_layer.alpha = alpha
        target_layer.img_start_idx = img_start_idx
        target_layer.img_end_idx = img_end_idx
        target_layer.decode_window = decode_window

        target_layer.visual_branch, target_layer.prefill_branch, target_layer.decode_branch = visual_branch, prefill_branch, decode_branch
        target_layer.visual_sigma, target_layer.prefill_sigma, target_layer.decode_sigma = visual_sigma, prefill_sigma, decode_sigma
        target_layer.adaptive_mode = adaptive_mode
        target_layer.record_weights = record_weights
        
        # Inject required attributes (prevent new_forward from failing to find variables)
        if not hasattr(target_layer, "num_key_value_groups"):
            target_layer.num_key_value_groups = target_layer.num_heads // target_layer.num_key_value_heads
        
        # Replace method
        target_layer.forward = types.MethodType(qwen2_5_vl_new_forward, target_layer)

def collect_recorded_weights(model, start_layer, end_layer):
    """
    Collect recorded steering weights from each layer.
    
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
    Reset the weights recorded by each layer (called after inference for each image).
    """
    for i in range(start_layer, end_layer):
        target_layer = model.language_model.layers[i].self_attn
        if hasattr(target_layer, "_recorded_weights"):
            target_layer._recorded_weights = []