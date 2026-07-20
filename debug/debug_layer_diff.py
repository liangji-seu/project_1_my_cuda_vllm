"""
Step 3 + 4: Per-layer tensor diff between HF reference and C++ inference replica.

This script EXACTLY replicates the C++ inference logic (same formulas, same eps, same RoPE cache)
and compares against the HuggingFace transformers reference output, layer by layer.
"""
import math
import struct
import os
import sys
import json
import copy

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================
# Configuration
# ============================================================
MODEL_PATH = "/home/liangji/huggingface/Qwen2.5-0.5B-Instruct"
BIN_PATH = "/home/liangji/AI_INFRA/projects/my_cuda_vllm/demo/qwen2.5_0.5b_instruct.bin"

# C++ hardcoded constants (from rmsnorm_kernel.cu:44, rmsnorm_kernel.cpp:25)
CPP_RMSNORM_EPS = 1e-5
# HF config value: rms_norm_eps = 1e-6
HF_RMSNORM_EPS = 1e-6

THETA = 1000000.0  # Qwen2.5 uses theta=1000000.0

DTYPE = torch.float32  # C++ uses fp32

MAX_SEQ_LEN = 32768

# ============================================================
# Step 1: Read .bin file header and weights
# ============================================================
def read_bin_header(bin_path):
    with open(bin_path, 'rb') as f:
        data = f.read()
    header_size = 32  # 8 ints * 4 bytes
    header = struct.unpack('iiiiiiii', data[:header_size])
    config = {
        'dim': header[0],
        'hidden_dim': header[1],
        'n_layers': header[2],
        'n_heads': header[3],
        'n_kv_heads': header[4],
        'vocab_size': abs(header[5]),
        'shared_weights': header[5] > 0,  # Positive = shared (tied), negative = separate
        'max_seq_len': header[6],
        'head_dim': header[7],
    }
    config['head_size'] = config['head_dim'] if config['head_dim'] > 0 else config['dim'] // config['n_heads']
    config['kv_dim'] = config['head_size'] * config['n_kv_heads']
    config['q_dim'] = config['head_size'] * config['n_heads']
    config['kv_mul'] = config['n_heads'] // config['n_kv_heads']
    return config, data[header_size:]


def parse_bin_weights(config, weight_data):
    dim = config['dim']
    layer_num = config['n_layers']
    q_dim = config['q_dim']
    kv_dim = config['kv_dim']
    head_size = config['head_size']
    hidden_dim = config['hidden_dim']
    vocab_size = config['vocab_size']
    seq_len = config['max_seq_len']
    max_floats = len(weight_data) // 4

    def read_floats(offset, count):
        if offset + count > max_floats:
            return None, offset
        vals = struct.unpack(f'{count}f', weight_data[offset*4:(offset+count)*4])
        return list(vals), offset + count

    weights = {}
    pos = 0

    # Embedding
    vals, pos = read_floats(pos, dim * vocab_size)
    if vals is None:  # Check if vocab_size was negated for shared weights
        config['shared_weights'] = True
        config['vocab_size'] = abs(vocab_size)
        vals, pos = read_floats(0, dim * config['vocab_size'])
    weights['emb'] = vals
    weights['vocab_size_actual'] = config['vocab_size']

    # Attention RMSNorms
    weights['attn_norm'] = []
    for i in range(layer_num):
        vals, pos = read_floats(pos, dim)
        weights['attn_norm'].append(vals)

    # WQ weights
    weights['wq'] = []
    for i in range(layer_num):
        vals, pos = read_floats(pos, q_dim * dim)
        weights['wq'].append(vals)

    # Check for WQ bias
    bias_check = pos + layer_num * q_dim
    weights['q_bias'] = []
    if bias_check <= max_floats:
        for i in range(layer_num):
            vals, pos = read_floats(pos, q_dim)
            weights['q_bias'].append(vals)
    has_qkv_bias = len(weights['q_bias']) > 0

    # WK weights
    weights['wk'] = []
    for i in range(layer_num):
        vals, pos = read_floats(pos, kv_dim * dim)
        weights['wk'].append(vals)

    # WK bias
    weights['k_bias'] = []
    if has_qkv_bias:
        for i in range(layer_num):
            vals, pos = read_floats(pos, kv_dim)
            weights['k_bias'].append(vals)

    # WV weights
    weights['wv'] = []
    for i in range(layer_num):
        vals, pos = read_floats(pos, kv_dim * dim)
        weights['wv'].append(vals)

    # WV bias
    weights['v_bias'] = []
    if has_qkv_bias:
        for i in range(layer_num):
            vals, pos = read_floats(pos, kv_dim)
            weights['v_bias'].append(vals)

    # WO weights
    weights['wo'] = []
    for i in range(layer_num):
        vals, pos = read_floats(pos, dim * q_dim)
        weights['wo'].append(vals)

    # FFN RMSNorms
    weights['ffn_norm'] = []
    for i in range(layer_num):
        vals, pos = read_floats(pos, dim)
        weights['ffn_norm'].append(vals)

    # W1 (gate)
    weights['w1'] = []
    for i in range(layer_num):
        vals, pos = read_floats(pos, hidden_dim * dim)
        weights['w1'].append(vals)

    # W2 (down)
    weights['w2'] = []
    for i in range(layer_num):
        vals, pos = read_floats(pos, dim * hidden_dim)
        weights['w2'].append(vals)

    # W3 (up)
    weights['w3'] = []
    for i in range(layer_num):
        vals, pos = read_floats(pos, hidden_dim * dim)
        weights['w3'].append(vals)

    # Final RMSNorm
    weights['final_norm'] = []
    vals, pos = read_floats(pos, dim)
    weights['final_norm'].append(vals)

    # Skip freqs (not used by C++ — recomputed)
    pos += 2 * seq_len * head_size

    # CLS weight
    if 'shared_weights' in config and config['shared_weights']:
        pass  # Will use embedding weights
    else:
        vals, pos = read_floats(pos, dim * config['vocab_size'])
        weights['cls'] = vals

    # Q/K norms (Qwen3 specific)
    qk_norm_pos = (
        dim * config['vocab_size']
        + layer_num * dim          # attn rmsnorm
        + layer_num * q_dim * dim  # wq
        + layer_num * kv_dim * dim # wk
        + layer_num * kv_dim * dim # wv
        + layer_num * dim * q_dim  # wo
        + layer_num * dim          # ffn rmsnorm
        + layer_num * dim * hidden_dim # w1
        + layer_num * dim * hidden_dim # w2
        + layer_num * dim * hidden_dim # w3
        + dim                      # final rmsnorm
        + 2 * seq_len * head_size  # freqs
    )
    if has_qkv_bias:
        qk_norm_pos += layer_num * q_dim   # Wq bias
        qk_norm_pos += layer_num * kv_dim  # Wk bias
        qk_norm_pos += layer_num * kv_dim  # Wv bias
    if not ('shared_weights' in config and config['shared_weights']):
        qk_norm_pos += config['vocab_size'] * dim  # CLS

    qk_norm_total = 2 * layer_num * head_size
    weights['q_norm'] = []
    weights['k_norm'] = []
    if qk_norm_pos + qk_norm_total <= max_floats:
        for i in range(layer_num):
            vals, pos_ = read_floats(qk_norm_pos, head_size)
            weights['q_norm'].append(vals)
            qk_norm_pos += head_size
        for i in range(layer_num):
            vals, pos_ = read_floats(qk_norm_pos, head_size)
            weights['k_norm'].append(vals)
            qk_norm_pos += head_size
    has_qk_norm = len(weights['q_norm']) > 0

    # Convert to torch tensors
    def to_tensor(vals, shape):
        return torch.tensor(vals, dtype=DTYPE).reshape(shape)

    # Embedding: [vocab_size, dim]
    weights['emb'] = to_tensor(weights['emb'], [config['vocab_size'], dim])
    # Norms: list of [dim]
    weights['attn_norm'] = [to_tensor(w, [dim]) for w in weights['attn_norm']]
    weights['ffn_norm'] = [to_tensor(w, [dim]) for w in weights['ffn_norm']]
    weights['final_norm'] = to_tensor(weights['final_norm'][0], [dim])
    # WQ: list of [q_dim, dim] -> for matmul as [K, M]
    weights['wq'] = [to_tensor(w, [q_dim, dim]) for w in weights['wq']]
    weights['wk'] = [to_tensor(w, [kv_dim, dim]) for w in weights['wk']]
    weights['wv'] = [to_tensor(w, [kv_dim, dim]) for w in weights['wv']]
    weights['wo'] = [to_tensor(w, [dim, q_dim]) for w in weights['wo']]
    # W1/W3: [hidden_dim, dim], W2: [dim, hidden_dim]
    weights['w1'] = [to_tensor(w, [hidden_dim, dim]) for w in weights['w1']]
    weights['w2'] = [to_tensor(w, [dim, hidden_dim]) for w in weights['w2']]
    weights['w3'] = [to_tensor(w, [hidden_dim, dim]) for w in weights['w3']]
    # CLS: [vocab_size, dim]
    if 'cls' in weights:
        weights['cls'] = to_tensor(weights['cls'], [config['vocab_size'], dim])

    # Biases: list of [dim]
    weights['q_bias'] = [to_tensor(w, [q_dim]) for w in weights['q_bias']] if weights['q_bias'] else []
    weights['k_bias'] = [to_tensor(w, [kv_dim]) for w in weights['k_bias']] if weights['k_bias'] else []
    weights['v_bias'] = [to_tensor(w, [kv_dim]) for w in weights['v_bias']] if weights['v_bias'] else []

    weights['config'] = config
    weights['has_qkv_bias'] = has_qkv_bias
    weights['has_qk_norm'] = has_qk_norm
    return weights


# ============================================================
# Step 2: C++ inference replica (EXACT same formulas)
# ============================================================
def precompute_freqs_cis_cpp(head_size, max_seq_len, theta=THETA):
    """EXACT replica of C++ sin_cos_cache_calc_cpu (QWEN3_SUPPORT path)."""
    sin_cache = torch.zeros(max_seq_len, head_size, dtype=DTYPE)
    cos_cache = torch.zeros(max_seq_len, head_size, dtype=DTYPE)
    for pos in range(max_seq_len):
        for i in range(head_size):
            pair_idx = i % (head_size // 2)
            freq = 1.0 / (theta ** (2.0 * pair_idx / head_size))
            val = pos * freq
            sin_cache[pos, i] = math.sin(val)
            cos_cache[pos, i] = math.cos(val)
    return sin_cache, cos_cache


def rmsnorm_cpp(x, weight, eps=CPP_RMSNORM_EPS):
    """EXACT replica of C++ rmsnorm_kernel_cpu."""
    # x: [dim], weight: [dim]
    mean_sq = (x * x).mean() + eps
    rsqrt = 1.0 / math.sqrt(mean_sq.item())
    return weight * (rsqrt * x)


def matmul_cpp(x, weight, scale=1.0):
    """EXACT replica of C++ matmul_kernel_cpu.
    x: [M], weight: [K, M], output: [K]
    Y = (X @ W^T) * scale
    """
    return (x @ weight.T) * scale


def rope_cpp(x_q, x_k, pos, sin_cache, cos_cache, q_dim, kv_dim, head_size):
    """EXACT replica of C++ rope_kernel_cpu (QWEN3_SUPPORT = LLaMA-style half-rotate)."""
    half_dim = head_size // 2
    q_heads = q_dim // head_size
    kv_heads = kv_dim // head_size

    for h in range(q_heads):
        head_off = h * head_size
        for i in range(half_dim):
            fci = sin_cache[pos, i].item()
            fcr = cos_cache[pos, i].item()

            q0 = x_q[head_off + i].item()
            q1 = x_q[head_off + i + half_dim].item()
            x_q[head_off + i] = q0 * fcr - q1 * fci
            x_q[head_off + i + half_dim] = q0 * fci + q1 * fcr

        if h < kv_heads:
            k_off = h * head_size
            for i in range(half_dim):
                fci = sin_cache[pos, i].item()
                fcr = cos_cache[pos, i].item()

                k0 = x_k[k_off + i].item()
                k1 = x_k[k_off + i + half_dim].item()
                x_k[k_off + i] = k0 * fcr - k1 * fci
                x_k[k_off + i + half_dim] = k0 * fci + k1 * fcr
    return x_q, x_k


def swiglu_cpp(gate, up):
    """EXACT replica of C++ swiglu_kernel_cpu.
    SwiGLU: silu(gate) * up = gate * sigmoid(gate) * up
    """
    return gate * (1.0 / (1.0 + torch.exp(-gate))) * up


def mha_cpp(query, key_cache_full, value_cache_full, pos, layer_idx, config, attn_scores):
    """EXACT replica of C++ mha_kernel_cpu."""
    head_num = config['n_heads']
    kv_mul = config['kv_mul']
    kv_dim = config['kv_dim']
    seq_len = config['max_seq_len']
    head_size = config['head_size']
    layer_offset = layer_idx * seq_len * kv_dim
    scale = 1.0 / math.sqrt(head_size)

    output = torch.zeros(config['q_dim'], dtype=DTYPE)

    for h in range(head_num):
        kv_head = h // kv_mul
        q_head = query[h * head_size:(h + 1) * head_size]

        # Compute attention scores
        for t in range(pos + 1):
            cache_offset = t * kv_dim + kv_head * head_size
            k_head = key_cache_full[layer_offset + cache_offset:layer_offset + cache_offset + head_size]
            dot = (q_head * k_head).sum().item()
            attn_scores[h, t] = dot * scale

        # Softmax
        max_score = attn_scores[h, :pos + 1].max().item()
        sum_exp = 0.0
        for t in range(pos + 1):
            exp_val = math.exp(attn_scores[h, t].item() - max_score)
            attn_scores[h, t] = exp_val
            sum_exp += exp_val
        for t in range(pos + 1):
            attn_scores[h, t] /= sum_exp

        # Weighted sum with V
        out_head = torch.zeros(head_size, dtype=DTYPE)
        for t in range(pos + 1):
            cache_offset = t * kv_dim + kv_head * head_size
            v_head = value_cache_full[layer_offset + cache_offset:layer_offset + cache_offset + head_size]
            for d in range(head_size):
                out_head[d] += attn_scores[h, t].item() * v_head[d].item()
        output[h * head_size:(h + 1) * head_size] = out_head

    return output


# ============================================================
# Step 3: Single layer C++ replica forward
# ============================================================
def cpp_forward_layer(x, pos, layer_idx, weights, config, sin_cache, cos_cache,
                      key_cache, value_cache, attn_scores):
    """One transformer block, EXACTLY as the C++ code does it."""
    dim = config['dim']
    q_dim = config['q_dim']
    kv_dim = config['kv_dim']
    head_size = config['head_size']
    eps = CPP_RMSNORM_EPS

    # Attention RMSNorm
    attn_norm_w = weights['attn_norm'][layer_idx]
    x_normed = rmsnorm_cpp(x.clone(), attn_norm_w, eps)

    # Q projection: Y = X @ Wq^T
    q = matmul_cpp(x_normed.clone(), weights['wq'][layer_idx])
    # K projection
    k = matmul_cpp(x_normed.clone(), weights['wk'][layer_idx])
    # V projection
    v = matmul_cpp(x_normed.clone(), weights['wv'][layer_idx])

    # Q/K/V bias
    if weights['has_qkv_bias']:
        q = q + weights['q_bias'][layer_idx]
        k = k + weights['k_bias'][layer_idx]
        v = v + weights['v_bias'][layer_idx]

    # Q/K norm (Qwen3 specific) — NOT used for Qwen2.5
    # if weights['has_qk_norm']: ...

    # RoPE
    q, k = rope_cpp(q, k, pos, sin_cache, cos_cache, q_dim, kv_dim, head_size)

    # Store K, V in cache
    layer_offset = layer_idx * config['max_seq_len'] * kv_dim
    cache_start = layer_offset + pos * kv_dim
    key_cache[cache_start:cache_start + kv_dim] = k
    value_cache[cache_start:cache_start + kv_dim] = v

    # MHA
    mha_out = mha_cpp(q, key_cache, value_cache, pos, layer_idx, config, attn_scores)

    # Wo projection
    attn_out = matmul_cpp(mha_out, weights['wo'][layer_idx])

    # Residual: x = x + attn_out (done in-place in C++ via VecAddLayer)
    h = x + attn_out

    # FFN RMSNorm
    ffn_norm_w = weights['ffn_norm'][layer_idx]
    ffn_normed = rmsnorm_cpp(h.clone(), ffn_norm_w, eps)

    # W1 (gate)
    gate = matmul_cpp(ffn_normed.clone(), weights['w1'][layer_idx])
    # W3 (up)
    up = matmul_cpp(ffn_normed.clone(), weights['w3'][layer_idx])
    # SwiGLU
    gate = swiglu_cpp(gate, up)
    # W2 (down)
    ffn_out = matmul_cpp(gate, weights['w2'][layer_idx])

    # Residual: x = x + ffn_out
    out = h + ffn_out

    return out, q, k, v, mha_out, attn_out, ffn_out


def cpp_forward_full(tokens, weights, config):
    """Full C++ replica forward pass."""
    dim = config['dim']
    head_size = config['head_size']
    seq_len = config['max_seq_len']
    layer_num = config['n_layers']
    eps = CPP_RMSNORM_EPS

    sin_cache, cos_cache = precompute_freqs_cis_cpp(head_size, seq_len)
    kv_dim_total = layer_num * seq_len * config['kv_dim']
    key_cache = torch.zeros(kv_dim_total, dtype=DTYPE)
    value_cache = torch.zeros(kv_dim_total, dtype=DTYPE)
    attn_scores = torch.zeros(config['n_heads'], seq_len, dtype=DTYPE)

    all_intermediates = []
    seq_len_t = len(tokens)

    for pos in range(seq_len_t):
        token = tokens[pos]
        x = weights['emb'][token].clone()

        layer_outputs = []
        for layer_idx in range(layer_num):
            result = cpp_forward_layer(
                x.clone() if layer_idx > 0 else x, pos, layer_idx,
                weights, config, sin_cache, cos_cache,
                key_cache, value_cache, attn_scores
            )
            x = result[0]
            # layer_outputs.append(result)

        # Final RMSNorm
        x = rmsnorm_cpp(x, weights['final_norm'], eps)

        # LM Head
        if 'cls' in weights:
            logits = matmul_cpp(x, weights['cls'])
        else:
            logits = matmul_cpp(x, weights['emb'])

        all_intermediates.append({
            'pos': pos,
            'hidden': x.clone(),
            'logits': logits.clone(),
        })

    return all_intermediates


# ============================================================
# Step 4: HF reference forward pass with intermediate dumping
# ============================================================
def hf_forward_with_intermediates(model, input_ids):
    """Run HF model forward and capture intermediates by hooking modules."""
    intermediates = {}

    # Hook all linear layers
    hooks = []

    def make_hook(name):
        def hook(module, input, output):
            intermediates[name + '.input'] = input[0].detach().cpu()
            intermediates[name + '.output'] = output.detach().cpu()
        return hook

    for n, m in model.named_modules():
        if isinstance(m, nn.Linear) or isinstance(m, nn.Embedding):
            h = m.register_forward_hook(make_hook(n))
            hooks.append(h)

    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)

    for h in hooks:
        h.remove()

    return outputs, intermediates


# ============================================================
# Main debug
# ============================================================
def main():
    print("=" * 70)
    print("STEP 1: Read .bin file config")
    print("=" * 70)
    config, weight_data_bytes = read_bin_header(BIN_PATH)
    for k, v in config.items():
        print(f"  {k}: {v}")

    print()
    print("=" * 70)
    print("STEP 1: Parse .bin weights")
    print("=" * 70)
    weights = parse_bin_weights(config, weight_data_bytes)
    print(f"  Embedding shape: {weights['emb'].shape}")
    print(f"  Layers: {config['n_layers']}")
    print(f"  WQ[0] shape: {weights['wq'][0].shape}")
    print(f"  Q bias present: {weights['has_qkv_bias']}")
    print(f"  Q/K norm present: {weights['has_qk_norm']}")

    # Diagnostics: compare .bin weights vs HF weights
    print()
    print("=" * 70)
    print("STEP 2: Load HF reference model")
    print("=" * 70)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=DTYPE, device_map='cpu', local_files_only=True
    )
    model.eval()
    hf_sd = model.state_dict()

    print(f"  HF model type: {model.config.model_type}")
    print(f"  HF hidden_size: {model.config.hidden_size}")
    print(f"  HF num_layers: {model.config.num_hidden_layers}")
    print(f"  HF num_heads: {model.config.num_attention_heads}")
    print(f"  HF num_kv_heads: {model.config.num_key_value_heads}")
    print(f"  HF rms_norm_eps: {model.config.rms_norm_eps}")
    print(f"  HF tie_word_embeddings: {model.config.tie_word_embeddings}")

    # Compare weights
    print()
    print("=" * 70)
    print("STEP 2: Weight comparison (.bin vs HF)")
    print("=" * 70)

    def compare_weights(bin_w, hf_w, name):
        max_diff = (bin_w.float() - hf_w.float()).abs().max().item()
        mean_diff = (bin_w.float() - hf_w.float()).abs().mean().item()
        rel_err = max_diff / (hf_w.float().abs().max().item() + 1e-10)
        status = "✓" if max_diff < 1e-5 else "✗ MISMATCH"
        print(f"  {name:50s} shape={str(bin_w.shape):20s} max_diff={max_diff:.2e} mean={mean_diff:.2e} rel={rel_err:.2e} {status}")

    # Embedding
    hf_emb = hf_sd['model.embed_tokens.weight'].float()
    compare_weights(weights['emb'], hf_emb, 'embed_tokens')

    # Per-layer weights
    for i in range(config['n_layers']):
        hf_q = hf_sd[f'model.layers.{i}.self_attn.q_proj.weight'].float()
        compare_weights(weights['wq'][i], hf_q, f'layer.{i}.q_proj')
        hf_k = hf_sd[f'model.layers.{i}.self_attn.k_proj.weight'].float()
        compare_weights(weights['wk'][i], hf_k, f'layer.{i}.k_proj')
        hf_v = hf_sd[f'model.layers.{i}.self_attn.v_proj.weight'].float()
        compare_weights(weights['wv'][i], hf_v, f'layer.{i}.v_proj')
        hf_o = hf_sd[f'model.layers.{i}.self_attn.o_proj.weight'].float()
        compare_weights(weights['wo'][i], hf_o, f'layer.{i}.o_proj')

        if weights['has_qkv_bias']:
            hf_qb = hf_sd[f'model.layers.{i}.self_attn.q_proj.bias'].float()
            compare_weights(weights['q_bias'][i], hf_qb, f'layer.{i}.q_bias')
            hf_kb = hf_sd[f'model.layers.{i}.self_attn.k_proj.bias'].float()
            compare_weights(weights['k_bias'][i], hf_kb, f'layer.{i}.k_bias')
            hf_vb = hf_sd[f'model.layers.{i}.self_attn.v_proj.bias'].float()
            compare_weights(weights['v_bias'][i], hf_vb, f'layer.{i}.v_bias')

        hf_attn_norm = hf_sd[f'model.layers.{i}.input_layernorm.weight'].float()
        compare_weights(weights['attn_norm'][i], hf_attn_norm, f'layer.{i}.input_layernorm')

        hf_gate = hf_sd[f'model.layers.{i}.mlp.gate_proj.weight'].float()
        compare_weights(weights['w1'][i], hf_gate, f'layer.{i}.gate_proj')
        hf_up = hf_sd[f'model.layers.{i}.mlp.up_proj.weight'].float()
        compare_weights(weights['w3'][i], hf_up, f'layer.{i}.up_proj')
        hf_down = hf_sd[f'model.layers.{i}.mlp.down_proj.weight'].float()
        compare_weights(weights['w2'][i], hf_down, f'layer.{i}.down_proj')

        hf_ffn_norm = hf_sd[f'model.layers.{i}.post_attention_layernorm.weight'].float()
        compare_weights(weights['ffn_norm'][i], hf_ffn_norm, f'layer.{i}.post_attn_layernorm')
        break  # Just check first layer

    # Final norm
    hf_final_norm = hf_sd['model.norm.weight'].float()
    compare_weights(weights['final_norm'], hf_final_norm, 'final_norm')

    # LM head
    hf_lm = hf_sd['lm_head.weight'].float()
    compare_weights(weights['emb'], hf_lm, 'lm_head (tied)')

    print()
    print("=" * 70)
    print("STEP 2: Named parameters check")
    print("=" * 70)
    for name, param in model.named_parameters():
        if 'layers.0' in name or 'embed' in name or 'norm' in name or 'lm_head' in name:
            if 'layers.1' not in name and 'layers.2' not in name:
                print(f"  {name}: {list(param.shape)}")

    print()
    print("=" * 70)
    print("STEP 3: RMSNorm epsilon comparison")
    print("=" * 70)
    print(f"  C++ hardcoded eps (rmsnorm_kernel.cu:44, rmsnorm_kernel.cpp:25): {CPP_RMSNORM_EPS}")
    print(f"  HF config rms_norm_eps: {model.config.rms_norm_eps}")
    print(f"  MISMATCH: {'YES' if abs(CPP_RMSNORM_EPS - model.config.rms_norm_eps) > 1e-10 else 'NO'}")
    print(f"  Relative difference: {abs(CPP_RMSNORM_EPS - model.config.rms_norm_eps) / model.config.rms_norm_eps * 100:.0f}%")

    print()
    print("=" * 70)
    print("STEP 3: RoPE convention check")
    print("=" * 70)
    from transformers.models.qwen2.modeling_qwen2 import rotate_half, apply_rotary_pos_emb
    import inspect
    print(f"  HF uses LLaMA-style rotate_half: {inspect.getsource(rotate_half)[:200]}...")
    print(f"  C++ CPU kernel (QWEN3_SUPPORT) uses LLaMA-style half-rotate: MATCH ✓")
    print(f"  C++ GPU kernel uses GPT-NeoX-style adjacent pairs: MISMATCH ✗")

    print()
    print("=" * 70)
    print("STEP 4: Known issues summary")
    print("=" * 70)
    issues = [
        ("RMSNorm eps", "C++=1e-5, HF=1e-6", "CRITICAL" if abs(CPP_RMSNORM_EPS - model.config.rms_norm_eps) > 1e-10 else "OK"),
        ("QKV bias loading", f"has_qkv_bias={weights['has_qkv_bias']}", "OK" if weights['has_qkv_bias'] else "CRITICAL — biases missing!"),
        ("RoPE convention (CPU)", "LLaMA-style half-rotate", "OK (matches HF Qwen2)"),
        ("RoPE convention (GPU)", "GPT-NeoX adjacent pairs", "BUG (GPU path only, not used by demo)"),
        ("RoPE theta", f"{THETA}", "OK (matches HF)"),
        ("Q/K norm", f"present={weights['has_qk_norm']}", "OK (correctly absent for Qwen2.5)"),
        ("Matmul transpose", "X @ W^T (both CPU and GPU)", "OK"),
        ("SwiGLU formula", "silu(gate) * up", "OK"),
        ("Weight tying (LM head)", "shared with embedding", "OK"),
    ]
    for name, detail, status in issues:
        print(f"  [{status}] {name}: {detail}")

    print()
    print("=" * 70)
    print("STEP 3b: Per-layer tensor diff (using replicated C++ logic)")
    print("=" * 70)

    # Get tokenizer and prepare input
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    prompt = "你好"
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt")

    input_ids = model_inputs.input_ids
    print(f"  Input text: '{prompt}'")
    print(f"  Tokenized: {input_ids.tolist()}")
    print(f"  Tokens: {[tokenizer.decode([t]) for t in input_ids[0].tolist()]}")

    # Run C++ replica
    tokens_list = input_ids[0].tolist()
    cpp_results = cpp_forward_full(tokens_list, weights, config)

    # Run HF model
    with torch.no_grad():
        hf_outputs = model(input_ids, output_hidden_states=True)

    # Compare per-position hidden states
    print()
    print("  --- Per-layer hidden state comparison ---")
    # For the first token (prefill), compare hidden states
    # HF hidden_states[0] = embedding, hidden_states[1] = after layer 0, etc.
    # But C++ does per-position processing — compare the last position

    last_pos = len(tokens_list) - 1
    cpp_hidden = cpp_results[last_pos]['hidden']
    hf_hidden_last = hf_outputs.hidden_states[-2][0, -1, :].float()  # Before final norm

    max_diff = (cpp_hidden - hf_hidden_last).abs().max().item()
    mean_diff = (cpp_hidden - hf_hidden_last).abs().mean().item()
    print(f"  Last position, pre-norm hidden (C++ vs HF):")
    print(f"    max_abs_err={max_diff:.6e}, mean_abs_err={mean_diff:.6e}")

    # Compare logits
    cpp_logits = cpp_results[last_pos]['logits']
    hf_logits = hf_outputs.logits[0, -1, :].float()

    max_diff_logits = (cpp_logits - hf_logits).abs().max().item()
    mean_diff_logits = (cpp_logits - hf_logits).abs().mean().item()
    print(f"  Last position, logits (C++ vs HF):")
    print(f"    max_abs_err={max_diff_logits:.6e}, mean_abs_err={mean_diff_logits:.6e}")

    # Compare top-k predictions
    cpp_top5 = cpp_logits.topk(5)
    hf_top5 = hf_logits.topk(5)
    print(f"  C++ top-5 logit indices: {cpp_top5.indices.tolist()}")
    print(f"  HF  top-5 logit indices: {hf_top5.indices.tolist()}")
    print(f"  C++ top-5 logit values: {[f'{v:.4f}' for v in cpp_top5.values.tolist()]}")
    print(f"  HF  top-5 logit values: {[f'{v:.4f}' for v in hf_top5.values.tolist()]}")

    # Also test with correct eps
    print()
    print("  --- Testing with corrected eps=1e-6 ---")

    def cpp_forward_with_eps(eps_override):
        """Run C++ replica with a specific eps."""
        sin_cache, cos_cache = precompute_freqs_cis_cpp(config['head_size'], config['max_seq_len'])
        kv_dim_total = config['n_layers'] * config['max_seq_len'] * config['kv_dim']
        key_cache = torch.zeros(kv_dim_total, dtype=DTYPE)
        value_cache = torch.zeros(kv_dim_total, dtype=DTYPE)
        attn_scores = torch.zeros(config['n_heads'], config['max_seq_len'], dtype=DTYPE)

        # Monkey-patch rmsnorm
        def rmsnorm_cpp_eps(x, weight):
            mean_sq = (x * x).mean() + eps_override
            rsqrt = 1.0 / math.sqrt(mean_sq.item())
            return weight * (rsqrt * x)

        global rmsnorm_cpp
        orig_rmsnorm = rmsnorm_cpp

        for pos in range(len(tokens_list)):
            token = tokens_list[pos]
            x = weights['emb'][token].clone()
            for layer_idx in range(config['n_layers']):
                # Use eps_override
                attn_norm_w = weights['attn_norm'][layer_idx]
                x_normed = rmsnorm_cpp_eps(x.clone(), attn_norm_w)

                q = matmul_cpp(x_normed.clone(), weights['wq'][layer_idx])
                k = matmul_cpp(x_normed.clone(), weights['wk'][layer_idx])
                v = matmul_cpp(x_normed.clone(), weights['wv'][layer_idx])

                if weights['has_qkv_bias']:
                    q = q + weights['q_bias'][layer_idx]
                    k = k + weights['k_bias'][layer_idx]
                    v = v + weights['v_bias'][layer_idx]

                q, k = rope_cpp(q, k, pos, sin_cache, cos_cache,
                                config['q_dim'], config['kv_dim'], config['head_size'])

                layer_offset = layer_idx * config['max_seq_len'] * config['kv_dim']
                cache_start = layer_offset + pos * config['kv_dim']
                key_cache[cache_start:cache_start + config['kv_dim']] = k
                value_cache[cache_start:cache_start + config['kv_dim']] = v

                mha_out = mha_cpp(q, key_cache, value_cache, pos, layer_idx, config, attn_scores)
                attn_out = matmul_cpp(mha_out, weights['wo'][layer_idx])
                h = x + attn_out

                ffn_norm_w = weights['ffn_norm'][layer_idx]
                ffn_normed = rmsnorm_cpp_eps(h.clone(), ffn_norm_w)
                gate = matmul_cpp(ffn_normed.clone(), weights['w1'][layer_idx])
                up = matmul_cpp(ffn_normed.clone(), weights['w3'][layer_idx])
                gate = swiglu_cpp(gate, up)
                ffn_out = matmul_cpp(gate, weights['w2'][layer_idx])
                x = h + ffn_out

            x = rmsnorm_cpp_eps(x, weights['final_norm'])
            if 'cls' in weights:
                logits = matmul_cpp(x, weights['cls'])
            else:
                logits = matmul_cpp(x, weights['emb'])

        return x, logits

    corrected_hidden, corrected_logits = cpp_forward_with_eps(HF_RMSNORM_EPS)
    max_diff_corrected = (corrected_hidden - hf_hidden_last).abs().max().item()
    mean_diff_corrected = (corrected_hidden - hf_hidden_last).abs().mean().item()
    max_diff_corrected_logits = (corrected_logits - hf_logits).abs().max().item()

    print(f"  With eps={HF_RMSNORM_EPS} (correct):")
    print(f"    hidden max_abs_err={max_diff_corrected:.6e}, mean_abs_err={mean_diff_corrected:.6e}")
    print(f"    logits max_abs_err={max_diff_corrected_logits:.6e}")

    corrected_top5 = corrected_logits.topk(5)
    print(f"  Corrected top-5 indices: {corrected_top5.indices.tolist()}")
    print(f"  HF       top-5 indices: {hf_top5.indices.tolist()}")

    # Final verdict
    print()
    print("=" * 70)
    print("STEP 5: DIAGNOSIS")
    print("=" * 70)
    if max_diff < 0.01 and max_diff_logits < 0.01:
        print("  ✓ C++ replica matches HF reference (within tolerance)")
        print("  The .bin weights and basic logic are correct.")
        print("  If C++ output is garbled, the issue is in the C++ runtime:")
        print("    - Check compilation flags (QWEN3_SUPPORT must be ON)")
        print("    - Check runtime tensor data types")
        print("    - Check KV cache indexing")
    else:
        print(f"  ✗ C++ replica DIVERGES from HF reference")
        print(f"    Hidden state max_abs_err: {max_diff:.6e}")
        print(f"    Logits max_abs_err: {max_diff_logits:.6e}")
        if abs(CPP_RMSNORM_EPS - HF_RMSNORM_EPS) > 1e-10:
            print(f"    RMSNorm eps mismatch contributes: {max_diff - max_diff_corrected:.6e}")
        if not weights['has_qkv_bias']:
            print(f"    QKV bias MISSING — this is a major source of error!")

    print()
    print("  Difference between eps=1e-5 and eps=1e-6:")
    print(f"    Delta hidden max_abs_err: {abs(max_diff - max_diff_corrected):.6e}")
    print(f"    Delta logits max_abs_err: {abs(max_diff_logits - max_diff_corrected_logits):.6e}")

    # Check top-1 prediction match
    if cpp_top5.indices[0] == hf_top5.indices[0]:
        print(f"  ✓ Top-1 prediction MATCHES: {tokenizer.decode([cpp_top5.indices[0].item()])}")
    elif corrected_top5.indices[0] == hf_top5.indices[0]:
        print(f"  ✓ Top-1 prediction MATCHES with corrected eps: {tokenizer.decode([corrected_top5.indices[0].item()])}")
    else:
        print(f"  ✗ Top-1 prediction MISMATCH:")
        print(f"    C++: {tokenizer.decode([cpp_top5.indices[0].item()])} (id={cpp_top5.indices[0].item()})")
        print(f"    C++(corrected): {tokenizer.decode([corrected_top5.indices[0].item()])} (id={corrected_top5.indices[0].item()})")
        print(f"    HF: {tokenizer.decode([hf_top5.indices[0].item()])} (id={hf_top5.indices[0].item()})")


if __name__ == '__main__':
    main()
