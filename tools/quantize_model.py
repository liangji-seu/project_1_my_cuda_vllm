#!/usr/bin/env python3
"""
INT8 weight-only quantization — matches teacher's KuiperLLama layout.

Per-group symmetric quantization (group_size=64), interleaved layout:
  [INT8 weight | float scales] for each tensor, then FP32 embedding/rmsnorm.

Usage:
    python3 tools/quantize_model.py demo/qwen2.5_0.5b_instruct.bin -o demo/qwen2.5_0.5b_instruct_int8.bin
"""

import argparse
import os
import struct
import sys

import numpy as np

MODEL_CONFIG_FMT = "iiiiiiii"
MODEL_CONFIG_SIZE = struct.calcsize(MODEL_CONFIG_FMT)

FLAG_HAS_QKV_BIAS = 1 << 0
FLAG_HAS_QK_NORM  = 1 << 1
FLAG_HAS_O_BIAS   = 1 << 2
FLAG_HAS_MLP_BIAS = 1 << 3
FLAG_TIED_WEIGHTS = 1 << 4

GROUP_SIZE = 64


def load_fp32_bin(path):
    with open(path, "rb") as f:
        config = struct.unpack(MODEL_CONFIG_FMT, f.read(MODEL_CONFIG_SIZE))
        flags_raw = f.read(4)
        flags = struct.unpack("i", flags_raw)[0] if len(flags_raw) == 4 else 0

    dim, hidden_dim, layer_num, head_num, kv_head_num, vocab_size_raw, seq_len, head_dim = config
    vocab_size = abs(vocab_size_raw)
    tied_embeddings = (flags & FLAG_TIED_WEIGHTS) != 0
    has_bias = (flags & FLAG_HAS_QKV_BIAS) != 0
    has_qk_norm = (flags & FLAG_HAS_QK_NORM) != 0
    kv_dim = kv_head_num * head_dim
    q_dim = head_num * head_dim

    config_dict = {
        "dim": dim, "hidden_dim": hidden_dim, "layer_num": layer_num,
        "head_num": head_num, "kv_head_num": kv_head_num,
        "vocab_size": vocab_size, "seq_len": seq_len, "head_dim": head_dim,
        "flags": flags, "tied_embeddings": tied_embeddings,
        "has_bias": has_bias, "has_qk_norm": has_qk_norm,
        "kv_dim": kv_dim, "q_dim": q_dim,
    }
    print(f"Model: dim={dim}, hidden_dim={hidden_dim}, layers={layer_num}, "
          f"heads={head_num}(kv={kv_head_num}), vocab={vocab_size}")

    with open(path, "rb") as f:
        f.seek(MODEL_CONFIG_SIZE + 4)
        all_data = f.read()
    all_floats = np.frombuffer(all_data, dtype=np.float32).copy()

    offset = 0
    weights = {}

    # 1. Embedding
    emb_size = vocab_size * dim
    weights["embedding"] = all_floats[offset:offset + emb_size].reshape(vocab_size, dim)
    offset += emb_size

    # 2. Attention RMSNorm
    attn_rms_size = layer_num * dim
    weights["attn_rmsnorm"] = all_floats[offset:offset + attn_rms_size].reshape(layer_num, dim)
    offset += attn_rms_size

    # 3. Wq
    wq_size = layer_num * q_dim * dim
    weights["wq"] = all_floats[offset:offset + wq_size].reshape(layer_num, q_dim, dim)
    offset += wq_size
    wq_bias = None
    if has_bias:
        wq_bias_size = layer_num * q_dim
        wq_bias = all_floats[offset:offset + wq_bias_size].reshape(layer_num, q_dim)
        offset += wq_bias_size

    # 4. Wk
    wk_size = layer_num * kv_dim * dim
    weights["wk"] = all_floats[offset:offset + wk_size].reshape(layer_num, kv_dim, dim)
    offset += wk_size
    wk_bias = None
    if has_bias:
        wk_bias_size = layer_num * kv_dim
        wk_bias = all_floats[offset:offset + wk_bias_size].reshape(layer_num, kv_dim)
        offset += wk_bias_size

    # 5. Wv
    wv_size = layer_num * kv_dim * dim
    weights["wv"] = all_floats[offset:offset + wv_size].reshape(layer_num, kv_dim, dim)
    offset += wv_size
    wv_bias = None
    if has_bias:
        wv_bias_size = layer_num * kv_dim
        wv_bias = all_floats[offset:offset + wv_bias_size].reshape(layer_num, kv_dim)
        offset += wv_bias_size

    # 6. Wo
    wo_size = layer_num * dim * q_dim
    weights["wo"] = all_floats[offset:offset + wo_size].reshape(layer_num, dim, q_dim)
    offset += wo_size

    # 7. FFN RMSNorm
    ffn_rms_size = layer_num * dim
    weights["ffn_rmsnorm"] = all_floats[offset:offset + ffn_rms_size].reshape(layer_num, dim)
    offset += ffn_rms_size

    # 8. W1
    w1_size = layer_num * hidden_dim * dim
    weights["w1"] = all_floats[offset:offset + w1_size].reshape(layer_num, hidden_dim, dim)
    offset += w1_size

    # 9. W2
    w2_size = layer_num * dim * hidden_dim
    weights["w2"] = all_floats[offset:offset + w2_size].reshape(layer_num, dim, hidden_dim)
    offset += w2_size

    # 10. W3
    w3_size = layer_num * hidden_dim * dim
    weights["w3"] = all_floats[offset:offset + w3_size].reshape(layer_num, hidden_dim, dim)
    offset += w3_size

    # 11. Final RMSNorm
    weights["final_rmsnorm"] = all_floats[offset:offset + dim]
    offset += dim

    # 12. freqs
    freqs_size = seq_len * head_dim
    weights["freqs_cos"] = all_floats[offset:offset + freqs_size].reshape(seq_len, head_dim)
    offset += freqs_size
    weights["freqs_sin"] = all_floats[offset:offset + freqs_size].reshape(seq_len, head_dim)
    offset += freqs_size

    # 13. CLS
    cls_weight = None
    if not tied_embeddings:
        cls_size = vocab_size * dim
        cls_weight = all_floats[offset:offset + cls_size].reshape(vocab_size, dim)
        offset += cls_size

    # 14. Q/K norm
    q_norm = None
    k_norm = None
    if has_qk_norm:
        q_norm_size = layer_num * head_dim
        q_norm = all_floats[offset:offset + q_norm_size].reshape(layer_num, head_dim)
        offset += q_norm_size
        k_norm = all_floats[offset:offset + q_norm_size].reshape(layer_num, head_dim)
        offset += q_norm_size

    return config_dict, weights, {
        "wq_bias": wq_bias, "wk_bias": wk_bias, "wv_bias": wv_bias,
        "cls_weight": cls_weight, "q_norm": q_norm, "k_norm": k_norm,
    }


def quantize_q80(w_fp32: np.ndarray, group_size: int = 64):
    """
    Per-group symmetric INT8 quantization over flattened weights.
    Matches teacher's quantize_q80 exactly.

    Args:
        w_fp32: FP32 weight matrix [N, K]
        group_size: elements per quantization group
    Returns:
        w_int8: int8 quantized weights, flattened [N*K]
        scales: per-group scales, [N*K/group_size]
    """
    assert w_fp32.size % group_size == 0, \
        f"Weight size {w_fp32.size} not divisible by group_size {group_size}"

    w_flat = w_fp32.ravel()
    w_reshaped = w_flat.reshape(-1, group_size)  # [num_groups, group_size]

    # Per-group max absolute value
    wmax = np.max(np.abs(w_reshaped), axis=1)  # [num_groups]
    wmax = np.maximum(wmax, 1e-12)
    scales = (wmax / 127.0).astype(np.float32)  # [num_groups]

    # Quantize
    quant = w_reshaped / scales[:, np.newaxis]
    int8val = np.clip(np.round(quant), -127, 127).astype(np.int8)

    return int8val.ravel(), scales


def write_int8_bin(output_path, config, weights, extras):
    dim = config["dim"]
    hidden_dim = config["hidden_dim"]
    layer_num = config["layer_num"]
    q_dim = config["q_dim"]
    kv_dim = config["kv_dim"]
    vocab_size = config["vocab_size"]
    seq_len = config["seq_len"]
    head_dim = config["head_dim"]
    flags = config["flags"]
    has_bias = config["has_bias"]
    has_qk_norm = config["has_qk_norm"]
    tied = config["tied_embeddings"]

    print(f"\nQuantizing ALL matmul layers (per-group, group_size={GROUP_SIZE})...")

    # Quantize each weight matrix per-layer (for interleaved layout)
    quantized_per_layer = {}  # key -> [(int8_0, scales_0), (int8_1, scales_1), ...]
    for key in ["wq", "wk", "wv", "wo", "w1", "w2", "w3"]:
        w_fp32 = weights[key]  # [L, N, K]
        L, N, K = w_fp32.shape
        quantized_per_layer[key] = []
        total_elems = 0
        total_scales = 0
        for i in range(L):
            wi, si = quantize_q80(w_fp32[i], GROUP_SIZE)
            quantized_per_layer[key].append((wi, si))
            total_elems += wi.size
            total_scales += si.size
        print(f"  {key}: [{L}×{N}×{K}] → INT8 {total_elems} elems, scales {total_scales}")

    # CLS
    cls_per_layer = None
    if not tied:
        cls_fp32 = extras["cls_weight"]
        cls_int8, cls_scales = quantize_q80(cls_fp32, GROUP_SIZE)
        cls_per_layer = (cls_int8, cls_scales)
        print(f"  cls: [{vocab_size}×{dim}] → INT8 {cls_int8.size} elems, scales {cls_scales.size}")
    else:
        print("  cls: shared with embedding (FP32, stored at embedding position)")

    # ================================================================
    # Write .bin — per-layer interleaved layout
    # [Wq[0]INT8|Wq[0]scales|Wq[1]INT8|Wq[1]scales|...|Wk[0]INT8|Wk[0]scales|...]
    # ================================================================
    with open(output_path, "wb") as f:
        legacy_vocab_size = vocab_size if tied else -vocab_size
        header = struct.pack(
            MODEL_CONFIG_FMT, dim, hidden_dim, layer_num,
            config["head_num"], config["kv_head_num"],
            legacy_vocab_size, seq_len,
            head_dim,  # actual head_dim (NOT group_size!)
        )
        f.write(header)
        f.write(struct.pack("i", flags))
        f.write(struct.pack("i", GROUP_SIZE))
        f.write(struct.pack("q", 0))

        # ── Per-layer interleaved INT8 + scales ──
        for key in ["wq", "wk", "wv", "wo", "w1", "w2", "w3"]:
            for w_int8, w_scales in quantized_per_layer[key]:
                f.write(w_int8.tobytes())
                f.write(w_scales.tobytes())

        # CLS
        if not tied and cls_per_layer is not None:
            f.write(cls_per_layer[0].tobytes())
            f.write(cls_per_layer[1].tobytes())

        # ── FP32 weights ──
        # Embedding
        f.write(weights["embedding"].ravel().astype(np.float32).tobytes())

        # Attention RMSNorm
        f.write(weights["attn_rmsnorm"].ravel().astype(np.float32).tobytes())

        # Wq/Wk/Wv bias (FP32)
        if has_bias:
            for bias_key in ["wq_bias", "wk_bias", "wv_bias"]:
                if extras[bias_key] is not None:
                    f.write(extras[bias_key].ravel().astype(np.float32).tobytes())

        # FFN RMSNorm
        f.write(weights["ffn_rmsnorm"].ravel().astype(np.float32).tobytes())

        # Final RMSNorm
        f.write(weights["final_rmsnorm"].ravel().astype(np.float32).tobytes())

        # RoPE freqs
        f.write(weights["freqs_cos"].ravel().astype(np.float32).tobytes())
        f.write(weights["freqs_sin"].ravel().astype(np.float32).tobytes())

        # Q/K norm
        if has_qk_norm and extras["q_norm"] is not None:
            f.write(extras["q_norm"].ravel().astype(np.float32).tobytes())
            f.write(extras["k_norm"].ravel().astype(np.float32).tobytes())

    file_size = os.path.getsize(output_path)
    total_int8_bytes = sum(
        sum(w_int8.nbytes for w_int8, _ in layers)
        for layers in quantized_per_layer.values()
    )
    print(f"\nOutput: {output_path} ({file_size / (1024**3):.2f} GB)")
    print(f"  INT8 data: {total_int8_bytes / (1024**2):.1f} MB")
    print(f"  Compression vs FP32 matmul weights: {file_size / (1024**3):.2f} GB total")


def main():
    parser = argparse.ArgumentParser(description="INT8 quantization for my_cuda_vllm")
    parser.add_argument("input", help="FP32 .bin file path")
    parser.add_argument("--output", "-o", required=True, help="Output INT8 .bin file path")
    args = parser.parse_args()

    print(f"Loading FP32 model: {args.input}")
    config, weights, extras = load_fp32_bin(args.input)
    write_int8_bin(args.output, config, weights, extras)


if __name__ == "__main__":
    main()
