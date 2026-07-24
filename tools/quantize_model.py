#!/usr/bin/env python3
"""
Per-channel symmetric INT8 weight-only quantization for matmul weights.

Reads FP32 .bin, quantizes every matmul weight (Wq/Wk/Wv/Wo/W1/W2/W3/CLS) as:
    scale[i] = max(|row[i]|) / 127
    W_int8[i] = round(W_fp32[i] / scale[i])

Embedding, RMSNorm, RoPE, and bias are kept FP32.

Output .bin layout (matches create_param_quant_layers() in llama3.cpp):
  Header: ModelConfig(32B) | flags(4B) | group_size(4B) | total_int8_elems(8B)
  [int8 weights  series] — all quantized matmul weights
  [float scales series]  — per-channel scales for each INT8 weight
  [float non-quant]      — embedding, rmsnorm, rope, bias (same order as FP32 .bin)

Usage:
    python3 tools/quantize_model.py demo/qwen2.5_0.5b_instruct.bin -o demo/qwen2.5_0.5b_instruct_int8.bin
"""

import argparse
import os
import struct
import sys

import numpy as np


# ── ModelConfig (from config.h) ──────────────────────────────
#   int32_t dim, hidden_dim, layer_num, head_num, kv_head_num, vocab_size, seq_len, head_dim
MODEL_CONFIG_FMT = "iiiiiiii"
MODEL_CONFIG_SIZE = struct.calcsize(MODEL_CONFIG_FMT)  # 32 bytes

# ── Flags (from config.h) ────────────────────────────────────
FLAG_HAS_QKV_BIAS = 1 << 0
FLAG_HAS_QK_NORM  = 1 << 1
FLAG_HAS_O_BIAS   = 1 << 2
FLAG_HAS_MLP_BIAS = 1 << 3
FLAG_TIED_WEIGHTS = 1 << 4


def load_fp32_bin(path):
    """Load FP32 .bin into numpy arrays in deterministic order."""
    with open(path, "rb") as f:
        # Parse header
        header_raw = f.read(MODEL_CONFIG_SIZE)
        config = struct.unpack(MODEL_CONFIG_FMT, header_raw)
        dim, hidden_dim, layer_num, head_num, kv_head_num, vocab_size_raw, seq_len, head_dim = config

        # Read flags
        flags_raw = f.read(4)
        flags = struct.unpack("i", flags_raw)[0]

        # Determine if there's extra header for quant (group_size)
        # For FP32 non-quant .bin, there's no group_size field after flags.
        # The file pointer is now at the start of weight data.

    # Derived config
    vocab_size = abs(vocab_size_raw)
    tied_embeddings = (flags & FLAG_TIED_WEIGHTS) != 0
    has_bias = (flags & FLAG_HAS_QKV_BIAS) != 0
    has_qk_norm = (flags & FLAG_HAS_QK_NORM) != 0
    kv_dim = kv_head_num * head_dim
    q_dim = head_num * head_dim

    config_dict = {
        "dim": dim,
        "hidden_dim": hidden_dim,
        "layer_num": layer_num,
        "head_num": head_num,
        "kv_head_num": kv_head_num,
        "vocab_size": vocab_size,
        "seq_len": seq_len,
        "head_dim": head_dim,
        "flags": flags,
        "tied_embeddings": tied_embeddings,
        "has_bias": has_bias,
        "has_qk_norm": has_qk_norm,
        "kv_dim": kv_dim,
        "q_dim": q_dim,
    }
    print(f"Model: dim={dim}, hidden_dim={hidden_dim}, layers={layer_num}, "
          f"heads={head_num}(kv={kv_head_num}), vocab={vocab_size}")
    print(f"Flags: bias={has_bias}, qk_norm={has_qk_norm}, tied_embeddings={tied_embeddings}")

    # Load all weights as fp32 numpy arrays
    with open(path, "rb") as f:
        f.seek(MODEL_CONFIG_SIZE + 4)  # skip header + flags
        all_data = f.read()
    all_floats = np.frombuffer(all_data, dtype=np.float32).copy()

    offset = 0  # in float elements

    # Parse weights in the same order as create_param_layers()
    weights = {}

    # 1. Embedding [vocab_size, dim]
    emb_size = vocab_size * dim
    weights["embedding"] = all_floats[offset:offset + emb_size].reshape(vocab_size, dim)
    offset += emb_size

    # 2. Attention RMSNorm [layer_num, dim]
    attn_rms_size = layer_num * dim
    weights["attn_rmsnorm"] = all_floats[offset:offset + attn_rms_size].reshape(layer_num, dim)
    offset += attn_rms_size

    # 3. Wq [layer_num, q_dim, dim]
    wq_size = layer_num * q_dim * dim
    weights["wq"] = all_floats[offset:offset + wq_size].reshape(layer_num, q_dim, dim)
    offset += wq_size
    # Wq bias (if present)
    wq_bias = None
    if has_bias:
        wq_bias_size = layer_num * q_dim
        wq_bias = all_floats[offset:offset + wq_bias_size].reshape(layer_num, q_dim)
        offset += wq_bias_size

    # 4. Wk [layer_num, kv_dim, dim]
    wk_size = layer_num * kv_dim * dim
    weights["wk"] = all_floats[offset:offset + wk_size].reshape(layer_num, kv_dim, dim)
    offset += wk_size
    wk_bias = None
    if has_bias:
        wk_bias_size = layer_num * kv_dim
        wk_bias = all_floats[offset:offset + wk_bias_size].reshape(layer_num, kv_dim)
        offset += wk_bias_size

    # 5. Wv [layer_num, kv_dim, dim]
    wv_size = layer_num * kv_dim * dim
    weights["wv"] = all_floats[offset:offset + wv_size].reshape(layer_num, kv_dim, dim)
    offset += wv_size
    wv_bias = None
    if has_bias:
        wv_bias_size = layer_num * kv_dim
        wv_bias = all_floats[offset:offset + wv_bias_size].reshape(layer_num, kv_dim)
        offset += wv_bias_size

    # 6. Wo [layer_num, dim, q_dim]
    wo_size = layer_num * dim * q_dim
    weights["wo"] = all_floats[offset:offset + wo_size].reshape(layer_num, dim, q_dim)
    offset += wo_size

    # 7. FFN RMSNorm [layer_num, dim]
    ffn_rms_size = layer_num * dim
    weights["ffn_rmsnorm"] = all_floats[offset:offset + ffn_rms_size].reshape(layer_num, dim)
    offset += ffn_rms_size

    # 8. W1 (gate) [layer_num, hidden_dim, dim]
    w1_size = layer_num * hidden_dim * dim
    weights["w1"] = all_floats[offset:offset + w1_size].reshape(layer_num, hidden_dim, dim)
    offset += w1_size

    # 9. W2 (down) [layer_num, dim, hidden_dim]
    w2_size = layer_num * dim * hidden_dim
    weights["w2"] = all_floats[offset:offset + w2_size].reshape(layer_num, dim, hidden_dim)
    offset += w2_size

    # 10. W3 (up) [layer_num, hidden_dim, dim]
    w3_size = layer_num * hidden_dim * dim
    weights["w3"] = all_floats[offset:offset + w3_size].reshape(layer_num, hidden_dim, dim)
    offset += w3_size

    # 11. Final RMSNorm [dim]
    weights["final_rmsnorm"] = all_floats[offset:offset + dim]
    offset += dim

    # 12. freqs_cos, freqs_sin [seq_len, head_dim] each
    freqs_size = seq_len * head_dim
    weights["freqs_cos"] = all_floats[offset:offset + freqs_size].reshape(seq_len, head_dim)
    offset += freqs_size
    weights["freqs_sin"] = all_floats[offset:offset + freqs_size].reshape(seq_len, head_dim)
    offset += freqs_size

    # 13. CLS (if not tied)
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
        "cls_weight": cls_weight,
        "q_norm": q_norm, "k_norm": k_norm,
    }


def quantize_per_channel_symmetric(w_fp32: np.ndarray):
    """
    Per-channel symmetric INT8 quantization.

    Args:
        w_fp32: FP32 weight matrix [N, K]
    Returns:
        w_int8: int8 quantized weights [N, K]
        scales: per-channel scales [N]
    """
    N = w_fp32.shape[0]
    # per-channel max absolute value
    amax = np.max(np.abs(w_fp32), axis=1)  # [N]
    # Avoid division by zero
    amax = np.maximum(amax, 1e-12)
    scales = amax / 127.0

    # Quantize
    w_quant = w_fp32 / scales[:, np.newaxis]  # [N, K]
    w_int8 = np.clip(np.round(w_quant), -127, 127).astype(np.int8)

    return w_int8, scales.astype(np.float32)


def write_int8_bin(output_path, config, weights, extras):
    """Write INT8 .bin file."""
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

    # Quantize all matmul weights
    print("\nQuantizing weights...")
    quantized = {}
    total_int8_elems = 0
    total_scale_elems = 0

    for key in ["wq", "wk", "wv", "wo", "w1", "w2", "w3"]:
        w_fp32 = weights[key]  # [L, N, K]
        L, N, K = w_fp32.shape
        w_int8_list = []
        scale_list = []
        for i in range(L):
            wi, si = quantize_per_channel_symmetric(w_fp32[i])
            w_int8_list.append(wi.ravel())
            scale_list.append(si.ravel())
        quantized[key] = (
            np.concatenate(w_int8_list),    # [L*N*K] int8
            np.concatenate(scale_list),     # [L*N] float32
        )
        total_int8_elems += L * N * K
        total_scale_elems += L * N
        print(f"  {key}: [{L}×{N}×{K}] → int8 {L*N*K} elems, scales {L*N}")

    # CLS if not tied
    cls_int8 = None
    cls_scales = None
    if not tied:
        cls_fp32 = extras["cls_weight"]  # [vocab_size, dim]
        cls_int8, cls_scales = quantize_per_channel_symmetric(cls_fp32)
        cls_int8 = cls_int8.ravel()
        cls_scales = cls_scales.ravel()
        total_int8_elems += vocab_size * dim
        total_scale_elems += vocab_size
        print(f"  cls: [{vocab_size}×{dim}] → INT8 {vocab_size*dim} elems, scales {vocab_size}")
    else:
        print("  cls: shared with embedding (FP32)")

    print(f"\nTotal INT8 elements: {total_int8_elems}")
    print(f"Total scale elements: {total_scale_elems}")

    # Write .bin
    with open(output_path, "wb") as f:
        # ── Header ──
        legacy_vocab_size = vocab_size if tied else -vocab_size
        header = struct.pack(
            MODEL_CONFIG_FMT,
            dim, hidden_dim, layer_num,
            config["head_num"], config["kv_head_num"],
            legacy_vocab_size, seq_len, head_dim,
        )
        f.write(header)                          # 32 bytes
        f.write(struct.pack("i", flags))         # 4 bytes
        f.write(struct.pack("i", 0))             # group_size = 0 (not used)
        f.write(struct.pack("q", total_int8_elems))  # 8 bytes

        # ── INT8 weights section ──
        for key in ["wq", "wk", "wv", "wo", "w1", "w2", "w3"]:
            w_int8, _ = quantized[key]
            f.write(w_int8.tobytes())
        if cls_int8 is not None:
            f.write(cls_int8.tobytes())

        # ── FP32 scales section ──
        for key in ["wq", "wk", "wv", "wo", "w1", "w2", "w3"]:
            _, scales = quantized[key]
            f.write(scales.tobytes())
        if cls_scales is not None:
            f.write(cls_scales.tobytes())

        # ── FP32 non-quant weights section ──
        # Embedding
        f.write(weights["embedding"].ravel().astype(np.float32).tobytes())

        # Attention RMSNorm
        f.write(weights["attn_rmsnorm"].ravel().astype(np.float32).tobytes())

        # Wq/Wk/Wv bias
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

        # CLS (only if tied — shared with embedding, skip because we already wrote embedding)
        # If not tied, CLS is in the INT8 section already

        # Q/K norm
        if has_qk_norm and extras["q_norm"] is not None:
            f.write(extras["q_norm"].ravel().astype(np.float32).tobytes())
            f.write(extras["k_norm"].ravel().astype(np.float32).tobytes())

    file_size = os.path.getsize(output_path)
    fp32_size_est = total_int8_elems * 4  # what it would be in FP32
    actual_int8_size = total_int8_elems * 1  # actual INT8
    scale_size = total_scale_elems * 4
    print(f"\nOutput: {output_path} ({file_size / (1024**3):.2f} GB)")
    print(f"  INT8 weight section: {actual_int8_size / (1024**2):.1f} MB")
    print(f"  Scale section:       {scale_size / 1024:.1f} KB")
    print(f"  Compression ratio:   {fp32_size_est / (actual_int8_size + scale_size):.2f}x")


def main():
    parser = argparse.ArgumentParser(
        description="Per-channel symmetric INT8 weight-only quantization for my_cuda_vllm"
    )
    parser.add_argument("input", help="FP32 .bin file path")
    parser.add_argument("--output", "-o", required=True, help="Output INT8 .bin file path")
    args = parser.parse_args()

    print(f"Loading FP32 model: {args.input}")
    config, weights, extras = load_fp32_bin(args.input)

    write_int8_bin(args.output, config, weights, extras)


if __name__ == "__main__":
    main()
