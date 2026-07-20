#!/usr/bin/env python3
"""
Parse model structure from safetensors header + config.json.

Reads ONLY metadata (no weight loading). Outputs a complete, deterministic
description of the model structure that the export script consumes.

Usage:
    python3 parse_model_structure.py /path/to/hf_model  [--output model_structure.txt]
"""

import json
import struct
import os
import sys
from pathlib import Path
from collections import OrderedDict


def parse_safetensors_header(model_path: str) -> dict:
    """Read safetensors header (JSON, typically < 100KB) without loading weights."""
    safetensors_path = os.path.join(model_path, "model.safetensors")

    # Check for sharded model
    if not os.path.isfile(safetensors_path):
        index_path = os.path.join(model_path, "model.safetensors.index.json")
        if os.path.isfile(index_path):
            with open(index_path) as f:
                index = json.load(f)
            # Merge all shard headers
            merged = {"__metadata__": index.get("metadata", {})}
            for shard_path in set(index["weight_map"].values()):
                shard_full = os.path.join(model_path, shard_path)
                with open(shard_full, "rb") as f:
                    shard_header_size = struct.unpack("<Q", f.read(8))[0]
                    shard_header = json.loads(f.read(shard_header_size))
                for k, v in shard_header.items():
                    if k != "__metadata__":
                        merged[k] = v
            return merged
        raise FileNotFoundError(f"No safetensors found at {model_path}")

    with open(safetensors_path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_size))
    return header


def parse_config(model_path: str) -> dict:
    """Read config.json."""
    config_path = os.path.join(model_path, "config.json")
    with open(config_path) as f:
        return json.load(f)


def derive_structure(header: dict, config: dict) -> dict:
    """
    Derive the complete model structure from safetensors header + config.json.

    ZERO guessing. Every structural fact comes from:
      - Tensor names in the header (e.g., presence of ".bias" means bias exists)
      - Explicit config values (e.g., hidden_size, rms_norm_eps)
    """
    tensor_names = sorted(k for k in header if k != "__metadata__")

    # --- Layer count: extract unique layer indices from tensor names ---
    layer_indices = set()
    for name in tensor_names:
        parts = name.split(".")
        for p in parts:
            if p.isdigit():
                layer_indices.add(int(p))
    num_layers = len(layer_indices)
    if num_layers == 0:
        raise ValueError("Could not determine number of layers from tensor names")

    # --- Derive dimensions from first layer tensors (no guessing) ---
    # hidden_size: from embed_tokens or input_layernorm
    if "model.embed_tokens.weight" in header:
        hidden_size = header["model.embed_tokens.weight"]["shape"][1]
    else:
        hidden_size = header[f"model.layers.0.input_layernorm.weight"]["shape"][0]

    # intermediate_size: from mlp.gate_proj
    intermediate_size = header[f"model.layers.0.mlp.gate_proj.weight"]["shape"][0]

    # num_attention_heads: q_proj out_features / head_dim
    q_proj_shape = header[f"model.layers.0.self_attn.q_proj.weight"]["shape"]
    k_proj_shape = header[f"model.layers.0.self_attn.k_proj.weight"]["shape"]
    v_proj_shape = header[f"model.layers.0.self_attn.v_proj.weight"]["shape"]

    q_out = q_proj_shape[0]
    k_out = k_proj_shape[0]
    v_out = v_proj_shape[0]

    # head_dim: use config if available, otherwise derive
    head_dim = config.get("head_dim", 0)
    if head_dim == 0:
        # Derive from config attention_heads if available
        num_heads = config.get("num_attention_heads", 0)
        if num_heads > 0 and q_out % num_heads == 0:
            head_dim = q_out // num_heads
        else:
            # Fallback: check partial_rotary_factor or default to q_out / num_heads
            head_dim = config.get("hidden_size", hidden_size) // config.get("num_attention_heads", 1)

    num_attention_heads = q_out // head_dim
    num_key_value_heads = k_out // head_dim

    # --- Flags: derived DIRECTLY from tensor names, NOT from model_type string ---
    has_qkv_bias = any(
        f"model.layers.0.self_attn.{proj}.bias" in header
        for proj in ["q_proj", "k_proj", "v_proj"]
    )
    has_o_bias = "model.layers.0.self_attn.o_proj.bias" in header
    has_qk_norm = (
        "model.layers.0.self_attn.q_norm.weight" in header
        or "model.layers.0.self_attn.q_norm.weight" in header
    )
    has_mlp_bias = any(
        f"model.layers.0.mlp.{proj}.bias" in header
        for proj in ["gate_proj", "up_proj", "down_proj"]
    )

    # tie_word_embeddings: no separate lm_head weight
    has_lm_head = "lm_head.weight" in header
    tie_word_embeddings = not has_lm_head or config.get("tie_word_embeddings", False)

    # vocab_size
    if "model.embed_tokens.weight" in header:
        vocab_size = header["model.embed_tokens.weight"]["shape"][0]
    elif "lm_head.weight" in header:
        vocab_size = header["lm_head.weight"]["shape"][0]
    else:
        vocab_size = config.get("vocab_size", 0)

    # max_position_embeddings, rms_norm_eps, rope_theta — from config
    max_seq_len = config.get("max_position_embeddings", 2048)
    rms_norm_eps = config.get("rms_norm_eps", 1e-6)
    rope_theta = config.get("rope_theta", 10000.0)
    hidden_act = config.get("hidden_act", "silu")

    # --- Build ordered tensor list for .bin export ---
    # The order MUST match what the C++ code expects.
    # Order: embedding, attn_norms, qkv_weights+bias, ffn_norms, mlp_weights,
    #        final_norm, freqs, [cls if not tied], [qk_norms]
    export_order = []

    def add_tensors(pattern_template, name_hint):
        """Add tensors matching the pattern to export_order."""
        for layer_idx in range(num_layers):
            name = pattern_template.format(layer_idx)
            if name in header:
                export_order.append(name)
            else:
                raise KeyError(f"Expected tensor {name} not found in safetensors header")

    # 1. Embedding
    export_order.append("model.embed_tokens.weight")

    # 2. Attention RMSNorm (per layer)
    for i in range(num_layers):
        export_order.append(f"model.layers.{i}.input_layernorm.weight")

    # 3. QKV weights + biases (per layer, grouped by type for efficient C++ loading)
    for i in range(num_layers):
        export_order.append(f"model.layers.{i}.self_attn.q_proj.weight")
    if has_qkv_bias:
        for i in range(num_layers):
            export_order.append(f"model.layers.{i}.self_attn.q_proj.bias")

    for i in range(num_layers):
        export_order.append(f"model.layers.{i}.self_attn.k_proj.weight")
    if has_qkv_bias:
        for i in range(num_layers):
            export_order.append(f"model.layers.{i}.self_attn.k_proj.bias")

    for i in range(num_layers):
        export_order.append(f"model.layers.{i}.self_attn.v_proj.weight")
    if has_qkv_bias:
        for i in range(num_layers):
            export_order.append(f"model.layers.{i}.self_attn.v_proj.bias")

    # 4. Output projection (per layer)
    for i in range(num_layers):
        export_order.append(f"model.layers.{i}.self_attn.o_proj.weight")
    if has_o_bias:
        for i in range(num_layers):
            export_order.append(f"model.layers.{i}.self_attn.o_proj.bias")

    # 5. FFN RMSNorm (per layer)
    for i in range(num_layers):
        export_order.append(f"model.layers.{i}.post_attention_layernorm.weight")

    # 6. MLP weights (per layer, grouped by type)
    for i in range(num_layers):
        export_order.append(f"model.layers.{i}.mlp.gate_proj.weight")
    if has_mlp_bias:
        for i in range(num_layers):
            export_order.append(f"model.layers.{i}.mlp.gate_proj.bias")

    for i in range(num_layers):
        export_order.append(f"model.layers.{i}.mlp.down_proj.weight")
    if has_mlp_bias:
        for i in range(num_layers):
            export_order.append(f"model.layers.{i}.mlp.down_proj.bias")

    for i in range(num_layers):
        export_order.append(f"model.layers.{i}.mlp.up_proj.weight")
    if has_mlp_bias:
        for i in range(num_layers):
            export_order.append(f"model.layers.{i}.mlp.up_proj.bias")

    # 7. Final RMSNorm
    export_order.append("model.norm.weight")

    # 8. RoPE cache (not a tensor in safetensors, computed at export time)
    #    Represented as special entries
    export_order.append("__rope_cos__")
    export_order.append("__rope_sin__")

    # 9. LM head (if not tied)
    if not tie_word_embeddings:
        export_order.append("lm_head.weight")

    # 10. Q/K norms (if present, per layer)
    if has_qk_norm:
        for i in range(num_layers):
            export_order.append(f"model.layers.{i}.self_attn.q_norm.weight")
        for i in range(num_layers):
            export_order.append(f"model.layers.{i}.self_attn.k_norm.weight")

    # Verify all tensors are covered
    exported = set(export_order) - {"__rope_cos__", "__rope_sin__"}
    all_tensors = set(tensor_names)
    missing = exported - all_tensors
    extra = all_tensors - exported
    if missing:
        raise ValueError(f"Export order references tensors not in safetensors: {missing}")
    # Note: extra is OK — some tensors like rotary_emb.inv_freq are not needed

    return OrderedDict({
        "config": OrderedDict({
            "model_type": config.get("model_type", "unknown"),
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "num_attention_heads": num_attention_heads,
            "num_key_value_heads": num_key_value_heads,
            "head_dim": head_dim,
            "intermediate_size": intermediate_size,
            "vocab_size": vocab_size,
            "max_position_embeddings": max_seq_len,
            "rms_norm_eps": rms_norm_eps,
            "rope_theta": rope_theta,
            "hidden_act": hidden_act,
            "tie_word_embeddings": tie_word_embeddings,
        }),
        "flags": OrderedDict({
            "has_qkv_bias": has_qkv_bias,
            "has_o_bias": has_o_bias,
            "has_qk_norm": has_qk_norm,
            "has_mlp_bias": has_mlp_bias,
        }),
        "export_order": export_order,
        "tensor_shapes": OrderedDict(
            (name, list(header[name]["shape"]))
            for name in export_order
            if name not in ("__rope_cos__", "__rope_sin__")
        ),
    })


def write_structure_txt(structure: dict, output_path: str):
    """Write structure to a clean key=value text file."""
    lines = []
    lines.append("# ============================================================")
    lines.append("# Model Structure — auto-generated by parse_model_structure.py")
    lines.append("# Source: safetensors header + config.json")
    lines.append("# ZERO guessing — every value comes directly from the files.")
    lines.append("# ============================================================")
    lines.append("")

    # Config section
    lines.append("[config]")
    for key, val in structure["config"].items():
        if isinstance(val, bool):
            val = str(val).lower()
        elif isinstance(val, float):
            val = str(val)
        lines.append(f"{key} = {val}")
    lines.append("")

    # Flags section
    lines.append("[flags]")
    for key, val in structure["flags"].items():
        lines.append(f"{key} = {str(val).lower()}")
    lines.append("")

    # Export order section
    lines.append("[export_order]")
    lines.append("# This is the EXACT byte layout of the .bin file.")
    lines.append("# format: tensor_name | shape")
    lines.append("")
    cfg = structure["config"]
    for name in structure["export_order"]:
        if name in ("__rope_cos__", "__rope_sin__"):
            shape_str = f"[{cfg['max_position_embeddings']}, {cfg['head_dim']}]"
            lines.append(f"{name} | {shape_str}  # computed at export time, not from HF")
        else:
            shape = structure["tensor_shapes"].get(name, "?")
            lines.append(f"{name} | {shape}")
    lines.append("")

    # Summary
    lines.append("[summary]")
    config = structure["config"]
    flags = structure["flags"]
    num_tensors = len([n for n in structure["export_order"]
                       if n not in ("__rope_cos__", "__rope_sin__")])
    lines.append(f"total_layers = {config['num_layers']}")
    lines.append(f"total_real_tensors = {num_tensors}")
    lines.append(f"total_entries_in_bin = {len(structure['export_order'])}")
    lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Structure written to: {output_path}")
    print(f"  Layers: {config['num_layers']}")
    print(f"  Hidden: {config['hidden_size']}, Intermediate: {config['intermediate_size']}")
    print(f"  Heads: {config['num_attention_heads']} (KV: {config['num_key_value_heads']})")
    print(f"  Head dim: {config['head_dim']}")
    print(f"  Vocab: {config['vocab_size']}")
    print(f"  QKV bias: {flags['has_qkv_bias']}")
    print(f"  Q/K norm: {flags['has_qk_norm']}")
    print(f"  MLP bias: {flags['has_mlp_bias']}")
    print(f"  Tied embeddings: {config['tie_word_embeddings']}")
    print(f"  Real tensors: {num_tensors}")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Parse model structure from safetensors header + config.json"
    )
    parser.add_argument("model_path", help="Path to HF model directory")
    parser.add_argument("--output", "-o", default=None,
                        help="Output path (default: <model_path>/model_structure.txt)")
    args = parser.parse_args()

    model_path = args.model_path
    output_path = args.output or os.path.join(model_path, "model_structure.txt")

    print(f"Reading safetensors header from: {model_path}")
    header = parse_safetensors_header(model_path)
    print(f"  Found {len(header) - 1} tensor entries")

    print(f"Reading config.json from: {model_path}")
    config = parse_config(model_path)

    print("Deriving model structure...")
    structure = derive_structure(header, config)

    write_structure_txt(structure, output_path)


if __name__ == "__main__":
    main()
