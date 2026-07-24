#!/usr/bin/env python3
"""
Compare my_cuda_vllm with HuggingFace transformers on Qwen2.5 models.

Usage:
    # Compare 0.5B FP32
    python3 tools/qwen_test.py --model_path /home/liangji/huggingface/Qwen2.5-0.5B-Instruct \
        --cpp_bin demo/qwen2.5_0.5b_instruct.bin --max_new_tokens 128

    # Compare 7B INT8
    python3 tools/qwen_test.py --model_path /home/liangji/huggingface/Qwen2.5-7B-Instruct-1M \
        --cpp_bin demo/qwen2.5_7b_instruct_int8.bin --cpp_quant --max_new_tokens 128
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_hf_model(model_path: str, dtype: str = "auto"):
    """Load HuggingFace model and tokenizer."""
    print(f"Loading HF model from {model_path} (dtype={dtype})...")
    t0 = time.perf_counter()

    if dtype == "fp32":
        td = torch.float32
    elif dtype == "fp16":
        td = torch.float16
    else:
        td = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=td, device_map="cuda", local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    load_ms = (time.perf_counter() - t0) * 1000
    print(f"  HF model loaded in {load_ms:.1f} ms")
    return model, tokenizer, load_ms


def build_chatml_prompt(user_input: str) -> str:
    """Build ChatML prompt matching C++ demo."""
    prompt = "<|im_start|>system\n"
    prompt += "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    prompt += "<|im_end|>\n"
    prompt += "<|im_start|>user\n"
    prompt += user_input
    prompt += "<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    return prompt


def run_hf_inference(model, tokenizer, prompt_text: str, max_new_tokens: int):
    """Run HF inference and measure performance."""
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": "你给我讲一个小红帽的故事"},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    prompt_tokens = inputs.input_ids.shape[1]

    print(f"  Prompt tokens: {prompt_tokens}")

    # Warmup
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=10, do_sample=False)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # Benchmark
    with torch.no_grad():
        t0 = time.perf_counter()
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

    output_ids = output[0][prompt_tokens:].tolist()
    output_tokens = len(output_ids)
    tpot = elapsed / max(output_tokens, 1) * 1000
    throughput = output_tokens / elapsed
    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)

    response = tokenizer.decode(output_ids, skip_special_tokens=True)

    return {
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "e2e_ms": elapsed * 1000,
        "tpot_ms": tpot,
        "throughput_tps": throughput,
        "peak_memory_mb": peak_mem,
        "output_ids": output_ids,
        "response": response,
    }


def run_cpp_inference(cpp_bin: str, model_path: str, tokenizer_path: str,
                      max_new_tokens: int, quant: bool = False):
    """Run my_cuda_vllm inference and parse benchmark JSON."""
    args = [
        cpp_bin,
        "--model", model_path,
        "--tokenizer", tokenizer_path,
        "--benchmark",
        "--max-new-tokens", str(max_new_tokens),
        "--warmup", "3",
        "--repeat", "5",
        "--greedy",
        "--no-early-stop",
        "--no-stream-output",
    ]
    if quant:
        args.append("--quant")

    result_path = "/tmp/cpp_bench_result.json"
    args.extend(["--output", result_path])

    print(f"  Running: {' '.join(args)}")
    subprocess.run(args, check=True, capture_output=True)

    with open(result_path) as f:
        data = json.load(f)

    summary = data["summary"]
    runs = data.get("runs", [])

    return {
        "prompt_tokens": summary["prompt_tokens"],
        "output_tokens": summary["output_tokens"],
        "e2e_ms": summary["end_to_end_latency_ms"],
        "prefill_ms": summary["prefill_time_ms"],
        "decode_ms": summary["decode_time_ms"],
        "tpot_ms": summary["tpot_ms"],
        "throughput_tps": summary["end_to_end_throughput_tokens_per_s"],
        "prefill_throughput_tps": summary["prefill_throughput_tokens_per_s"],
        "decode_throughput_tps": summary["decode_throughput_tokens_per_s"],
        "peak_memory_mb": summary["gpu_memory_peak_mb"],
        "model_load_ms": summary["model_load_time_ms"],
        "p50_itl_ms": summary.get("p50_itl_ms", 0),
        "p95_itl_ms": summary.get("p95_itl_ms", 0),
    }


def main():
    parser = argparse.ArgumentParser(description="Compare my_cuda_vllm vs HF transformers")
    parser.add_argument("--model_path", required=True, help="HF model directory")
    parser.add_argument("--cpp_bin", required=True, help="my_cuda_vllm .bin file")
    parser.add_argument("--cpp_quant", action="store_true", help="Use INT8 quantized model")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--hf_dtype", default="auto", help="HF dtype: fp32, fp16, auto")
    parser.add_argument("--prompt", default="你给我讲一个小红帽的故事")
    args = parser.parse_args()

    print("=" * 70)
    print("Qwen2.5 Inference Comparison: my_cuda_vllm vs HuggingFace transformers")
    print("=" * 70)
    print(f"Model: {args.model_path}")
    print(f"Prompt: {args.prompt}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"C++ quant: {args.cpp_quant}")
    print()

    # ── HF Inference ──
    print("[1/2] HuggingFace transformers...")
    model, tokenizer, hf_load_ms = load_hf_model(args.model_path, args.hf_dtype)
    hf = run_hf_inference(model, tokenizer, args.prompt, args.max_new_tokens)

    # ── C++ Inference ──
    print("\n[2/2] my_cuda_vllm...")
    cpp_bin = os.path.join(os.path.dirname(os.path.dirname(__file__)), "bin", "demo")
    tokenizer_path = os.path.join(args.model_path, "tokenizer.json")
    if not os.path.exists(tokenizer_path):
        tokenizer_path = args.model_path  # fallback

    cpp = run_cpp_inference(cpp_bin, args.cpp_bin, tokenizer_path,
                            args.max_new_tokens, args.cpp_quant)

    # ── Compare ──
    print("\n" + "=" * 70)
    print("Comparison Summary")
    print("=" * 70)
    print(f"{'Metric':<28} {'HF Transformers':>18} {'my_cuda_vllm':>18}")
    print("-" * 70)

    def compare_row(name, hf_val, cpp_val, unit="", lower_is_better=True):
        if isinstance(hf_val, float) and isinstance(cpp_val, float) and hf_val > 0:
            ratio = cpp_val / hf_val
            direction = "x" if not lower_is_better else "x"
            if ratio > 1:
                note = f"  ({ratio:.2f}{direction} slower)" if lower_is_better else f"  ({ratio:.2f}{direction} faster)"
            else:
                note = f"  ({1/ratio:.2f}{direction} faster)" if lower_is_better else f"  ({1/ratio:.2f}{direction} slower)"
        else:
            note = ""
        print(f"{name:<28} {hf_val:>12.2f}{unit:>6} {cpp_val:>12.2f}{unit:>6}{note}")

    compare_row("Model Load (ms)", hf_load_ms, cpp["model_load_ms"], "")
    compare_row("Prompt Tokens", hf["prompt_tokens"], cpp["prompt_tokens"], "")
    compare_row("Output Tokens", hf["output_tokens"], cpp["output_tokens"], "")
    compare_row("E2E Latency (ms)", hf["e2e_ms"], cpp["e2e_ms"], "")
    compare_row("TPOT (ms)", hf["tpot_ms"], cpp["tpot_ms"], "")
    compare_row("Throughput (tok/s)", hf["throughput_tps"], cpp["throughput_tps"], "")
    compare_row("Peak GPU Memory (MB)", hf["peak_memory_mb"], cpp["peak_memory_mb"], "")
    if "p50_itl_ms" in cpp:
        print(f"{'ITL P50 (ms)':<28} {'N/A':>18} {cpp['p50_itl_ms']:>12.2f}{'':>6}")
    if "p95_itl_ms" in cpp:
        print(f"{'ITL P95 (ms)':<28} {'N/A':>18} {cpp['p95_itl_ms']:>12.2f}{'':>6}")

    print("-" * 70)
    print(f"{'Prefill (ms)':<28} {'N/A':>18} {cpp.get('prefill_ms', 0):>12.2f}{'':>6}")
    print(f"{'Decode (ms)':<28} {'N/A':>18} {cpp.get('decode_ms', 0):>12.2f}{'':>6}")
    print(f"{'Prefill Thr. (tok/s)':<28} {'N/A':>18} {cpp.get('prefill_throughput_tps', 0):>12.1f}{'':>6}")
    print(f"{'Decode Thr. (tok/s)':<28} {'N/A':>18} {cpp.get('decode_throughput_tps', 0):>12.1f}{'':>6}")

    # ── Response quality ──
    print("\n" + "=" * 70)
    print("Response Comparison")
    print("=" * 70)
    print(f"\n[HF Response] ({hf['output_tokens']} tokens):")
    print(hf["response"][:500])
    print(f"\n[HF Token IDs (first 20)]: {hf['output_ids'][:20]}")

    print("-" * 70)
    print(f"\nToken overlap check: HF output_tokens={hf['output_tokens']}, "
          f"C++ output_tokens={cpp['output_tokens']}")


if __name__ == "__main__":
    main()
