#!/usr/bin/env python3
"""
PyTorch HuggingFace baseline — 与 C++ benchmark 同条件对比
匹配 test.sh: warmup=3, repeat=10, max_new_tokens=128, greedy
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
import json
import numpy as np
import os
import sys

MODEL_PATH  = "/home/liangji/huggingface/Qwen2.5-0.5B-Instruct"
PROMPT      = "你介绍一下东南大学"
MAX_NEW     = 128
WARMUP      = 3
REPEAT      = 10
OUTPUT_DIR  = "profile/results/baseline"

# ── Model load ──────────────────────────────────────────
t0 = time.perf_counter()
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float32, device_map="cuda", local_files_only=True
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model_load_ms = (time.perf_counter() - t0) * 1000

# ── ChatML prompt ───────────────────────────────────────
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user",   "content": PROMPT},
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to("cuda")
prompt_tokens = inputs.input_ids.shape[1]
print(f"Prompt tokens: {prompt_tokens}")

# ── GPU memory before ───────────────────────────────────
mem_before = torch.cuda.memory_allocated() / 1024**2

# ── Warmup ──────────────────────────────────────────────
print(f"Warmup ({WARMUP} iterations)...")
with torch.no_grad():
    for _ in range(WARMUP):
        _ = model.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=False)

# ── Benchmark ───────────────────────────────────────────
torch.cuda.reset_peak_memory_stats()
runs = []

print(f"Benchmark ({REPEAT} iterations)...")
for r in range(REPEAT):
    print(f"  Run {r+1}/{REPEAT}", end="", flush=True)
    torch.cuda.synchronize()
    run_start = time.perf_counter()

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=False)

    torch.cuda.synchronize()
    run_end = time.perf_counter()

    output_ids = output[0][prompt_tokens:]
    output_tokens = len(output_ids)
    e2e_ms     = (run_end - run_start) * 1000

    run_rec = {
        "run_index":        r,
        "prompt_tokens":    int(prompt_tokens),
        "output_tokens":    int(output_tokens),
        "e2e_ms":           round(e2e_ms, 4),
        "prefill_ms":       0,   # HF generate() 无法拆分 prefill/decode
        "decode_total_ms":  round(e2e_ms, 4),
        "tpot_ms":          round(e2e_ms / max(output_tokens, 1), 4),
    }
    runs.append(run_rec)
    print(f" — {output_tokens} tokens, {e2e_ms:.1f} ms")

# ── Memory ───────────────────────────────────────────────
mem_after = torch.cuda.memory_allocated() / 1024**2
mem_peak  = torch.cuda.max_memory_allocated() / 1024**2

# ── Aggregate ───────────────────────────────────────────
n      = len(runs)
avg_e2e  = sum(r["e2e_ms"]           for r in runs) / n
avg_out  = sum(r["output_tokens"]    for r in runs) / n
avg_tpot = sum(r["tpot_ms"]          for r in runs) / n

print(f"\n{'='*60}")
print(f"Inference Benchmark (PyTorch HF)")
print(f"{'='*60}")
print(f"Model:        {MODEL_PATH}")
print(f"GPU:          {torch.cuda.get_device_name(0)}")
print(f"Precision:    float32")
print(f"Prompt tokens:{int(prompt_tokens)}")
print(f"Output tokens:{int(avg_out)}")
print(f"Warmup:       {WARMUP}")
print(f"Repeat:       {REPEAT}")
print(f"")
print(f"Model load:        {model_load_ms:8.2f} ms")
print(f"E2E latency:       {avg_e2e:8.2f} ms")
print(f"TPOT:              {avg_tpot:8.2f} ms")
print(f"E2E throughput:    {avg_out / avg_e2e * 1000:8.2f} tok/s")
print(f"Peak GPU memory:   {mem_peak:8.0f} MB")
print(f"{'='*60}")

# ── JSON ─────────────────────────────────────────────────
ts = time.strftime("%Y%m%d_%H%M%S")
out_dir = os.path.join(OUTPUT_DIR, ts)
os.makedirs(out_dir, exist_ok=True)
latest_link = os.path.join(OUTPUT_DIR, "latest")
if os.path.islink(latest_link) or os.path.exists(latest_link):
    os.remove(latest_link)
os.symlink(ts, latest_link)

result = {
    "environment": {
        "gpu_name":    torch.cuda.get_device_name(0),
        "cuda_version": torch.version.cuda,
        "model":       MODEL_PATH,
        "precision":   "float32",
        "framework":   "transformers",
    },
    "config": {
        "prompt_tokens":  int(prompt_tokens),
        "max_new_tokens": MAX_NEW,
        "warmup":         WARMUP,
        "repeat":         REPEAT,
        "greedy":         True,
    },
    "summary": {
        "model_load_ms":              round(model_load_ms, 4),
        "e2e_latency_ms":             round(avg_e2e, 4),
        "tpot_ms":                    round(avg_tpot, 4),
        "e2e_throughput_tokens_per_s":round(avg_out / avg_e2e * 1000, 2),
        "output_tokens":              int(avg_out),
        "gpu_memory_before_model_mb": round(mem_before, 1),
        "gpu_memory_after_model_mb":  round(mem_after, 1),
        "gpu_memory_peak_mb":         round(mem_peak, 1),
    },
    "runs": runs,
}

path = os.path.join(out_dir, "results.json")
with open(path, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nResults written to: {path}")
