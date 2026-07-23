# Inference Profiling Guide

## Overview

This directory contains tools for systematic benchmarking and profiling of the inference framework. The workflow follows a four-layer analysis approach:

```
End-to-end Metrics → Stage Breakdown → Nsight Systems → Nsight Compute
```

- **Layer 1 — End-to-end Metrics**: Baseline throughput, latency, TPOT, ITL percentiles
- **Layer 2 — Stage Breakdown**: Per-stage (tokenizer/prefill/decode/sampling) and per-layer timing
- **Layer 3 — Nsight Systems**: System-level timeline — kernel concurrency, memory transfers, GPU utilization
- **Layer 4 — Nsight Compute**: Individual kernel analysis — occupancy, memory bandwidth, instruction throughput

## Directory Structure

```
profile/
├── README.md               ← This file
├── configs/                 ← Benchmark configuration presets
├── results/
│   ├── baseline/            ← Baseline (unoptimized) benchmark results
│   └── optimized/           ← Future optimized results for comparison
├── nsys/                    ← Nsight Systems .nsys-rep reports
├── ncu/                     ← Nsight Compute .ncu-rep reports
└── scripts/
    ├── run_baseline.sh      ← Run benchmark, save JSON + console log
    ├── run_nsys.sh          ← Run Nsight Systems profile
    ├── run_ncu.sh           ← Profile a specific kernel with Nsight Compute
    └── export_profile.sh    ← Package results for Windows transfer
```

## Quick Start

### 1. Run Baseline Benchmark

```bash
cd profile/scripts
chmod +x *.sh

# Default settings
./run_baseline.sh

# Custom settings via environment variables
MODEL_PATH=/path/to/model.bin \
VOCAB_PATH=/path/to/tokenizer.json \
PROMPT="Explain quantum computing" \
MAX_NEW_TOKENS=256 \
WARMUP=3 \
REPEAT=10 \
./run_baseline.sh
```

Or using the demo directly:

```bash
./bin/demo \
  --benchmark \
  --max-new-tokens 128 \
  --warmup 3 \
  --repeat 10 \
  --no-stream-output \
  --output profile/results/baseline/results.json
```

### 2. Profile with Nsight Systems

```bash
# Profile with NVTX markers (requires ENABLE_NVTX=ON build)
MAX_NEW_TOKENS=32 ./profile/scripts/run_nsys.sh
```

**Note**: For NVTX markers to appear in the timeline, build with:
```bash
cmake .. -DENABLE_NVTX=ON && make -j
```

### 3. Profile a Specific Kernel with Nsight Compute

```bash
# First, identify kernel names from nsys timeline or source code.
# Common kernels in this project:
#   matmul_kernel_cuda_fp32
#   mha_kernel_cuda_fp32
#   rmsnorm_kernel_cuda
#   rope_kernel_cuda
#   swiglu_kernel_cuda

# Profile matmul kernel (single launch)
KERNEL_REGEX="matmul_kernel_cuda_fp32" ./profile/scripts/run_ncu.sh

# Profile MHA kernel (first 3 launches)
KERNEL_REGEX="mha_kernel_cuda_fp32" LAUNCH_COUNT=3 ./profile/scripts/run_ncu.sh
```

### 4. Export Results for Windows Analysis

```bash
./profile/scripts/export_profile.sh
# Creates profile_results_YYYYMMDD_HHMMSS.tar.gz

# Copy to Windows and extract, then open .nsys-rep/.ncu-rep in Nsight GUIs
```

## Test Conditions

For fair comparison between baseline and optimized runs, fix these parameters:

| Parameter | Recommendation | Flag |
|-----------|---------------|------|
| GPU | Same device | N/A |
| Model | Same .bin file | `--model` |
| Precision | fp32 (only supported) | N/A |
| Batch size | 1 (only supported) | N/A |
| Prompt | Same text | `--prompt` or `--prompt-file` |
| Max new tokens | 128 | `--max-new-tokens` |
| Warmup | 3 | `--warmup` |
| Repeat | 10 | `--repeat` |
| Decoding | Greedy | `--greedy` |
| Seed | 42 | `--seed` |

## Key Metrics Explained

### Prefill
Time from the first prompt token entering the model to the first output token being ready.
- `prefill_time_ms`: GPU time for the prefill phase
- `ttft_model_only_ms`: Time to first token (model computation only) = prefill_time_ms
- `ttft_end_to_end_ms`: End-to-end TTFT = tokenizer_encode_ms + prefill_time_ms

### Decode
Time for all tokens after the first one.
- `decode_time_ms`: Total GPU time for decode phase
- `tpot_ms` (Time Per Output Token): `decode_time / (output_tokens - 1)`
  - If only 1 token is generated, TPOT is reported as the decode_time
- `decode_throughput_tps`: `(output_tokens - 1) / decode_time * 1000`

### Inter-Token Latency (ITL)
Per-step latency for each decode token. Reported as:
- mean, min, max
- p50, p90, p95, p99

### Throughput
- `prefill_throughput_tps`: `prompt_tokens / prefill_time * 1000`
- `decode_throughput_tps`: `decode_tokens / decode_time * 1000`
- `e2e_throughput_tps`: `output_tokens / e2e_latency * 1000`

## GPU Memory Measurement

GPU memory is measured using `cudaMemGetInfo()`, which reports free/total device memory as observed by the process. This reflects overall device memory allocation but does NOT provide per-buffer tracking from the internal allocator.

**Limitations**:
- `cudaMemGetInfo` returns OS-level device memory view, not internal allocator state
- Peak memory is estimated as max(before_model, after_model), not tracked dynamically
- The internal Memory Allocator does not expose allocation statistics

## Timing Methodology

| Measurement | Method | Synchronization |
|-------------|--------|-----------------|
| End-to-end latency | `std::chrono` + `cudaEvent` | `cudaEventSynchronize` at e2e boundary |
| Prefill/Decode GPU time | `cudaEvent` pairs | `cudaEventSynchronize` at stage boundaries |
| Per-layer module timing | `cudaEvent` pairs | `cudaEventSynchronize` at each module boundary |
| Tokenizer encode | `std::chrono` | N/A (CPU-only) |
| Model load | `std::chrono` | N/A (CPU + GPU transfer) |

### Important: On Synchronization Overhead

**Normal benchmark mode** (`--benchmark`):
- Only synchronizes at prefill/decode boundaries → minimal overhead
- Timing reflects real end-to-end performance

**Layer profile mode** (`--layer-profile`):
- Synchronizes at every layer module boundary → significant overhead
- Per-module timings include synchronization cost
- Do NOT compare layer-profile timing directly with normal benchmark timing
- Use layer-profile only for understanding WHERE time is spent, not absolute timing

## NVTX Markers

When built with `-DENABLE_NVTX=ON`, the following NVTX ranges are added to the Nsight Systems timeline:

| NVTX Range | Location |
|------------|----------|
| `transformer_layer_N` | Each transformer layer iteration |
| `input_rmsnorm_LN` | Attention input RMSNorm |
| `qkv_projection_LN` | Q/K/V projection + RoPE + KV cache write |
| `attention_LN` | Multi-head attention + output projection |
| `mlp_LN` | Feed-forward network (gate/up/swiglu/down + residuals) |
| `final_rmsnorm+lm_head` | Final RMSNorm + LM head projection |
| `embedding` | Token embedding lookup |

When NVTX is disabled (default), all NVTX macros expand to empty — zero overhead.

## Nsight Systems Viewing

1. Copy `.nsys-rep` file from `profile/nsys/` to a Windows machine
2. Open **NVIDIA Nsight Systems** (download from [NVIDIA Developer](https://developer.nvidia.com/nsight-systems))
3. File → Open → select the `.nsys-rep` file
4. Explore:
   - **Timeline View**: GPU kernel execution, memory copies, stream activity
   - **NVTX ranges**: Overlaid on the timeline for semantic context
   - **CUDA Memory**: Memory allocation/deallocation events

## Nsight Compute Viewing

1. Copy `.ncu-rep` file from `profile/ncu/` to a Windows machine
2. Open **NVIDIA Nsight Compute** (download from [NVIDIA Developer](https://developer.nvidia.com/nsight-compute))
3. File → Open → select the `.ncu-rep` file
4. Explore:
   - **Summary Page**: High-level kernel metrics (occupancy, bandwidth, compute utilization)
   - **Details Page**: Source code analysis, assembly (SASS), memory patterns
   - **Raw Page**: All collected metrics

## Profiling Overhead Warning

> **Do NOT compare Nsight profiling results directly with baseline benchmark results.**

- **Nsight Systems**: Adds moderate overhead (~10-30%). Useful for relative comparison between runs.
- **Nsight Compute**: Serializes kernel execution. Timing is completely invalid. Only use for hardware metric analysis (occupancy, bandwidth, instruction mix).
- **Layer profile mode**: Sync overhead makes absolute timing invalid. Only use for distribution analysis ("40% of decode time in attention").

## Adding New Kernels to Profile

To add a new model or kernel variant:

1. Ensure the kernel name is descriptive and consistent in the `.cu` file
2. Add NVTX ranges around the key code paths
3. Update the kernel name list in `run_ncu.sh`
4. Run baseline before and after to verify correctness is preserved

## Requirements

- CUDA Toolkit 12.1+ (provides nsys, ncu, nvtx3)
- Bash shell (for scripts)
- Compiled `bin/demo` binary with profile support

## Limitations

1. **Batch size = 1 only**: Current framework does not support batched inference
2. **No FP8/INT8 quantization profiling**: Only fp32 inference
3. **No PagedAttention or CUDA Graph**: These would change the profiling characteristics
4. **Layer profile granularity**: Currently groups QKV projection + RoPE + KV cache write into one timing boundary. See `.cpp` source for exact boundaries.
5. **Sampling time not separately measured**: Sampling overhead is included in decode step timing
6. **No internal allocator stats**: GPU memory is tracked via `cudaMemGetInfo` only
7. **CPU overhead not isolated**: Tokenizer time includes Python tokenizer JSON parsing overhead
8. **ITL measured with cudaEvent in benchmark mode**: When not using cudaEvent (CPU timer only), ITL may include CPU thread scheduling jitter
