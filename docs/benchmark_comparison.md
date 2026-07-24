# Benchmark Comparison: 3 Configurations

## Configurations

| # | Branch | Matmul | MHA | Precision | Model |
|---|--------|--------|-----|-----------|-------|
| 1 | `bench/baseline-fp32` | Basic (float4 + BlockReduce) | Simple (1-thread/head) | FP32 | Qwen2.5-0.5B |
| 2 | `bench/optimized-fp32` | Tiled (双缓冲 + float4) | Optimized (256-thread/head) | FP32 | Qwen2.5-0.5B |
| 3 | `bench/optimized-int8` | Tiled (双缓冲 + float4) | Optimized (256-thread/head) | INT8 W8A32 | Qwen2.5-7B |

## Benchmark Results (Qwen2.5-0.5B: Config 1 vs 2)

All tests: RTX 4090, prompt=50 tokens, output=128 tokens, warmup=3, repeat=10, greedy.

| Metric | Baseline FP32 | Optimized FP32 | Speedup |
|--------|--------------|----------------|---------|
| **TPOT** | 5.43 ms | **1.80 ms** | **3.0×** |
| **Decode Throughput** | 184 tok/s | **555 tok/s** | **3.0×** |
| **E2E Latency** | 907 ms | **364 ms** | **2.5×** |
| **Prefill** | 217 ms | **135 ms** | **1.6×** |
| Prefill Throughput | 230 tok/s | 371 tok/s | 1.6× |
| GPU Memory | 4183 MB | 4183 MB | — |
| ITL P50 | 8.70 ms | 2.90 ms | 3.0× |
| ITL P95 | 11.02 ms | 2.93 ms | 3.8× |

### Key Takeaway
Tiled matmul + optimized MHA delivers **3× speedup** over the basic implementation on the 0.5B model.

## Benchmark Results (Qwen2.5-7B: Config 3)

| Metric | Optimized INT8 (7B) |
|--------|---------------------|
| **TPOT** | 5.93 ms |
| **Decode Throughput** | 168 tok/s |
| E2E Latency | 1207 ms |
| Prefill | 453 ms |
| GPU Memory | 10,931 MB |
| ITL P50 | 9.54 ms |

### Key Takeaway
INT8 quantization enables running a 7B model comfortably within 24GB VRAM (10.9 GB used). The 7B INT8 model achieves similar TPOT (5.93 ms) to the 0.5B FP32 baseline (5.43 ms) — delivering 14× more parameters at the same inference speed.

## Cross-Configuration Comparison

| | Baseline 0.5B FP32 | Optimized 0.5B FP32 | Optimized 7B INT8 |
|---|---|---|---|
| Model Size | 0.5B | 0.5B | **7B** |
| Weight Precision | FP32 | FP32 | INT8 (W8A32) |
| Weight Size | ~2 GB | ~2 GB | ~6.2 GB + scales |
| TPOT | 5.43 ms | **1.80 ms** | 5.93 ms |
| Throughput | 184 tok/s | **555 tok/s** | 168 tok/s |
| VRAM | 4.2 GB | 4.2 GB | 10.9 GB |
| Tokens per GB VRAM | 43.8 | **132** | 15.4 |

## Nsight Systems Reports

| Configuration | nsys Report |
|--------------|-------------|
| Baseline FP32 | `profile/nsys/baseline_fp32.nsys-rep` |
| Optimized FP32 | `profile/nsys/optimized_fp32.nsys-rep` |
| Optimized INT8 | `profile/nsys/optimized_int8.nsys-rep` |

## Branches

| Configuration | Branch |
|--------------|--------|
| Baseline FP32 | [`bench/baseline-fp32`](https://github.com/liangji-seu/project_1_my_cuda_vllm/tree/bench/baseline-fp32) |
| Optimized FP32 | [`bench/optimized-fp32`](https://github.com/liangji-seu/project_1_my_cuda_vllm/tree/bench/optimized-fp32) |
| Optimized INT8 | [`bench/optimized-int8`](https://github.com/liangji-seu/project_1_my_cuda_vllm/tree/bench/optimized-int8) |

## Reproducing

```bash
# Checkout any branch
git checkout bench/baseline-fp32   # or bench/optimized-fp32 / bench/optimized-int8

# Build with NVTX for nsys profiling
cmake -S . -B build -DENABLE_NVTX=ON
cmake --build build -j$(nproc)

# Benchmark
./bin/demo --benchmark --max-new-tokens 128 --warmup 3 --repeat 10 \
  --greedy --no-early-stop --no-stream-output --output results.json

# Nsight Systems profile
nsys profile --trace=cuda,nvtx,osrt --cuda-memory-usage=true \
  --force-overwrite=true --output=profile/nsys/<name> \
  ./bin/demo --benchmark --max-new-tokens 128 --warmup 1 --repeat 1 \
  --greedy --no-early-stop --no-stream-output

# Transformers comparison
python3 tools/qwen_test.py --model_path /path/to/hf/model \
  --cpp_bin demo/model.bin [--cpp_quant]
```
