# my_cuda_vllm 架构文档

## 1. 项目概览

my_cuda_vllm 是一个从零构建的大语言模型推理框架，采用 C++20 和 CUDA 实现，遵循 RAII 资源管理思想。当前支持 Qwen2.5-0.5B-Instruct 模型在单卡 RTX 4090 上的完整自回归推理。

### 1.1 核心特性

| 特性 | 描述 |
|------|------|
| **模型支持** | Qwen2.5-0.5B-Instruct（主要），Llama3（可选），支持 GQA |
| **精度** | FP32 计算 |
| **分词器** | BPE（tiktoken 引擎），Qwen3 变体 |
| **采样策略** | 贪心采样（Argmax）、Top-K / Top-P 采样 |
| **推理模式** | Prefill / Decode 分离 |
| **KV Cache** | 预分配连续显存，支持 GQA |
| **构建系统** | CMake 3.16+ |

### 1.2 Benchmark 基线

Qwen2.5-0.5B-Instruct, RTX 4090, FP32, prompt=50 tokens, output=128 tokens, greedy:

```
Model load:          1665.63 ms
TTFT (model-only):    335.38 ms
Prefill throughput:   149.09 tok/s
Decode:              1370.86 ms
TPOT:                  10.79 ms
Decode throughput:     92.64 tok/s
E2E latency:         1706.63 ms
E2E throughput:        75.00 tok/s
Peak GPU mem:          4183 MB
```

---

## 2. 项目结构

```
my_cuda_vllm/
├── CMakeLists.txt                    # 顶层构建文件
├── sys/                              # 框架核心源码
│   ├── include/                      # 头文件
│   │   ├── base/                     # 基础设施层
│   │   │   ├── base.h                # 设备类型、模型类型、错误码、Status类
│   │   │   ├── DeviceController.h    # 内存控制器（CPU/GPU 工厂）
│   │   │   ├── Buffer.h              # RAII 内存缓冲区
│   │   │   ├── cuda_stream.h         # CUDA Stream RAII 封装
│   │   │   ├── unicode.h/data.h      # Unicode 转换
│   │   │   └── tiktoken.h            # BPE 分词引擎
│   │   ├── tensor/
│   │   │   └── tensor.h              # 多维张量（shape + Buffer）
│   │   ├── op/                       # 算子层头文件
│   │   │   ├── layer.h               # BaseLayer → Layer → LayerParam 类层次
│   │   │   ├── add.h, embedding.h, rmsnorm.h, matmul.h
│   │   │   ├── mha.h, rope.h, softmax.h, swiglu.h, encode.h
│   │   ├── model/                    # 模型层头文件
│   │   │   ├── buffer_type.h         # ModelBufferType 枚举（预分配张量索引）
│   │   │   ├── config.h              # ModelConfig / TransformerConfig
│   │   │   ├── raw_model_data.h      # mmap 权重数据
│   │   │   ├── model.h               # Model 抽象基类
│   │   │   └── llama3.h              # LLama2Model 具体实现
│   │   ├── sampler/                  # 采样器头文件
│   │   │   ├── sampler.h             # Sampler 抽象基类
│   │   │   ├── argmax_sampler.h      # 贪心采样
│   │   │   ├── topk_sampler.h        # Top-K/Top-P 采样
│   │   │   └── gpu_argmax.h          # GPU 端 argmax 声明
│   │   └── profile/                  # 性能评估
│   │       ├── profiler.h            # Profiler / CudaTimer / BenchmarkResult
│   │       └── nvtx_utils.h          # NVTX 标记工具
│   └── src/                          # 实现文件
│       ├── base/                     # Buffer, DeviceController, Status, unicode
│       ├── tensor/
│       │   └── tensor.cpp            # Tensor 实现（reshape, to(device), slice...）
│       ├── op/                       # 算子前端（forward, check_layer）
│       │   ├── kernel/               # 后端实现
│       │   │   ├── kernel_interface.h / .cpp   # 双后端接口分发
│       │   │   ├── cpu/              # CPU 后端（8个算子）
│       │   │   └── gpu/              # GPU 后端（8个算子，含2个优化版）
│       │   ├── layer.cpp             # Layer / LayerParam 基类实现
│       │   └── *.cpp                 # 各算子前端
│       ├── model/
│       │   ├── model.cpp             # Model 基类（权重加载、buffer管理、KV cache切片）
│       │   ├── llama3.cpp            # LLama2Model 完整推理流水线
│       │   └── raw_model_data.cpp    # mmap 权重读写
│       ├── sampler/
│       │   ├── argmax_sampler.cpp    # CPU argmax
│       │   ├── topk_sampler.cpp      # CPU Top-K/Top-P
│       │   └── gpu_argmax.cu         # GPU argmax kernel（优化项三）
│       └── profile/
│           └── profiler.cpp          # Profiler 实现 + BenchmarkResult JSON 序列化
├── demo/
│   ├── main.cpp                      # 推理入口（CLI + benchmark 模式）
│   ├── chat.cpp                      # 交互式对话入口
│   └── CMakeLists.txt
├── test/                             # GTest 单元测试
├── tools/                            # 模型导出/解析工具
├── profile/                          # 性能分析脚本和 nsys 输出
├── docs/                             # 文档
└── bin/                              # 编译产物（demo, chat）
```

---

## 3. 架构分层

整个框架采用经典的**五层架构**，自底向上依次是：

```
┌──────────────────────────────────────────────┐
│  应用层    │ demo/main.cpp, demo/chat.cpp      │
├──────────────────────────────────────────────┤
│  模型层    │ Model → LLama2Model               │
│            │ 推理流水线, KV Cache, 采样         │
├──────────────────────────────────────────────┤
│  算子层    │ BaseLayer → Layer → LayerParam    │
│            │ 10 种算子, 双后端分发              │
├──────────────────────────────────────────────┤
│  张量层    │ Tensor (shape + Buffer)            │
│            │ reshape, to(device), slice         │
├──────────────────────────────────────────────┤
│  基础层    │ DeviceController, Buffer, Stream   │
│            │ RAII 内存, CPU/GPU 工厂, 错误码    │
└──────────────────────────────────────────────┘
```

### 3.1 基础层（`base/`）

**DeviceController** — 抽象内存分配器，子类 `CPUDeviceController` 和 `GPUDeviceController` 分别封装 `malloc/free` 和 `cudaMalloc/cudaFree`。通过工厂方法获取全局单例。

**Buffer** — RAII 内存缓冲区，持有 `void*` 指针和 `DeviceController` 的 shared_ptr。析构时自动调用控制器的 `mem_release`。支持外部指针引用（`flag_is_external = true`）避免重复分配。

**CudaStream** — RAII CUDA 流封装，构造时 `cudaStreamCreate`，析构时 `cudaStreamDestroy`。

**Status** — 错误码 + 错误消息，通过 `STATUS_CHECK` 宏在失败时执行 `LOG(FATAL)`。

### 3.2 张量层（`tensor/`）

**Tensor** — 多维数组抽象。核心成员：

```
dims:         vector<size_t>    形状
size:         size_t            元素总数
data_type:    DataType_t        fp32 / int32 / int8
buffer:       shared_ptr<Buffer> 底层 RAII 内存
```

关键操作：
- `to("cuda" / "cpu", stream)` — 设备间转移，内部通过 DeviceController 的 `mem_copy` 实现
- `reshape(dims)` — 改变形状（总元素数不变）
- `slice(axis, start, end)` — 零拷贝切片视图
- `peek_index<T>(offset)` / `peek_position<T>(pos)` — 模板化的元素访问

### 3.3 算子层（`op/`）

类继承关系：

```
BaseLayer (抽象)                     ← 名字、类型、精度、设备
  └── Layer (无权重的层)              ← inputs[], outputs[], cuda_stream
        └── LayerParam (有权重的层)    ← weights[], scales, group_size
```

**算子列表**：

| 算子 | 继承 | 输入数 | 权重 | GPU 后端 | 优化 |
|------|------|--------|------|----------|------|
| Embedding | LayerParam | 2 | vocab[vs,dim] | float4 向量化读取 | — |
| RMSNorm | LayerParam | 1 | gamma[dim] | grid-stride + tree reduction | — |
| Matmul | LayerParam | 1 | W[N,K] + bias[N] | 双版本（基础 + 优化） | ✅ 优化版 |
| RoPE | Layer | 5(Q,K,pos,sin,cos) | — | 每线程处理一对旋转 | — |
| MHA | Layer | 5(Q,score,K,V,out) | — | block-per-head + shared memory | ✅ 优化版 |
| Softmax | Layer | 1 | — | 原地，cub::BlockReduce | — |
| SwiGLU | Layer | 2(gate,up) | — | 逐元素 | — |
| Add | Layer | 2(x1,x2) | — | 逐元素 float4 | — |
| Encode | Layer | — | — | CPU 专用 (tiktoken) | — |

**Kernel 接口分发**（`kernel_interface.h/.cpp`）：

每个算子通过 `get_xxx_interface(device_type)` 函数返回对应后端的函数指针。例如：

```cpp
Matmul_backend get_matmul_interface(DeviceType_t type) {
    if (type == CPU) return matmul_kernel_cpu;
    if (type == GPU) return matmul_kernel_cuda;  // 基础版
}
```

### 3.4 模型层（`model/`）

**Model** 抽象基类持有的核心资源：

```
config_           : TransformerConfig   模型超参
raw_model_data_   : RawModelData        mmap 权重文件
encode_layer_     : EncodeLayerBase     分词器
buffers_          : map<BufferType,Tensor> 预分配中间张量
sampler_          : Sampler             采样器
profiler_         : Profiler*           可选的性能分析器（不属于模型）
```

**LLama2Model** — 继承 Model，额外持有：

```
llama_layers_  : LLama2Layers   所有算子实例的集合
cuda_stream_   : CudaStream     CUDA 流
```

**LLama2Layers** 结构：

```
非参数算子（单例）:
  add_layer_, rope_layer_, mha_layer_, swiglu_layer_

参数算子（每层一个）:
  wq_layers_[L], wk_layers_[L], wv_layers_[L], wo_layers_[L]   (QKV 投影)
  w1_layers_[L], w2_layers_[L], w3_layers_[L]                   (FFN gate/up/down)
  rmsnorm_layers_[2L+1]                                          (2L层内 + 1层末)

特殊:
  embedding_layer_, cls_layer_, q_norm_weights_, k_norm_weights_
```

**ModelBufferType** — 预分配的中间张量（避免运行时反复 alloc/free）：

| Buffer 类型 | 形状 | 用途 |
|------------|------|------|
| kInputTokens | [token_num] | 输入 token ID |
| kInputEmbeddings | [token_num, dim] | 词嵌入输出 |
| kOutputRMSNorm | [dim] | RMSNorm 输出（复用） |
| kKeyCache | [L, seq_len, kv_dim] | K 缓存 |
| kValueCache | [L, seq_len, kv_dim] | V 缓存 |
| kQuery | [q_dim] | Q 投影输出 |
| kScoreStorage | [head_num, seq_len] | 注意力分数 |
| kOutputMHA | [q_dim] | MHA 输出 |
| kAttnOutput | [dim] | Wo 投影输出 |
| kW1Output | [hidden_dim] | Gate 投影输出 |
| kW2Output | [dim] | Down 投影输出 |
| kW3Output | [hidden_dim] | Up 投影输出 |
| kFFNRMSNorm | [dim] | FFN RMSNorm 输出（复用） |
| kForwardOutput | [vocab_size] | 最终 logits |
| kSinCache | [head_size × seq_len] | RoPE sin |
| kCosCache | [head_size × seq_len] | RoPE cos |

### 3.5 权重文件格式

`.bin` 文件布局（由 `tools/export_from_structure.py` 导出）：

```
Header:  ModelConfig (7×int32)  +  flags (int32)  [+  group_size for int8]
Weights: embedding, attn_rmsnorm×L, Wq×L, Wk×L, Wv×L, Wo×L,
         ffn_rmsnorm×L, W1×L, W2×L, W3×L, final_rmsnorm,
         freqs_cos, freqs_sin, [CLS weight if not tied]
```

权重通过 `mmap` 映射到 CPU 虚拟地址空间，初始化时复制到 GPU 显存。

---

## 4. 推理流水线

### 4.1 端到端流程

```
Tokenizer.encode(prompt)
        │
        ▼
Embedding(tokens) ──────── 查表获取词向量
        │
        ▼
for pos in 0..max_steps:
  ┌──────────────────────────────────────┐
  │  if pos < prompt_len-1:  PREFILL     │
  │  else:                  DECODE       │
  │                                      │
  │  for layer 0..L-1:                   │
  │    ┌──────────────────────────────┐  │
  │    │ ① attention_rms             │  │  RMSNorm(input)
  │    │ ② attention_qkv             │  │  Wq/Wk/Wv(rms_out) + RoPE(Q,K) + write KV cache
  │    │ ③ attention_mha             │  │  MHA(Q,K_cache,V_cache) + Wo
  │    │ ④ feed_forward              │  │  input+=attn_out→RMSNorm→gate/up→SwiGLU→down→residual
  │    └──────────────────────────────┘  │
  │                                      │
  │  cls_logits: RMSNorm(input) → lm_head│
  │  post_processing: argmax(logits)     │
  └──────────────────────────────────────┘
```

### 4.2 Prefill 阶段

- 一次性处理所有 prompt token
- embedding 输出 shape 为 `[prompt_len, dim]`，每个位置取一行作为该层输入
- 所有层的 KV Cache 被依次写入
- **不做采样** — 用 prompt 的 ground truth token 作为 next

### 4.3 Decode 阶段

- 每步只处理 1 个 token
- KV Cache 追加写入新位置
- **零拷贝闭环**（关键优化）：`post_processing` 中 GPU argmax 直接将 token_id 写入 GPU `kInputTokens` buffer，下一轮 `embed_next_token` 跳过 CPU 中转
- 通过 `cudaEventSynchronize` 保证 GPU argmax 完成后再读取 CPU token（仅用于解码显示）

### 4.4 Transformer Block 详细流程

```
                        input [dim]
                           │
              ┌────────────┼────────────┐
              │            │            │
         RMSNorm      RMSNorm      (skip)
              │            │            │
            Wq            Wk           Wv
              │            │            │
           RoPE          RoPE          │
              │            │            │
              └────────────┼────────────┘
                           │
                    KV Cache Write
                      (pos 位置)
                           │
               ┌───────────┼───────────┐
               │           │           │
             Q_new      K_cache     V_cache
               │           │           │
               └───────────┼───────────┘
                           │
                    Scaled Dot-Product
                     Attention + GQA
                           │
                          Wo
                           │
                      attn_out [dim]
                           │
                    ┌──────┴──────┐
                    │    Add (+)   │ ← residual connection
                    └──────┬──────┘
                           │
                        RMSNorm
                           │
                    ┌──────┼──────┐
                    │      │      │
                   W1     W3     (skip)
                 (gate)  (up)
                    │      │
                    └──┬───┘
                       │
                    SwiGLU
                       │
                      W2
                    (down)
                       │
                    ┌──┴─────┐
                    │ Add (+) │ ← residual connection
                    └──┬─────┘
                       │
                    output [dim]
```

---

## 5. GPU 算子优化详解

### 5.1 优化版 Matmul（`matmul_kernel_optimized.cu`）

这是项目的**第一个重点优化**，实现了基于 Tiling + 双缓冲共享内存 + 寄存器预取的矩阵乘法。

**基础版 vs 优化版对比**：

| 维度 | 基础版 | 优化版 |
|------|--------|--------|
| Block 策略 | 每个输出元素一个 block | BM×BN=128×128 的 tile |
| 共享内存 | 无（只用 shared memory 做 BlockReduce） | 双缓冲 As[2][BK×BM] + Bs[2][BK×BN] |
| K 维度加载 | 每个线程多次从 global memory 读取 | float4 向量化，一次读 4 个 float |
| 内存隐式 | 无 | 全局→寄存器→共享内存 流水线 |
| FMA 计算 | 标量 | 寄存器 TM×TN=8×8 累加器 |
| SM 利用率 | M=1 时只有 N 个 block | M≥128 时 grid 为 ceil(N/128)×ceil(M/128) |
| 适用场景 | M < 128（decode 阶段） | M ≥ 128（prefill 阶段） |

**核心超参数**：

```cpp
BM=128, BN=128, BK=8     // 输出 tile 和 K 切片大小
TM=8, TN=8                // 每个线程负责 8×8 输出子块
NUM_THREADS = 256         // (128/8) × (128/8)
```

**双缓冲流水线**（do-while 结构）：

```
Tile 0: 全局→共享内存 → __syncthreads → 预取 l=0
    ↓
主循环 do { k += BK
  ┌─ 全局→寄存器：预取下一个 tile 的 A、W (可与计算并行)  ← 指令级重叠
  ├─ 计算：l = 0..BK-2，预取 l+1，计算 l            ← 寄存器 FMA
  ├─ 计算：l = BK-1（无需预取）
  └─ 寄存器→共享内存 + 预取新 tile l=0
} while (k < K)

写回：累加器 → 全局内存 C（scale + bias, float4 vectorized store）
```

**策略自动选择**：

```cpp
if (M < BM) {
    matmul_kernel_cuda(...);            // 回退基础版 (M=1, decode)
} else {
    matmul_kernel_tiled<BM,BN,BK,TM,TN>(...);  // 优化版 (M≥128, prefill)
}
```

### 5.2 优化版 MHA（`mha_kernel.cu`）

这是项目的**第二个重点优化**，将原本单线程处理一个 head 的朴素 MHA 改为 256 线程协作处理。

**优化点**：

| 维度 | 朴素版 (CPU) | 优化版 (GPU) |
|------|-------------|-------------|
| 并行粒度 | 1 thread/head | 256 threads/head (block-per-head) |
| Query 读取 | 每个 K 位置从 global memory 重复读取 | 预加载到 shared memory（只读一次） |
| QK 点积 | 标量逐元素 | float4 向量化 + grid-stride 分布 |
| Softmax | 串行 C++ | cub::BlockReduce（warp shuffle） |
| V 加权求和 | 标量逐元素 | grid-stride 并行 |
| GQA 支持 | ✅ | ✅ (kv_mul 参数) |

**Kernel 实现**：

```cpp
__global__ void mha_kernel_cuda_fp32(params) {
    int head = blockIdx.x;     // 每个 head 一个 block (grid = head_num)

    // ① Query 预加载到 shared memory (256 threads 协作)
    for (i = tid; i < head_size; i += 256)
        s_query[i] = query_head[i];
    __syncthreads();

    // ② QK 点积 -> score[t] (float4 向量化)
    for (t = tid; t <= pos; t += 256) {
        float s = 0;
        for (i = 0; i < head_size; i += 4)
            s += k4.x * q4.x + k4.y * q4.y + k4.z * q4.z + k4.w * q4.w;
        score[t] = s * rsqrt(head_size);
    }

    // ③ Softmax (cub::BlockReduce)
    softmax_block(score, pos + 1);

    // ④ V 加权求和 -> output[d]
    for (d = tid; d < head_size; d += 256) {
        float val = 0;
        for (t = 0; t <= pos; t++)
            val += score[t] * value_cache[t][d];
        output[d] = val;
    }
}
```

---

## 6. GPU 采样器（优化项三）

### 6.1 问题

原始流程中，LM Head 输出完整的 vocab_size 维 logits 后，需要将 logits 从 GPU 拷贝到 CPU，再由 CPU 执行 argmax。这引入了：

- 一次 `cudaMemcpy`（vocab_size × sizeof(float) ≈ 600KB for Qwen2.5-0.5B）
- CPU-GPU 同步开销
- 下一轮 embedding 需要再将 token_id 从 CPU 拷回 GPU

### 6.2 优化方案：GPU argmax + 零拷贝闭环

**`gpu_argmax.cu`** — GPU 端 argmax kernel：

```cpp
template <int BLOCK_SIZE=256>
__global__ void argmax_kernel(const float* input, size_t size,
                               int32_t* gpu_out) {
    // 每线程找局部最大值
    float max_val = -FLT_MAX;
    size_t max_idx = SIZE_MAX;
    for (i = tid; i < size; i += BLOCK_SIZE)
        if (input[i] > max_val) { max_val = input[i]; max_idx = i; }

    // cub::BlockReduce 全局规约
    auto result = BlockReduce(temp).Reduce({max_idx, max_val}, cub::ArgMax());

    if (tid == 0) *gpu_out = result.key;  // 写入 GPU buffer
}
```

**零拷贝闭环**（`llama3.cpp` 中的 `post_processing`）：

```cpp
int32_t LLama2Model::post_processing(pos, is_prompt) {
    if (is_prompt) return -1;

    // GPU argmax → token_id 直接写入 GPU kInputTokens buffer
    int32_t* token_gpu = input_tokens.get_ptr();  // GPU 指针
    sampler::gpu_argmax_async(logits_gpu, vocab_size,
                              token_gpu,           // GPU 输出（下一轮 embedding 直接用）
                              &async_next_token_,  // CPU 异步接收（仅用于解码显示）
                              cuda_stream);

    return async_next_token_;  // 可能还未完成，调用方用 cudaEventSynchronize 等待
}
```

**数据流对比**：

```
优化前:
  GPU logits → cudaMemcpy D2H → CPU logits → CPU argmax → token_id → cudaMemcpy H2D → GPU input_tokens

优化后:
  GPU logits ──→ GPU argmax kernel ──→ GPU input_tokens  (全程不离开 GPU)
                 └── cudaMemcpyAsync D2H → CPU token (仅显示用, 不阻塞)
```

---

## 7. 采样器设计

```
Sampler (抽象基类)
  ├── ArgmaxSampler     贪心采样：直接取 logits 最大值索引
  └── TopKSampler       Top-K + Top-P 采样 (CPU only)
                         temperature → softmax → top-k 过滤 → top-p 过滤 → 随机采样
```

GPU 路径不走 `Sampler::sample()` 接口，而是直接调用 `gpu_argmax_async()`（见第 6 节）。

---

## 8. 性能分析体系

### 8.1 Profiler 模块

`profile::Profiler` 提供多层级的性能数据收集：

```
CudaTimer         ← cudaEvent 精确 GPU 计时
  └── Profiler    ← 管理多个 CudaTimer + CPU chrono 计时
        ├── RunRecord        单次推理 (prefill+decode)
        ├── LayerModuleRecord 每算子每层耗时 (--layer-profile)
        ├── StageRecord       阶段耗时
        └── BenchmarkResult   聚合统计 + JSON 输出
```

### 8.2 Benchmark 模式

```bash
./bin/demo --benchmark --warmup 3 --repeat 10 \
  --max-new-tokens 128 --greedy --no-early-stop \
  --output results.json
```

输出指标：
- **TTFT** (Time To First Token): model-only（prefill）+ end-to-end（+ tokenizer encode）
- **TPOT** (Time Per Output Token): decode 总耗时 / (输出 token 数 - 1)
- **Prefill/Decode throughput** (tok/s)
- **ITL** (Inter-Token Latency): P50/P90/P95/P99
- **GPU Memory**: before/after/peak

### 8.3 Layer Profile 模式

```bash
./bin/demo --layer-profile --max-new-tokens 128
```

通过 `cudaEvent` 测量每个 Transformer Block 的四个子模块耗时：
- `input_rmsnorm` — 输入的 RMSNorm
- `qkv_projection` — Wq/Wk/Wv 投影 + RoPE + KV Cache 写入
- `attention` — MHA + Wo 投影
- `mlp` — FFN RMSNorm + gate/up + SwiGLU + down + 残差连接

### 8.4 NVTX 标记

通过 `ENABLE_NVTX` CMake 选项开启。在 Nsight Systems 中可以看到分阶段分层的标记：

```
R1/prefill0/L0/rmsnorm      (蓝色)
R1/prefill0/L0/qkv           (蓝色)
R1/decode50/L3/attn          (绿色)
R1/decode50/L3/mlp           (绿色)
```

---

## 9. 构建与运行

### 9.1 依赖

| 库 | 用途 |
|----|------|
| CUDA 12.x | GPU 计算 |
| glog | 日志 |
| GTest | 单元测试 |
| Armadillo | 线性代数（CPU 端辅助） |
| abseil-cpp | 字符串处理（分词器） |
| re2 | 正则表达式（分词器） |
| nlohmann_json | JSON 序列化（tokenizer + benchmark 输出） |

### 9.2 编译

```bash
cd my_cuda_vllm
cmake -S . -B build -DENABLE_NVTX=ON
cmake --build build -j$(nproc)
```

产物：
- `lib/libmy_cuda_vllm.so` — 核心库
- `bin/demo` — 推理+benchmark 入口
- `bin/chat` — 交互式对话入口

### 9.3 运行

```bash
# 交互模式
./bin/demo

# Benchmark 模式
./bin/demo --benchmark --max-new-tokens 128 --output results.json

# Layer Profile 模式
./bin/demo --layer-profile --max-new-tokens 128

# nsys 性能分析
test.sh  # 包含 benchmark + nsys profile 两步

# 交互式对话
./bin/chat

# 与 PyTorch baseline 对比
python3 py_baseline.py
```

---

## 10. 关键设计决策

### 10.1 预分配 vs 动态分配

所有中间张量和 KV Cache 在 `init_mem()` 中一次性预分配。优势：
- 避免运行时 `cudaMalloc`/`cudaFree` 开销
- 避免内存碎片
- 便于显存用量预估

代价是显存占用较大（Qwen2.5-0.5B 约 4GB）。

### 10.2 算子前后端分离

每个算子有三层：
1. **头文件**（`include/op/xxx.h`）— 类声明、参数
2. **前端**（`src/op/xxx.cpp`）— `check_layer` + 调用 `kernel_interface`
3. **后端**（`src/op/kernel/{cpu,gpu}/`）— 实际计算

这使得新增一个后端（如 ROCm、Metal）只需添加对应 kernel 文件并注册到 `kernel_interface.cpp`。

### 10.3 权重加载策略

使用 `mmap` 将 `.bin` 文件映射到 CPU 虚拟地址空间（`MAP_PRIVATE`），然后通过 `cudaMemcpy` 将每层权重异步拷贝到 GPU。所有权重拷贝在模型构造阶段完成，推理阶段无磁盘 IO。

### 10.4 单 Stream 设计

整个推理过程使用一个 CUDA Stream。Kernel 按依赖顺序串行发射，避免多 stream 同步的复杂性。开销在 decode 阶段（每次只算 1 个 token）可接受。

### 10.5 Tokenizer 实现

使用 [tiktoken](https://github.com/liangji/tiktoken) C++ 库作为 BPE 引擎。特殊的控制 token（`<|im_start|>`, `<|im_end|>`）单独管理。在 CMake 中通过 `QWEN3_SUPPORT` 宏控制。

---

## 11. 已有文档索引

| 文档 | 内容 |
|------|------|
| [architecture.md](architecture.md) | 本文档：总体架构 |
| [operator_guide.md](operator_guide.md) | 算子层设计：类继承、后端接口、新增算子步骤 |
| [model_guide.md](model_guide.md) | 模型层设计：推理流水线、buffer 类型、采样器、权重格式 |
| [resume_project.md](resume_project.md) | 项目总结：技术栈、benchmark 数据、与 PyTorch HF 对比 |
