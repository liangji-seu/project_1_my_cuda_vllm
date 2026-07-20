# 模型层设计指南

## 架构概览

```
Model (抽象基类)
  ├── LLama2Model (Llama 2/3 实现)
  │     ├── LLama2Layers (持有所有算子实例)
  │     │     ├── 非参数算子 (Add, RoPE, MHA, SwiGLU)
  │     │     └── 参数算子 (Embedding, Matmul×8, RMSNorm)
  │     └── buffers_ (预分配的中间张量)
  └── (未来模型)
```

## 类继承关系

```
Model (抽象基类)
  └── LLama2Model (具体模型)
```

## 文件清单

| 文件 | 说明 |
|------|------|
| [sys/include/model/buffer_type.h](../sys/include/model/buffer_type.h) | ModelBufferType 枚举，定义所有中间缓冲区类型 |
| [sys/include/model/config.h](../sys/include/model/config.h) | ModelConfig (文件头) + TransformerConfig (派生参数) |
| [sys/include/model/raw_model_data.h](../sys/include/model/raw_model_data.h) | RawModelData 基类 + Fp32/Int8 子类，管理 mmap 权重 |
| [sys/include/model/model.h](../sys/include/model/model.h) | Model 抽象基类：encode/decode, fill_input, slice_kv_cache |
| [sys/include/model/llama3.h](../sys/include/model/llama3.h) | LLama2Model 声明 |
| [sys/src/model/model.cpp](../sys/src/model/model.cpp) | Model 基类实现：read_model_file, generate_model_infos 等 |
| [sys/src/model/llama3.cpp](../sys/src/model/llama3.cpp) | LLama2Model 实现：完整的 Llama 推理流水线 |
| [sys/src/model/raw_model_data.cpp](../sys/src/model/raw_model_data.cpp) | RawModelData 析构与 weight() 方法 |

## 模型权重文件格式

二进制文件布局：

```
┌─────────────────────────┐  offset 0
│  ModelConfig (8×int32)  │  28 bytes
├─────────────────────────┤
│  embedding weight       │  vocab_size × dim
│  attn RMSNorm weights   │  layer_num × dim
│  Wq weights             │  layer_num × dim × dim
│  Wk weights             │  layer_num × kv_dim × dim
│  Wv weights             │  layer_num × kv_dim × dim
│  Wo weights             │  layer_num × dim × dim
│  FFN RMSNorm weights    │  layer_num × dim
│  W1 (gate) weights      │  layer_num × hidden_dim × dim
│  W2 (down) weights      │  layer_num × dim × hidden_dim
│  W3 (up) weights        │  layer_num × hidden_dim × dim
│  final RMSNorm weight   │  dim
│  freqs_cos / freqs_sin  │  seq_len × head_size
│  CLS weight (可选)       │  vocab_size × dim
└─────────────────────────┘
```

## ModelBufferType 枚举

预分配的中间张量，避免反复分配内存：

| 缓冲区 | 形状 | 用途 |
|--------|------|------|
| kInputTokens | [token_num] | 输入 token IDs |
| kInputEmbeddings | [token_num, dim] | 输入词向量 |
| kOutputRMSNorm | [dim] | RMSNorm 输出 |
| kKeyCache | [layer_num, seq_len, kv_dim] | K 缓存 |
| kValueCache | [layer_num, seq_len, kv_dim] | V 缓存 |
| kQuery | [dim] | Q 向量 |
| kScoreStorage | [head_num, seq_len] | 注意力分数 |
| kOutputMHA | [dim] | MHA 输出 |
| kAttnOutput | [dim] | 注意力残差输出 |
| kW1Output | [hidden_dim] | W1 (gate) 输出 |
| kW2Output | [dim] | W2 (down) 输出 |
| kW3Output | [hidden_dim] | W3 (up) 输出 |
| kFFNRMSNorm | [dim] | FFN RMSNorm 输出 |
| kForwardOutput | [vocab_size] | 最终 logits |
| kSinCache | [head_size × seq_len] | RoPE sin 缓存 |
| kCosCache | [head_size × seq_len] | RoPE cos 缓存 |

## 推理流水线

```
embedding(tokens)
    │
    ▼
for each layer:
    ├── attention_rms ← RMSNorm(input)
    ├── attention_qkv ← Wq,Wk,Wv(rms_output) + RoPE(Q,K)
    ├── attention_mha ← MHA(Q,K,V) + Wo(mha_output)
    └── feed_forward ← add(residual) + RMSNorm + W1,W3 + SwiGLU + W2 + add(residual)
    │
    ▼
cls_logits ← RMSNorm(input) + Wcls(output) → logits
    │
    ▼
post_processing ← argmax(logits)
```

## 采样器

通过独立的 `sampler` 模块完成 token 采样，解耦模型推理与输出选择：

```
sampler::Sampler (抽象基类)
  └── sampler::ArgmaxSampler (贪婪采样)
```

| 文件 | 说明 |
|------|------|
| [sys/include/sampler/sampler.h](../sys/include/sampler/sampler.h) | Sampler 基类 |
| [sys/include/sampler/argmax_sampler.h](../sys/include/sampler/argmax_sampler.h) | ArgmaxSampler 声明 |
| [sys/src/sampler/argmax_sampler.cpp](../sys/src/sampler/argmax_sampler.cpp) | CPU: std::max_element, GPU: 预留 |

在 `Model` 基类中持有 `sampler_`，由 `post_processing()` 调用。未来可扩展 temperature/top-p/top-k 采样器。

## 量化支持

通过 `RawModelDataInt8` 子类支持 INT8 量化权重：

```cpp
if (!is_quant_model_) {
  raw_model_data_ = std::make_shared<RawModelDataFp32>();
} else {
  raw_model_data_ = std::make_shared<RawModelDataInt8>();
}
```

量化模型文件头在 `ModelConfig` 之后额外包含 `group_size` (int32)。当前量化推理尚未实现（`create_param_quant_layers()` 为 stub）。

## 新增模型的步骤

1. 在 `base/base.h` 的 `ModelType` 枚举中添加模型类型
2. 创建 `sys/include/model/xxx.h` 声明模型类
3. 创建 `sys/src/model/xxx.cpp` 实现：
   - `create_param_layers()` — 创建参数算子并加载权重
   - `create_nonparam_layers()` — 创建非参数算子
   - `init_mem()` — 预分配中间缓冲区
   - `forward()` — 实现推理流水线
   - `embedding()` — 实现词嵌入
   - `post_processing()` — 实现输出采样
4. 确保 `aux_source_directory(sys/src/model DIR_MODEL)` 在 CMakeLists.txt 中
