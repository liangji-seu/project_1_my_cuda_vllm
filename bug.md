# Qwen2.5 推理框架生成乱码 — 完整诊断报告

> 诊断日期：2026-07-21
> 目标：自制 CUDA/C++ 大模型推理框架输出与 HuggingFace Transformers reference 对齐
> 模型：Qwen2.5-0.5B-Instruct

---

## 步骤 1：真实模型结构确认

通过 `config.json` 和 `model.safetensors` 检查，完整 tensor 清单如下：

### config.json 关键参数

| 参数 | 值 |
|---|---|
| model_type | qwen2 |
| hidden_size | 896 |
| num_hidden_layers | 24 |
| num_attention_heads | 14 |
| num_key_value_heads | 2 |
| head_dim | 64 (896/14) |
| intermediate_size | 4864 |
| vocab_size | 151936 |
| rms_norm_eps | **1e-6** |
| rope_theta | 1000000.0 |
| tie_word_embeddings | true |
| hidden_act | silu |

### safetensors 完整结构（每层）

```
model.embed_tokens.weight                          [151936, 896]

model.layers.{0..23}.input_layernorm.weight        [896]
model.layers.{0..23}.self_attn.q_proj.weight       [896, 896]
model.layers.{0..23}.self_attn.q_proj.bias         [896]          ← 存在!
model.layers.{0..23}.self_attn.k_proj.weight       [128, 896]
model.layers.{0..23}.self_attn.k_proj.bias         [128]          ← 存在!
model.layers.{0..23}.self_attn.v_proj.weight       [128, 896]
model.layers.{0..23}.self_attn.v_proj.bias         [128]          ← 存在!
model.layers.{0..23}.self_attn.o_proj.weight       [896, 896]     ← 无 bias
model.layers.{0..23}.post_attention_layernorm.weight [896]
model.layers.{0..23}.mlp.gate_proj.weight          [4864, 896]    ← 无 bias
model.layers.{0..23}.mlp.down_proj.weight          [896, 4864]    ← 无 bias
model.layers.{0..23}.mlp.up_proj.weight            [4864, 896]    ← 无 bias

model.norm.weight                                  [896]
lm_head.weight                                     (tied with embed_tokens)
```

### 结构对照表

| 组件 | HF Qwen2.5 | C++ 期望/实现 | 匹配? |
|---|---|---|---|
| q_proj bias | **True** [896] | 启发式检测 → 误报 | ✗ |
| k_proj bias | **True** [128] | 启发式检测 → 误报 | ✗ |
| v_proj bias | **True** [128] | 启发式检测 → 误报 | ✗ |
| o_proj bias | False | False | ✓ |
| MLP bias | False | False | ✓ |
| q_norm / k_norm | **不存在** (Qwen2) | 不存在 | ✓ |
| rms_norm_eps | **1e-6** | **1e-5 (硬编码!)** | ✗ |
| rope_theta | 1000000.0 | 1000000.0 (QWEN3_SUPPORT) | ✓ |
| RoPE 风格 | LLaMA式 (half-rotate) | CPU: LLaMA式 / GPU: GPT-NeoX式 | GPU ✗ |

---

## 步骤 2：对照 HuggingFace reference

加载 HF `AutoModelForCausalLM`，打印 `named_parameters()`：

```
model.embed_tokens.weight                              [151936, 896]
model.layers.0.self_attn.q_proj.weight                 [896, 896]
model.layers.0.self_attn.q_proj.bias                   [896]          ← 有 bias
model.layers.0.self_attn.k_proj.weight                 [128, 896]
model.layers.0.self_attn.k_proj.bias                   [128]          ← 有 bias
model.layers.0.self_attn.v_proj.weight                 [128, 896]
model.layers.0.self_attn.v_proj.bias                   [128]          ← 有 bias
model.layers.0.self_attn.o_proj.weight                 [896, 896]     ← 无 bias
model.layers.0.mlp.gate_proj.weight                    [4864, 896]    ← 无 bias
model.layers.0.mlp.up_proj.weight                      [4864, 896]    ← 无 bias
model.layers.0.mlp.down_proj.weight                    [896, 4864]    ← 无 bias
model.layers.0.input_layernorm.weight                  [896]
model.layers.0.post_attention_layernorm.weight         [896]
model.norm.weight                                      [896]
```

确认：
- **Qwen2.5 的 q_proj/k_proj/v_proj 全部有 bias**
- RMSNorm 类型为 `Qwen2RMSNorm`（无 bias，仅 weight 参数）
- RoPE 使用 LLaMA 式 `rotate_half`（`modeling_qwen2.py:116-120`）
- 激活函数为 SiLU（Swish），MLP 使用 SwiGLU：`down(silu(gate(x)) * up(x))`

---

## 步骤 3：逐层 Tensor Diff

### 方法

编写 `debug/debug_layer_diff.py`，逐字节解析 `.bin` 文件，与 HF state_dict 进行逐层 weight 比较。

### .bin 文件尺寸分析

| 指标 | float 数量 |
|---|---|
| .bin 文件实际大小 | **498,199,424** |
| 带 QKV bias 的期望大小 | 498,227,072 |
| **不带 QKV bias 的期望大小** | **498,199,424** ← 精确匹配! |
| 差值 | 27,648 = 24×(896+128+128) = 全部 QKV bias 之和 |

### Weight 比较结果

```
embed_tokens                      max_diff=0.00e+00  ✓ 匹配
layer.0.q_proj                    max_diff=0.00e+00  ✓ 匹配
layer.0.q_bias                    max_diff=7.91e+01  ✗ 完全不匹配 (差 100%)
layer.0.k_proj                    max_diff=1.18e+00  ✗ 严重不匹配 (差 115%)
layer.0.k_bias                    max_diff=1.30e+02  ✗ 完全不匹配
layer.0.v_proj                    max_diff=1.25e-01  ✗ 严重不匹配
layer.0.v_bias                    max_diff=9.56e-02  ✗ 完全不匹配
layer.0.o_proj                    max_diff=5.59e-01  ✗ 严重不匹配
layer.0.input_layernorm           max_diff=0.00e+00  ✓ 匹配
layer.0.gate_proj (W1)            max_diff=4.86e-01  ✗ 严重不匹配
layer.0.up_proj (W3)              max_diff=2.94e-01  ✗ 严重不匹配
layer.0.down_proj (W2)            max_diff=3.18e-01  ✗ 严重不匹配
layer.0.post_attn_layernorm       max_diff=2.00e+00  ✗ 严重不匹配
final_norm                        max_diff=1.58e+01  ✗ 严重不匹配
lm_head (tied with embed)         max_diff=0.00e+00  ✓ 匹配
```

### C++ 副本推理 vs HF 参考

使用与 C++ 完全相同的公式（相同 eps、相同 RoPE、相同 matmul 顺序）进行逐 token 推理：

```
Last position, pre-norm hidden (C++ vs HF):
  max_abs_err = 5.28e+01   (期望: < 0.01)
  mean_abs_err = 1.36e+00

Last position, logits (C++ vs HF):
  max_abs_err = 2.45e+01   (期望: < 0.01)
  mean_abs_err = 2.74e+00

C++ replica top-5 logit indices: [21267, 84443, 111166, 65083, 13783]
HF       top-5 logit indices: [108386, 111308, 9707, 112488, 56568]
                               ↑ 完全不重叠
```

### 第一个错误位置

**`model.layers.0.self_attn.q_proj.bias` 的读取位置** — 即 `.bin` 文件中 WQ weight 结束后的位置。

在 `.bin` 文件中，WQ weight 结束后的数据是 **WK weight 的起始**，但 C++ 代码的启发式检测将其误判为 Q bias，从这里读取了 21,504 个 float 作为 "bias 数据"。

验证：跳过 bias 读取，直接从 WQ 结束位置读取 WK：
```
Without bias skip: WK[0] .bin vs HF max_diff = 0.0000000000  ← 完美匹配！
```

**结论：`.bin` 文件中不存在 QKV bias。C++ 启发式检测产生误报，导致从 WK 开始的所有权重全部读错位置。**

---

## 步骤 4：按模块检查

### 4.1 Embedding
- **状态：正确** ✓
- token id、vocab size、embedding weight 加载均正确
- LM Head 通过 weight tying 共享 embedding weight，正确

### 4.2 RMSNorm

**问题 A（致命）：ffn_norm 和 final_norm 的 position tracker 不含 bias 偏移量**

在 `sys/src/model/llama3.cpp:226-231`：
```cpp
// 跳过 attention QKV weights 到达 FFN rmsnorm
rmsnorm_pos += config_->layer_num_ * config_->q_dim_ * dim;    // wq
rmsnorm_pos += config_->layer_num_ * config_->kv_dim_ * dim;  // wk
rmsnorm_pos += config_->layer_num_ * config_->kv_dim_ * dim;  // wv
rmsnorm_pos += config_->layer_num_ * dim * config_->q_dim_;    // wo
```

缺少 bias 大小：`layer_num * (q_dim + kv_dim + kv_dim) = 24 * (896+128+128) = 27,648` float。

当 .bin 文件重新导出包含 bias 后，ffn_norm 的读取位置会偏移 27,648 个 float（落
在 WO 权重数据中间），final_norm 也同样偏移。

**问题 B（中等）：`rms_norm_eps` 硬编码为 `1e-5`，HF 配置为 `1e-6`**

文件位置：
- `sys/src/op/kernel/cpu/rmsnorm_kernel.cpp:25`: `const float eps = 1e-5f;`
- `sys/src/op/kernel/gpu/rmsnorm_kernel.cu:44`: `const float eps = 1e-5f;`

测试表明，仅修正 eps 不足以修复乱码（因为权重损坏的影响远大于 eps 差异）。
但 eps 差异在 24 层 × 2 RMSNorm 的累积下，会引入约 1e-3 量级的数值误差。

### 4.3 Linear (Matmul)
- **状态：公式正确** ✓
- 公式：`Y = X @ W^T + b`
- CPU kernel (Armadillo) 和 GPU kernel (CUDA reduction) 的矩阵乘法逻辑均正确
- 无 transpose 问题，weight layout 为 [out_dim, in_dim]
- **但由于 QKV/FFN 权重全部从错误位置读取，所有 Linear 层输出完全错误**

### 4.4 Attention (MHA)
- QKV shape: [B=1, S, H=14, D=64] ✓
- GQA: num_attention_heads=14, num_key_value_heads=2, kv_mul=7 ✓
- repeat_kv 逻辑正确 ✓
- Attention score: QK^T / sqrt(head_dim) ✓
- **但由于 WK/WV/WO 权重全部移位，QKV projection 的输出完全错误**

### 4.5 RoPE

**CPU kernel (QWEN3_SUPPORT 路径)：正确** ✓
- LLaMA 式 half-rotate，rotate (i, i+half_dim)
- theta = 1000000.0（与 HF 一致）
- Sin/Cos cache 计算正确（`sys/src/op/kernel/cpu/rope_kernel.cpp:10-25`）

**GPU kernel：有两处错误** ✗（当前 demo 用 CPU，不影响结果）
- `sys/src/op/kernel/gpu/rope_kernel.cu:7-30`:
  1. 使用 GPT-NeoX 式相邻对旋转 (idx, idx+1)，而 Qwen2.5 使用 LLaMA 式 half-rotate
  2. Cache 索引：`sin_cache[pos * head_size + head_dim]` 其中 `head_dim = idx % head_size`，
     idx 步长为 2（0, 2, 4, ...），但 cache 中每个频率占据连续位置（0, 1, 2, ...），
     应该用 `head_dim / 2` 索引

### 4.6 MLP (SwiGLU)
- **状态：公式正确** ✓
- SwiGLU: `silu(gate) * up` = `gate * sigmoid(gate) * up`
- `sys/src/op/kernel/cpu/swiglu_kernel.cpp:30`: 公式与 HF Qwen2 `hidden_act=silu` + gate/up/down 一致
- **但由于 W1/W2/W3 权重全部移位，MLP 输出完全错误**

### 4.7 LM Head
- **状态：正确** ✓
- Weight tying: `lm_head.weight = embed_tokens.weight`
- `sys/src/model/llama3.cpp:206-208`: 共享权重时使用 embedding weight 指针

---

## 步骤 5：最终定位与修复方案

### 5.1 第一个错误位置

**文件偏移 155,423,744 float 处**（WQ weight 结束后的第一个 float）。

此处本应是 WK weight layer 0 的第一个元素，但 C++ 代码将其误读为 Q bias。

### 5.2 根本原因

**.bin 文件是用不支持 QKV bias 的旧版导出逻辑生成的，但 C++ 的 bias 检测使用了不可靠的文件尺寸启发式算法，产生了误报（False Positive）。**

具体链：
1. `.bin` 文件中只有 `WQ → WK → WV → WO → ...`（无 bias 穿插）
2. C++ 代码在 [llama3.cpp:112-122](sys/src/model/llama3.cpp#L112-L122) 用 `pos + layer_num*q_dim <= max_floats` 检测 bias
3. 由于 `.bin` 文件足够大（包含大量 freqs 数据），`155,445,248 <= 498,199,424` 为 **True**
4. C++ 在 WQ 结束后读取 21,504 个 float 作为 "Q bias" → 实际是 WK 的前面部分
5. `pos` 被人为推进了 21,504 → WK 读取从偏移位置开始 → 所有后续权重全部移位

### 5.3 修改文件清单

| 优先级 | 文件 | 行号 | 问题 |
|---|---|---|---|
| **P0** | — | — | **重新导出 .bin 文件（包含 QKV bias）** |
| P1 | `sys/src/op/kernel/cpu/rmsnorm_kernel.cpp` | 25 | eps 1e-5 → 1e-6 |
| P1 | `sys/src/op/kernel/gpu/rmsnorm_kernel.cu` | 44 | eps 1e-5 → 1e-6 |
| P1 | `sys/src/model/llama3.cpp` | 226-238 | rmsnorm_pos 缺少 bias 偏移量 |
| P2 | `sys/src/model/llama3.cpp` | 111-122 | bias 启发式检测不可靠（建议用显式 flag） |
| P2 | `sys/src/model/llama3.cpp` | 132-139 | 同上：Wk bias 检测 |
| P2 | `sys/src/model/llama3.cpp` | 149-156 | 同上：Wv bias 检测 |
| P3 | `sys/src/op/kernel/gpu/rope_kernel.cu` | 13-24 | RoPE 风格错误（GPT-NeoX → LLaMA）+ cache 索引错误 |

### 5.4 修改方案

#### P0: 重新导出 .bin 文件

确认 `tools/export_qwen3.py` 使用 `load_hf_model` 正确检测 `has_qkv_bias`，然后执行：

```bash
cd tools
python3 export_qwen3.py \
  ../demo/qwen2.5_0.5b_instruct.bin \
  --version 0 \
  --hf /home/liangji/huggingface/Qwen2.5-0.5B-Instruct
```

导出后验证文件尺寸：
- 期望 float 数量：498,227,072
- 期望文件大小（含 32 字节 header）：约 1.86 GB

#### P1: 修复 RMSNorm eps

**文件：`sys/src/op/kernel/cpu/rmsnorm_kernel.cpp:25`**

```diff
- const float eps = 1e-5f;
+ const float eps = 1e-6f;  // Qwen2.5 rms_norm_eps = 1e-6
```

**文件：`sys/src/op/kernel/gpu/rmsnorm_kernel.cu:44`**

```diff
- const float eps = 1e-5f;
+ const float eps = 1e-6f;  // Qwen2.5 rms_norm_eps = 1e-6
```

*更好方案：从 config.json 读取 eps，通过 TransformerConfig 传入，在 RMSNorm 层构造时使用。*

#### P1: 修复 rmsnorm_pos 计算（含 bias 偏移）

**文件：`sys/src/model/llama3.cpp:226-231`**

```cpp
// 改前
rmsnorm_pos += config_->layer_num_ * config_->q_dim_ * dim;    // wq
rmsnorm_pos += config_->layer_num_ * config_->kv_dim_ * dim;  // wk
rmsnorm_pos += config_->layer_num_ * config_->kv_dim_ * dim;  // wv
rmsnorm_pos += config_->layer_num_ * dim * config_->q_dim_;    // wo

// 改后
rmsnorm_pos += config_->layer_num_ * config_->q_dim_ * dim;    // wq
if (!llama_layers_->q_bias_.empty()) {
    rmsnorm_pos += config_->layer_num_ * config_->q_dim_;       // Wq bias
}
rmsnorm_pos += config_->layer_num_ * config_->kv_dim_ * dim;  // wk
if (!llama_layers_->q_bias_.empty()) {
    rmsnorm_pos += config_->layer_num_ * config_->kv_dim_;      // Wk bias
}
rmsnorm_pos += config_->layer_num_ * config_->kv_dim_ * dim;  // wv
if (!llama_layers_->q_bias_.empty()) {
    rmsnorm_pos += config_->layer_num_ * config_->kv_dim_;      // Wv bias
}
rmsnorm_pos += config_->layer_num_ * dim * config_->q_dim_;    // wo
```

#### P2: 修复 bias 启发式检测（建议）

当前的文件尺寸启发式不可靠。建议：
1. 在 header 中增加 flags 字段（或使用 version 2 格式的 shared_classifier 字节后增加 flags）
2. 或在检测到 bias 后验证值是否在合理范围（bias 的 std 应与权重 std 差异显著不同）

#### P3: 修复 GPU RoPE kernel

**文件：`sys/src/op/kernel/gpu/rope_kernel.cu:7-30`**

改为 LLaMA 式 half-rotate，与 CPU kernel 保持一致：

```cuda
__global__ void rope_kernel_cuda_fp32(int32_t pos, int32_t dim, int32_t kv_dim,
                                       int32_t head_size, float* input_q, float* input_k,
                                       const float* sin_cache, const float* cos_cache) {
  int32_t half_dim = head_size / 2;
  int32_t head_idx = blockDim.x * blockIdx.x + threadIdx.x;
  int32_t num_heads = dim / head_size;
  if (head_idx >= num_heads) return;

  int32_t head_off = head_idx * head_size;
  for (int32_t i = 0; i < half_dim; ++i) {
    float fci = sin_cache[pos * head_size + i];
    float fcr = cos_cache[pos * head_size + i];

    // Q
    float q0 = input_q[head_off + i];
    float q1 = input_q[head_off + i + half_dim];
    input_q[head_off + i] = q0 * fcr - q1 * fci;
    input_q[head_off + i + half_dim] = q0 * fci + q1 * fcr;

    // K
    if (head_idx < kv_dim / head_size) {
      float k0 = input_k[head_off + i];
      float k1 = input_k[head_off + i + half_dim];
      input_k[head_off + i] = k0 * fcr - k1 * fci;
      input_k[head_off + i + half_dim] = k0 * fci + k1 * fcr;
    }
  }
}
```

### 5.5 修改后验证

1. 重新导出 .bin → 确认文件尺寸 = 498,227,072 float（包含 bias）
2. 重新编译 C++ 项目并运行 demo
3. 运行 `debug/debug_layer_diff.py` 验证：
   - 所有权重 max_diff < 1e-5
   - 顶层 hidden state max_diff < 0.01
   - 顶层 logits top-1 与 HF 一致
   - 最终输出文本可读
