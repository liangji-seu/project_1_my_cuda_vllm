# GPU 乱码问题诊断报告

诊断日期: 2026-07-22

---

## 结论：KV Cache 写入丢失

**GPU 推理输出乱码的根本原因是：attention_qkv 中的 bias 加法路径破坏了 KV cache 的数据流，导致 K/V 在投影后、加 bias 和做 RoPE 之前就从 key_cache/value_cache 中"丢失"了。**

---

## 详细分析

### 代码路径（llama3.cpp attention_qkv 函数）

```cpp
void LLama2Model::attention_qkv(int32_t layer_idx, int32_t pos) {
    tensor::Tensor& query = get_buffer(ModelBufferType::kQuery);     // 引用 → 指向 buffers_[kQuery]
    auto [key, val] = slice_kv_cache(layer_idx, pos);               // ← 局部变量！值拷贝！

    // Q/K/V 投影 — 写入 key_cache（此时 key 共享 key_cache 的 buffer）
    STATUS_CHECK(query_layer->forward(rmsnorm_output, query));
    STATUS_CHECK(key_layer->forward(rmsnorm_output, key));          // 写到 key_cache ✓
    STATUS_CHECK(value_layer->forward(rmsnorm_output, val));        // 写到 val_cache ✓

    // Bias 加法 — 这里出问题！
    if (!llama_layers_->q_bias_.empty()) {
        add_bias(query, ...);  // query 是引用 → 更新 buffers_[kQuery] ✓
        add_bias(key, ...);    // key 是局部变量 → 创建新 buffer，key_cache 未更新 ✗
        add_bias(val, ...);    // val 是局部变量 → 创建新 buffer，val_cache 未更新 ✗
    }

    // RoPE — 在局部变量的 buffer 上做，key_cache 仍然没被更新！
    rope_layer_->forward(query, key, ...);  // key 已经指向新 buffer，不是 key_cache！
}
```

### 关键区别：query vs key/val

| 变量 | 类型 | 指向 | add_bias 的行为 |
|---|---|---|---|
| `query` | `Tensor&` 引用 | `buffers_[kQuery]` | `to()` 替换 buffer → 更新 buffers_ | 
| `key` | `Tensor` 值（局部） | 和 `buffers_[kKeyCache]` 共享同一个 `shared_ptr<Buffer>` | `to()` 替换 buffer → 只更新局部变量！ |
| `val` | `Tensor` 值（局部） | 同上 | 同上 |

### add_bias 的 GPU 路径做了什么

```cpp
auto add_bias = [this](tensor::Tensor& t, const float* bias, int32_t dim) {
    if (device_type_ == base::DeviceType_t::GPU) {
        t.to("cpu", nullptr);    // 1. GPU→CPU: 创建新 CPU buffer, 拷贝数据, 替换 t.buffer
    }
    float* ptr = ...;
    for (int32_t j = 0; j < dim; ++j) ptr[j] += bias[j];  // 2. CPU 上加 bias
    if (device_type_ == base::DeviceType_t::GPU) {
        t.to("cuda", nullptr);   // 3. CPU→GPU: 创建新 GPU buffer, 拷贝数据, 替换 t.buffer
    }
};
```

- 对于 `query`（引用）：`t.to("cuda")` 替换了 `buffers_[kQuery]` 的 buffer → 后续代码能看到更新后的 query ✓
- 对于 `key`（局部变量）：`t.to("cuda")` 替换了局部变量的 buffer → **`buffers_[kKeyCache]` 仍然指向原始的、未加 bias 的 GPU 内存** ✗

### 数据流断裂示意

```
K 投影:     key_cache[pos] = Wk * rmsnorm_output           ← 写入 key_cache ✓
Bias 加法:  key.to("cpu") → 新 CPU buffer                  ← 从 key_cache 读出
            bias 加到 CPU buffer                            ← 修改 CPU buffer
            key.to("cuda") → 新 GPU buffer                 ← 写入新 GPU buffer
            key_cache[pos] 仍然 = Wk * rmsnorm_output      ← KEY_CACHE 没变!!!
RoPE:       key buffer 上做 rotate                         ← 在新 GPU buffer 上做
            key_cache[pos] 仍然没有 RoPE                    ← KEY_CACHE 没变!!!
MHA 读取:   key_head = key_cache[layer][t][kv_head]        ← 读到的是未加bias、未RoPE 的 K！
```

### 为什么 Q 不受影响

`query` 是 `Tensor&` 引用，直接指向 `buffers_[kQuery]`。`to()` 替换 buffer 时会更新 `buffers_` 中的条目。后续 `attention_mha` 从 `buffers_[kQuery]` 读取 query → 能拿到正确的、加了 bias、做了 RoPE 的 Q。

### 对 CPU 路径无影响的原因

CPU 路径：`device_type_ == CPU`，`add_bias` 中的 `to("cpu")` 和 `to("cuda")` 都是 no-op（已经在目标设备上了）。所以对于局部变量 `key`，bias 直接加在 key_cache 的 CPU 内存上。✓

---

## 影响范围

这个 bug 影响所有 `FLAG_HAS_QKV_BIAS` 为 true 的模型在 GPU 上的推理。对于 Qwen2.5-0.5B-Instruct，QKV bias 存在，所以 GPU 推理完全乱码。

MHA 读到的 K/V 是：
- **K**: Wk * rmsnorm_output（无 bias，无 RoPE）
- **V**: Wv * rmsnorm_output（无 bias）

这导致 attention score 计算完全错误，进而导致所有后续层的输出错误，最终 logits 完全乱码。

---

## 修复方向（供参考）

1. **简单方案**：在 bias 加法循环中，不调用 `to()`。改为手动 cudaMemcpy：
   - 分配 CPU staging buffer
   - cudaMemcpy DeviceToHost（从 key_cache 读到 staging）
   - CPU 上加 bias
   - cudaMemcpy HostToDevice（从 staging 写回 key_cache）

2. **更好方案**：把 bias 预加载到 GPU，写一个 GPU bias addition kernel，直接在 GPU 上完成 bias + RoPE，零拷贝。

---

## 其他 GPU Kernel 审查结论

已逐一对比所有 8 个 GPU kernel 与 CPU kernel 的实现，**算法层面全部一致，无逻辑错误**：

| Kernel | 审查结果 |
|---|---|
| Embedding | ✅ 正确 |
| RMSNorm | ✅ 正确（eps=1e-6，公式一致）|
| MatMul | ✅ 正确（支持 1D/2D，归约正确）|
| RoPE | ✅ 正确（LLaMA-style rotate_half，cache 索引一致）|
| MHA | ✅ 正确（GQA repeat_kv，softmax，weighted sum 均一致）|
| SwiGLU | ✅ 正确（silu(gate) * up）|
| Add | ✅ 正确（逐元素加法）|
| Softmax | ✅ 正确（in-place，max-exp-sum-normalize）|

既然所有 kernel 算法都正确，乱码的唯一原因就是 **输入数据不对** — KV cache 中的数据不是正确的 K/V 值。
