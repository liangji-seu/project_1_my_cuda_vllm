# GPU 推理框架完整审查报告

审查日期: 2026-07-22

---

## 总览

| 层 | 状态 | 致命 Bug | 严重 Bug | 中等 Bug | 轻微问题 |
|---|---|---|---|---|---|
| Tensor/Buffer | ⚠️ 有问题 | 3 | 0 | 4 | 3 |
| Op/Kernel | ⚠️ 有问题 | 1 | 0 | 3 | 7 |
| Model | ✅ 基本正确 | 0 | 0 | 0 | 2 |
| Demo/Chat | ❌ 不支持 GPU | 2 | 0 | 0 | 0 |

---

## 一、Tensor/Buffer 层 — GPU 内存管理

### 1.1 正常工作的部分

- **`to("cuda")` / `to("cpu")`**: 正确。分配新 Buffer → 通过 DeviceController::mem_copy 拷贝数据 → 替换 buffer。CPU→GPU 使用 `cudaMemcpyHostToDevice`，GPU→CPU 使用 `cudaMemcpyDeviceToHost`。
- **`DeviceController::mem_copy()`**: 正确。支持全部 4 种拷贝方向 (CPU↔CPU, CPU↔GPU, GPU↔CPU, GPU↔GPU)，支持 stream 参数和 need_sync。
- **`Buffer::~Buffer()`**: 正确。仅释放自己分配的内存 (flag_is_external == false)。
- **`Buffer::buffer_copy_from()`**: 正确。通过 controller->mem_copy 分发。
- **`Buffer::buffer_self_allocate()`**: 正确。重新分配内存（之前的 bug 已修复）。
- **`GPUDeviceController::mem_alloc()`**: 正确。大块/小块缓存池 + cudaMalloc 兜底。

### 1.2 致命 Bug

#### BUG-1 [CRITICAL] `DeviceController::mem_release()` — CHECK 崩溃

**文件**: [DeviceController.cpp:221](sys/src/base/DeviceController.cpp#L221)

```cpp
void GPUDeviceController::mem_release(void* ptr) {
    CHECK(ptr != nullptr);
    CHECK(!mini_block_map.empty());  // ← 如果从未分配过小块，这里直接崩溃！
```

**场景**: 如果只分配了大块内存（权重加载全部走大块路径），第一次 `mem_release` 就会 CHECK-fail 崩溃。

#### BUG-2 [CRITICAL] `DeviceController::mem_release()` — 大块搜索不可达

**文件**: [DeviceController.cpp:246-263](sys/src/base/DeviceController.cpp#L246-L263)

```cpp
for(auto& it: mini_block_map){          // ← 外层循环遍历 mini_block_map
    auto& mini_block_list = it.second;
    for(int i = 0; i < mini_block_list.size(); i++){
        if(mini_block_list[i].ptr == ptr){ ... return; }
    }

    auto& big_block_list = big_block_map[it.first];  // ← 大块搜索在 mini_block_map 循环内部！
    for(int i = 0; i < big_block_list.size(); i++){
        if(big_block_list[i].ptr == ptr){ ... return; }
    }
}
```

**问题**: 大块搜索嵌套在 `for(auto& it: mini_block_map)` 内部。如果 `mini_block_map` 为空，大块搜索永远执行不到。即使不为空，也只在 mini_block_map 有对应 device entry 时才搜索大块。

#### BUG-3 [CRITICAL] `Tensor::slice()` — 使用 std::memcpy 而非 controller->mem_copy

**文件**: [tensor.cpp:350](sys/src/tensor/tensor.cpp#L350)

```cpp
std::memcpy(dst, src, inner_bytes);  // ← GPU 张量上直接 CPU 内存拷贝！
```

**问题**: slice 操作绕过 DeviceController，直接用 `std::memcpy`。如果张量在 GPU 上，这会 segfault 或静默产生错误数据。

### 1.3 中等 Bug

#### BUG-4 [MODERATE] `Tensor::clone()` — 无 controller 时崩溃

**文件**: [tensor.cpp:137-145](sys/src/tensor/tensor.cpp#L137-L145)

如果原始张量的 buffer 没有 controller（外部内存），clone 会 CHECK-fail。

#### BUG-5 [MODERATE] `Tensor::reshape()` — 无 controller 时崩溃

**文件**: [tensor.cpp:85-108](sys/src/tensor/tensor.cpp#L85-L108)

同理，扩容 reshape 时如果 buffer 没有 controller 会崩溃。

#### BUG-6 [MODERATE] `Tensor::init_buffer()` — 外部内存 device_type 为 Unknown

**文件**: [tensor.cpp:179-190](sys/src/tensor/tensor.cpp#L179-L190)

使用外部 ptr 时 allocator == nullptr，Buffer 构造函数跳过 device_type 设置（因为 ptr != nullptr 跳过了 controller 分支），导致 device_type 始终为 Unknown。

#### BUG-7 [MODERATE] `GPUDeviceController::mem_release()` — 大块永不释放

大块只标记 `busy = false`，永不调用 `cudaFree`。长时间运行会导致显存池无限增长。

### 1.4 轻微问题

#### BUG-8 [LOW] Buffer 构造函数 — buffer_device_type 参数无用

`Buffer(byte_size, ptr, buffer_device_type, controller, flag_is_external)` 中的 `buffer_device_type` 参数从未赋值给 `this->device_type`，是无用代码。

#### BUG-9 [LOW] `peek_index` / `peek_position` — GPU 不安全

```cpp
template<typename T>
T& Tensor::peek_index(size_t offset){
    auto start = static_cast<T*>(this->buffer->get_ptr()) + offset;
    return *start;  // ← GPU 指针在 CPU 端解引用 → segfault
}
```

这些方法仅适用于 CPU 张量。对 GPU 张量调用会崩溃。

#### BUG-10 [LOW] `to()` — 无 buffer 时崩溃

`CHECK_NE(this->buffer, nullptr)` 在没有 buffer 时直接崩溃。

---

## 二、Op/Kernel 层 — 算子 GPU 支持

### 2.1 架构

```
BaseLayer (device_type, layer_type, data_type)
  ├── Layer (inputs, outputs, cuda_stream)
  │     ├── LayerParam (weights, to()方法)  ← 仅参数层有 to()
  │     │     ├── MatmulLayer     ✅ 有 to()
  │     │     ├── RmsNormLayer    ✅ 有 to()
  │     │     └── EmbeddingLayer  ✅ 有 to()
  │     ├── VecAddLayer           ❌ 无 to() (无参数)
  │     ├── MultiHeadAttention    ❌ 无 to() (无参数)
  │     ├── RoPELayer             ❌ 无 to() (无参数)
  │     ├── SoftmaxLayer          ❌ 无 to() (无参数)
  │     └── SwiGLULayer           ❌ 无 to() (无参数)
  └── EncodeLayerBase             🔒 始终 CPU
```

所有算子都通过 `kernel_interface.cpp` 根据 `device_type` 分发到 CPU/GPU kernel。

### 2.2 各算子状态

| Op | to() | kernel_interface | GPU forward | cuda_stream | GPU kernel |
|---|---|---|---|---|---|
| **Add** | ❌ | ✅ | ✅ 正确 | ✅ | 无 bug |
| **MatMul** | ✅ | ✅ | ✅ 正确 | ✅ | ⚠️ 中等 bug |
| **Embedding** | ✅ | ✅ | ✅ 正确 | ✅ | ⚠️ 中等 bug |
| **RMSNorm** | ✅ | ✅ | ✅ 正确 | ✅ | 无 bug |
| **RoPE** | ❌ | ✅ | ✅ 正确 | ✅ | 无 bug |
| **MHA** | ❌ | ✅ | ✅ 正确 | ✅ | ⚠️ 性能问题 |
| **Softmax** | ❌ | ✅ | 🔴 严重 bug | ✅ | kernel 正确 |
| **SwiGLU** | ❌ | ✅ | ✅ 正确 | ✅ | ⚠️ 轻微问题 |

### 2.3 致命 Bug

#### BUG-11 [CRITICAL] Softmax — GPU 张量上使用 std::memcpy

**文件**: [softmax.cpp:33-35](sys/src/op/softmax.cpp#L33-L35)

```cpp
auto output = this->get_output(0);
if (output.get_ptr() != input.get_ptr()) {
    size_t byte_size = input.get_byte_size();
    std::memcpy(output.get_ptr(), input.get_ptr(), byte_size);  // ← GPU 内存上 CPU memcpy！
}
```

**问题**: softmax kernel 是 in-place 操作，但如果 input 和 output 指向不同张量，会用 `std::memcpy` 拷贝。这在 GPU 上会崩溃或静默出错。

### 2.4 中等 Bug

#### BUG-12 [MODERATE] MatMul GPU kernel — 忽略 batch 维度

**文件**: [matmul_kernel.cu](sys/src/op/kernel/gpu/matmul_kernel.cu)

GPU kernel 将输入当作 1D 数组处理。CPU 版本正确处理 1D 和 2D 输入。如果传入 2D (batch) 输入，GPU 计算结果错误。

#### BUG-13 [MODERATE] Embedding GPU kernel — 缺少 token 越界检查

**文件**: [emb_kernel.cu](sys/src/op/kernel/gpu/emb_kernel.cu)

CPU kernel 检查 `token < vocab_size` 并报 fatal error。GPU kernel 不做检查，越界 token 会静默读取非法 GPU 内存。

### 2.5 轻微问题

#### BUG-14 [LOW] MHA GPU kernel — 严重欠并行化

**文件**: [mha_kernel.cu](sys/src/op/kernel/gpu/mha_kernel.cu)

一个线程处理一个 head。对 32 head 模型只用 32 个线程，GPU 利用率极低。

#### BUG-15 [LOW] 所有 GPU kernel — 缺少 launch error 检查

所有 `.cu` 文件中 kernel launch 后没有 `cudaGetLastError()` 检查，静默的 launch 失败无法被检测。

#### BUG-16 [LOW] 所有 op forward — 缺少 dispatch nullptr 检查

如果 device_type 为 Unknown，kernel_interface 返回 nullptr，直接调用空函数指针会 segfault。

#### BUG-17 [LOW] LayerParam::to() — stream 可能为 nullptr

如果 Layer 的 cuda_stream 为 nullptr，`to()` 传递 nullptr stream，降级为同步拷贝，但不影响正确性。

#### BUG-18 [LOW] SwiGLU GPU kernel — 不必要的 shared memory

使用了 shared memory 缓存输入但每个元素只读一次，浪费 shared memory 但不影响正确性。

---

## 三、Model 层 — 模型推理

### 3.1 正常工作的部分

| 组件 | 状态 | 说明 |
|---|---|---|
| `init()` GPU 设置 | ✅ | 条件路径，仅 GPU 模式创建 stream/传输 |
| `transfer_to_device()` 权重 | ✅ | 覆盖所有参数层 (WQ/WK/WV/WO/W1/W2/W3/RMSNorm/CLS/Embedding) |
| `transfer_to_device()` 缓冲区 | ✅ | 覆盖所有计算缓冲区，正确跳过 kInputTokens/kInputPos |
| `set_cuda_stream_on_all_layers()` | ✅ | 覆盖所有向量层 + 单例层，无遗漏 |
| `forward()` device_type | ✅ | 所有张量保持正确设备 |
| `post_processing()` | ✅ | 正确 GPU→CPU→GPU 往返 |
| `embedding()` | ✅ | GPU 路径正确 |
| `fill_input()` | ✅ | GPU 视图张量创建正确 |

### 3.2 问题

#### BUG-19 [LOW] `attention_qkv()` — bias/QK norm 每步 GPU↔CPU 往返

**文件**: [llama3.cpp:492-532](sys/src/model/llama3.cpp#L492-L532)

每个 decode step 每层都要把 Q/K/V 张量拷回 CPU 加 bias，再拷回 GPU。功能正确但性能差（每层每步 6 次设备间拷贝）。

**建议**: 将 bias 作为额外的 GPU 张量预先传输，在 GPU 上做 bias addition。

#### BUG-20 [LOW] `cls_logits()` — const_cast 去除 const

**文件**: [llama3.cpp:623](sys/src/model/llama3.cpp#L623)

```cpp
tensor::Tensor& input_mut = const_cast<tensor::Tensor&>(input);
STATUS_CHECK(norm->forward(input, input_mut));  // in-place 修改 const 引用
```

这绕过 const 保护做 in-place RMSNorm，属于未定义行为（虽然当前实现恰好工作）。

---

## 四、Demo/Chat 层 — 应用入口

### 4.1 致命 Bug

#### BUG-21 [CRITICAL] peek_index 直接写 GPU 内存

**文件**: [chat.cpp:57](demo/chat.cpp#L57) 和 [main.cpp](demo/main.cpp)

```cpp
pos_tensor.peek_index<int32_t>(0) = pos;  // ← pos_tensor 在 GPU 上时直接 CPU 写设备内存！
```

`peek_index` 实现为 `static_cast<T*>(ptr) + offset` 然后直接解引用。GPU 设备指针从 CPU 端解引用 → segfault。

#### BUG-22 [CRITICAL] 两个 demo 都硬编码 CPU

**文件**: [chat.cpp:104](demo/chat.cpp#L104), [main.cpp:68](demo/main.cpp)

```cpp
auto init_status = model.init(base::DeviceType_t::CPU);  // ← 硬编码 CPU
```

从未使用 `DeviceType_t::GPU`。

---

## 五、修复优先级

### 第一优先级 (必须修复，阻止 GPU 正常运行)

| # | Bug | 影响 |
|---|---|---|
| BUG-21 | peek_index 写 GPU 内存 | Demo 在 GPU 模式下 segfault |
| BUG-22 | Demo 硬编码 CPU | 无法切换到 GPU |
| BUG-11 | Softmax std::memcpy GPU 内存 | MHA 中 softmax 可能被调用 |
| BUG-1 | mem_release CHECK 崩溃 | 权重释放时崩溃 |
| BUG-3 | slice std::memcpy GPU 内存 | 如果 slice 被调用会崩溃 |

### 第二优先级 (影响正确性但不一定触发)

| # | Bug | 影响 |
|---|---|---|
| BUG-2 | 大块搜索不可达 | GPU 大块释放走不到缓存池 |
| BUG-12 | MatMul 忽略 batch 维度 | 批量推理时计算错误 |
| BUG-13 | Embedding 缺少越界检查 | 坏 token 静默读非法内存 |

### 第三优先级 (性能/代码质量)

| # | Bug | 影响 |
|---|---|---|
| BUG-19 | Bias GPU↔CPU 往返 | 每层每步额外拷贝 |
| BUG-14 | MHA 欠并行化 | GPU 利用率低 |
| BUG-7 | 大块永不释放 | 显存池增长 |
| BUG-20 | const_cast | 代码质量问题 |
