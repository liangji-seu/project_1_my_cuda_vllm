# 算子层设计指南

## 架构概览

```
Layer Frontend (include/op/xxx.h + src/op/xxx.cpp)
    │
    │  forward() calls check_layer() then dispatches via kernel_interface
    ▼
Kernel Interface (kernel_interface.h / kernel_interface.cpp)
    │
    │  get_xxx_interface(device_type) returns CPU or GPU backend
    ├──▶ CPU Backend (kernel/cpu/xxx_kernel.h + xxx_kernel.cpp)
    └──▶ GPU Backend (kernel/gpu/xxx_kernel.cuh + xxx_kernel.cu)
```

## 类继承关系

```
BaseLayer
  └── Layer          ← 无权重的算子
        └── LayerParam  ← 有权重的算子
```

## 判断算子该继承谁

| 条件 | 继承 | 示例 |
|------|------|------|
| 无权重，纯计算 | `Layer` | Add, MHA, Softmax |
| 有权重（可学习参数） | `LayerParam` | Embedding, RMSNorm, Linear, SwiGLU |

## 已实现的算子

### Add (无权重 → Layer)

逐元素向量加法。

```
输入: 2个 (x1, x2)
输出: 1个 (y)
公式: y[i] = x1[i] + x2[i]
```

| 文件 | 路径 |
|------|------|
| 头文件 | [sys/include/op/add.h](../sys/include/op/add.h) |
| 前端 | [sys/src/op/add.cpp](../sys/src/op/add.cpp) |
| CPU后端 | [sys/src/op/kernel/cpu/add_kernel.cpp](../sys/src/op/kernel/cpu/add_kernel.cpp) |
| GPU后端 | [sys/src/op/kernel/gpu/add_kernel.cu](../sys/src/op/kernel/gpu/add_kernel.cu) |

---

### Embedding (有权重 → LayerParam)

词嵌入查表。根据 token ID 从词表中取出对应的稠密向量。

```
输入: 2个 — input(0): token IDs [token_num], input(1): token数量
权重: 1个 — 词表 [vocab_size, dim]
输出: 1个 — 词向量 [token_num, dim]
```

| 文件 | 路径 |
|------|------|
| 头文件 | [sys/include/op/embedding.h](../sys/include/op/embedding.h) |
| 前端 | [sys/src/op/embedding.cpp](../sys/src/op/embedding.cpp) |
| CPU后端 | [sys/src/op/kernel/cpu/emb_kernel.h](../sys/src/op/kernel/cpu/emb_kernel.h) / [emb_kernel.cpp](../sys/src/op/kernel/cpu/emb_kernel.cpp) |
| GPU后端 | [sys/src/op/kernel/gpu/emb_kernel.cuh](../sys/src/op/kernel/gpu/emb_kernel.cuh) / [emb_kernel.cu](../sys/src/op/kernel/gpu/emb_kernel.cu) |

---

### RMSNorm (有权重 → LayerParam)

均方根归一化。对输入做归一化，再乘可学习的 gamma 缩放参数。

```
输入: 1个 — 输入向量 [dim]
权重: 1个 — gamma 缩放参数 [dim]
输出: 1个 — 归一化输出 [dim]
公式: y = x / sqrt(mean(x²) + ε) * weight
```

| 文件 | 路径 |
|------|------|
| 头文件 | [sys/include/op/rmsnorm.h](../sys/include/op/rmsnorm.h) |
| 前端 | [sys/src/op/rmsnorm.cpp](../sys/src/op/rmsnorm.cpp) |
| CPU后端 | [sys/src/op/kernel/cpu/rmsnorm_kernel.h](../sys/src/op/kernel/cpu/rmsnorm_kernel.h) / [rmsnorm_kernel.cpp](../sys/src/op/kernel/cpu/rmsnorm_kernel.cpp) |
| GPU后端 | [sys/src/op/kernel/gpu/rmsnorm_kernel.cuh](../sys/src/op/kernel/gpu/rmsnorm_kernel.cuh) / [rmsnorm_kernel.cu](../sys/src/op/kernel/gpu/rmsnorm_kernel.cu) |

---

### MHA (无权重 → Layer)

多头自注意力。Q/K/V 已由上游 Linear 层投影完成，MHA 只负责注意力计算。支持 GQA（Grouped Query Attention）。

```
输入: 5个
  - input(0): query tensor [head_num, head_size]
  - input(1): score buffer [head_num, seq_len]  ← 临时存放注意力分数
  - input(2): key_cache [layer, seq_len, kv_dim]
  - input(3): value_cache [layer, seq_len, kv_dim]
  - input(4): 预留
输出: 1个 — 注意力输出 [head_num, head_size]

参数:
  - layer_index: 第几层（用于索引KV Cache）
  - pos: 当前token位置（自回归时递增）
  - kv_mul: Q头数是KV头数的几倍（MHA=1, GQA>1）
  - kv_dim: K/V的维度 = (head_num/kv_mul) * head_size
  - seq_len: 最大上下文窗口
  - head_num: 注意力头数
  - head_size: 每个头的维度

计算流程:
  对每个 head h:
    1. score[t] = Q_h · K_{h/kv_mul}[t] / sqrt(head_size)  (t = 0..pos)
    2. softmax(score[0..pos])
    3. output_h = sum_t score[t] * V_{h/kv_mul}[t]
```

| 文件 | 路径 |
|------|------|
| 头文件 | [sys/include/op/mha.h](../sys/include/op/mha.h) |
| 前端 | [sys/src/op/mha.cpp](../sys/src/op/mha.cpp) |
| CPU后端 | [sys/src/op/kernel/cpu/mha_kernel.h](../sys/src/op/kernel/cpu/mha_kernel.h) / [mha_kernel.cpp](../sys/src/op/kernel/cpu/mha_kernel.cpp) |
| GPU后端 | [sys/src/op/kernel/gpu/mha_kernel.cuh](../sys/src/op/kernel/gpu/mha_kernel.cuh) / [mha_kernel.cu](../sys/src/op/kernel/gpu/mha_kernel.cu) |

---

### Encode (无权重 → Layer，CPU专用)

BPE 分词器。不走 kernel dispatch 模式，继承 Layer 但在 `forward()` 里直接调用 tiktoken。

```
输入: 无（encode接收字符串，不走张量）
输出: 无（decode返回字符串）
```

| 文件 | 路径 |
|------|------|
| 头文件 | [sys/include/op/encode.h](../sys/include/op/encode.h) |
| 实现 | [sys/src/op/encode.cpp](../sys/src/op/encode.cpp) |

---

## 新增算子的标准步骤

以新增一个 **Matmul** 为例：

### 1. 确认继承关系

确定有权重还是无权：
- Linear/Matmul 有权重 W 和偏置 b → **LayerParam**

### 2. 在 `kernel_interface.h` 定义后端函数类型

```cpp
typedef void (*Matmul_backend)(
    const tensor::Tensor& x,
    const tensor::Tensor& w,
    float scale,
    const tensor::Tensor& y,
    void* stream
);

Matmul_backend get_matmul_interface(base::DeviceType_t device_type);
```

### 3. 写头文件 `sys/include/op/matmul.h`

```cpp
class MatmulLayer : public LayerParam {
public:
  explicit MatmulLayer(base::DeviceType_t device_type, ...);
  base::error::Status check_layer() override;
  base::error::Status forward() override;
};
```

### 4. 写前端 `sys/src/op/matmul.cpp`

```cpp
base::error::Status MatmulLayer::forward() {
  CHECK(this->check_layer() == base::error::Status());
  // ... get inputs/outputs/weights ...
  kernel::get_matmul_interface(device_type)(input, weight, scale, output, stream_ptr);
  return base::error::Status();
}
```

### 5. 写 CPU 后端

- `sys/src/op/kernel/cpu/matmul_kernel.h` — 声明
- `sys/src/op/kernel/cpu/matmul_kernel.cpp` — 实现

### 6. 写 GPU 后端

- `sys/src/op/kernel/gpu/matmul_kernel.cuh` — 声明
- `sys/src/op/kernel/gpu/matmul_kernel.cu` — 实现（含 `__global__` kernel）

### 7. 注册 `kernel_interface.cpp`

```cpp
#include "cpu/matmul_kernel.h"
#include "gpu/matmul_kernel.cuh"

Matmul_backend get_matmul_interface(base::DeviceType_t device_type){
    if(device_type == base::DeviceType_t::CPU){
        return matmul_kernel_cpu;
    } else if(device_type == base::DeviceType_t::GPU){
        return matmul_kernel_cuda;
    } else {
        LOG(ERROR)<<"error device type";
        return nullptr;
    }
}
```

### 8. 写测试

在 `test/` 下创建 `test_matmul_op.cpp`，参考 [test/test_add_op.cpp](../test/test_add_op.cpp)。

---

## Kernel 接口调度

`kernel_interface.h` 中注册了所有算子的后端接口：

| 算子 | 后端类型 | get函数 |
|------|----------|---------|
| Add | `Add_backend` | `get_add_interface` |
| Embedding | `Embedding_backend` | `get_emb_interface` |
| RMSNorm | `RMSNorm_backend` | `get_rmsnorm_interface` |
| MHA | `MHA_backend` | `get_mha_interface` |
| Linear | `Linear_backend` | (待实现) |
| Matmul | `Matmul_backend` | (待实现) |
| SwiGLU | `SwiGLU_backend` | (待实现) |

## CMake 注意事项

`aux_source_directory` 不是递归的，新增 kernel 子目录需要显式添加：

```cmake
aux_source_directory(sys/src/op/kernel/cpu DIR_OP_KERNEL_CPU)
aux_source_directory(sys/src/op/kernel/gpu DIR_OP_KERNEL_GPU)
```

新增 `.cpp` / `.cu` 文件后需重新执行 `cmake -S . -B build` 让 CMake 重新扫描。
