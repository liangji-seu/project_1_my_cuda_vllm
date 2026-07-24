Ready for review
Select text to add comments on the plan
INT8 Weight-Only Quantization Plan
Context
将 Qwen2.5-0.5B-Instruct 的 matmul 权重从 FP32 量化为 INT8（per-channel symmetric），激活值保持 FP32。权重显存减少 4x，内存带宽压力降低 ~4x。

改动概览（7 步，按依赖顺序）
Step 1: 扩展基础设置支持 INT8
1a. RawModelDataInt8 加 scale_data 指针 — raw_model_data.h + raw_model_data.cpp

新增 float* scale_data 成员，指向 mmap 中 scale 数组起始位置
新增 scale(size_t offset) 方法，返回 scale_data + offset
1b. 量化文件头增加 total_int8_elems 字段 — model.cpp

当前 quant header: ModelConfig(32B) | flags(4B) | group_size(4B)
改为: ModelConfig(32B) | flags(4B) | group_size(4B) | total_int8_elems(8B)
read_model_file() 读取该字段，据此计算 scale_data 偏移
1c. LayerParam::set_weight() 支持 data_type 参数 — layer.h + layer.cpp

添加 tensor::DataType_t data_type = fp32 参数（默认值保证向后兼容）
Tensor 构造使用该参数而非硬编码的 this->data_type
Step 2: 添加 INT8 Matmul 内核接口
2a. 新增 MatmulInt8_backend 类型 — kernel_interface.h

typedef void (*MatmulInt8_backend)(
    const Tensor& input,     // fp32 [M, K]
    const Tensor& weight,    // int8 [N, K]
    const Tensor& scales,    // fp32 [N] per-channel
    const float* bias,       // fp32 [N] optional
    const Tensor& output,    // fp32 [M, N]
    void* stream);
MatmulInt8_backend get_matmul_int8_interface(DeviceType_t);
2b. 注册调度 — kernel_interface.cpp

GPU → matmul_int8_kernel_cuda, CPU → matmul_int8_kernel_cpu
Step 3: 写 INT8 Matmul GPU Kernel
新建文件: matmul_int8_kernel.cuh + matmul_int8_kernel.cu

沿用现有 FP32 kernel 的 1-block-per-output-element 模式（适合 M=1 decode）
256 线程/block，int4 打包加载 INT8 权重（一次读 4 个 int8），float4 加载输入
cub::BlockReduce 规约后：val = sum * scales[n] + bias[n]
CPU 回退：简单三重循环（正确性验证用）
Step 4: MatmulLayer 支持量化分发
修改: matmul.h + matmul.cpp

构造函数加 bool is_quant = false 参数
forward() 检查 is_quant_layer：true → get_matmul_int8_interface()，false → 原 FP32 路径
check_layer() 量化模式下校验 scale 非空、weight data_type == int8
Step 5: 实现 create_param_quant_layers()
修改: llama3.cpp

量化的 .bin 文件布局（header 之后）：

[int8 weights 顺序存放] [float scales 顺序存放] [float 非量化权重 (embed/rmsnorm/rope)]
每个量化 matmul 权重（Wq/Wk/Wv/Wo/W1/W2/W3/CLS）的模式：

set_weight(0, {N,K}, ptr, CPU, DataType_t::int8) — INT8 权重
构造 scale Tensor，调用 set_scales() — per-channel float [N]
bias（如有）— 仍是 float
Embedding、RMSNorm、RoPE 保持 FP32。

Step 6: Python 量化脚本
新建: tools/quantize_model.py

读取 FP32 .bin，按 create_param_layers() 相同顺序遍历权重
每个 matmul 权重做 per-channel symmetric quant: scale = max(|row|) / 127
输出 INT8 .bin（header + int8 weights + float scales + float non-quant weights）
无需 PyTorch，纯 numpy + struct
Step 7: 集成
main.cpp 加 --quant 参数，传给 LLama2Model 构造函数的 is_quant_model 参数
LayerParam::to() 加 scales 的 GPU 传输
CMakeLists.txt 无需改动（aux_source_directory 自动拾取新 .cu 文件）
关键设计决策
独立 MatmulInt8_backend 而非扩展原有接口 — 权重类型不同（int8 vs float），scale 语义不同（per-channel tensor vs scalar），改动隔离
只量化 matmul 权重 — Embedding/RMSNorm/RoPE 保持 FP32，这些不是瓶颈
Per-channel symmetric — 实现最简单，精度损失最小（相比 per-tensor）
验证
python tools/quantize_model.py demo/qwen2.5_0.5b_instruct.bin -o demo/qwen2.5_0.5b_instruct_int8.bin
./bin/demo --model demo/qwen2.5_0.5b_instruct_int8.bin --quant --max-new-tokens 128 — 确认生成文本合理
./bin/demo --model ... --quant --benchmark --warmup 3 --repeat 10 — 对比 FP32 的 TPOT
cd build && cmake .. && make -j$(nproc) && cd .. — 确认编译通过