#include "matmul_int8_kernel.cuh"
#include <glog/logging.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

namespace kernel {

/**
 * INT8 权重矩阵乘 kernel — 1 block per output element
 *
 * 计算: output[m, n] = scales[n] * Σ_k input[m, k] * weight[n, k] + bias[n]
 *
 * 数据布局:
 *   input  [M, K] fp32 行优先
 *   weight [N, K] int8 行优先
 *   scales [N]    fp32 per-channel
 *   output [M, N] fp32 行优先
 *
 * 每个 block 计算一个 output[m,n] 元素:
 *   - char4 向量化加载 INT8 权重 (4×int8=32bit，一次读取)
 *   - float4 向量化加载 FP32 输入
 *   - 混合精度点积: FP32 accumulate += INT8 weight × FP32 input
 *   - cub::BlockReduce 规约后: result = sum * scales[n] + bias[n]
 */
template <int THREAD_PER_BLOCK>
__global__ void matmul_int8_kernel_fp32(
    const float* __restrict__ input,
    const int8_t* __restrict__ weight,
    const float* __restrict__ scales,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int M, int N, int K) {

  int tid = threadIdx.x;
  int bid = blockIdx.x;

  // block 负责 output[m, n]
  int m = bid / N;
  int n = bid % N;
  if (m >= M || n >= N) return;

  const float* input_row = input + m * K;
  const int8_t* weight_row = weight + n * K;

  // ── char4 向量化 ──
  // INT8 权重按 char4 打包: 每线程读 4 个 int8 = 32 bits
  constexpr int pack_size = 4;
  const int pack_num = K / pack_size;
  const int pack_off = pack_size * pack_num;

  __shared__ float sdata[THREAD_PER_BLOCK];
  sdata[tid] = 0.0f;

  // char4 向量化点积
  const char4* weight_char4_ptr = reinterpret_cast<const char4*>(weight_row);
  const float4* input_float4_ptr = reinterpret_cast<const float4*>(input_row);

#pragma unroll
  for (int i = tid; i < pack_num; i += blockDim.x) {
    char4 w_pack = weight_char4_ptr[i];
    float4 in_pack = input_float4_ptr[i];
    // INT8 × FP32 → FP32 混合精度累加
    sdata[tid] += static_cast<float>(w_pack.x) * in_pack.x
                + static_cast<float>(w_pack.y) * in_pack.y
                + static_cast<float>(w_pack.z) * in_pack.z
                + static_cast<float>(w_pack.w) * in_pack.w;
  }

  // 处理不能被 4 整除的余数
  for (int i = pack_off + tid; i < K; i += blockDim.x) {
    sdata[tid] += static_cast<float>(weight_row[i]) * input_row[i];
  }

  __syncthreads();

  // ── cub::BlockReduce 规约 ──
  using BlockReduce = cub::BlockReduce<float, THREAD_PER_BLOCK>;
  __shared__ typename BlockReduce::TempStorage temp;
  float sum = BlockReduce(temp).Sum(sdata[tid]);

  // 反量化 + bias
  if (tid == 0) {
    float val = sum * scales[n] + (bias ? bias[n] : 0.0f);
    output[m * N + n] = val;
  }
}

// ═══ GPU 对外接口 ═══
void matmul_int8_kernel_cuda(const tensor::Tensor& input,
                              const tensor::Tensor& weight,
                              const tensor::Tensor& scales,
                              const float* bias,
                              const tensor::Tensor& output,
                              void* stream) {
  CHECK(!input.is_empty());
  CHECK(!weight.is_empty());
  CHECK(!scales.is_empty());
  CHECK(!output.is_empty());
  CHECK(weight.get_data_type() == tensor::DataType_t::int8)
      << "Weight must be INT8 for quantized matmul";

  // weight 形状: [N, K]
  const int32_t N = static_cast<int32_t>(weight.get_dim(0));
  const int32_t K = static_cast<int32_t>(weight.get_dim(1));

  // input: 1D [K] 或 2D [M, K]
  int32_t M = 1;
  if (input.get_dims_size() == 2)
    M = static_cast<int32_t>(input.get_dim(0));

  CHECK_EQ(static_cast<int32_t>(input.get_dim(input.get_dims_size() - 1)), K)
      << "Input last dim must equal weight inner dim";
  CHECK_EQ(static_cast<int32_t>(scales.get_size()), N)
      << "Scales must have N elements (per-channel)";

  constexpr int THREADS = 256;
  int grid = M * N;

  if (stream) {
    matmul_int8_kernel_fp32<THREADS>
        <<<grid, THREADS, 0, static_cast<cudaStream_t>(stream)>>>(
            static_cast<const float*>(input.get_ptr()),
            static_cast<const int8_t*>(weight.get_ptr()),
            static_cast<const float*>(scales.get_ptr()),
            bias,
            const_cast<float*>(static_cast<const float*>(output.get_ptr())),
            M, N, K);
  } else {
    matmul_int8_kernel_fp32<THREADS>
        <<<grid, THREADS>>>(
            static_cast<const float*>(input.get_ptr()),
            static_cast<const int8_t*>(weight.get_ptr()),
            static_cast<const float*>(scales.get_ptr()),
            bias,
            const_cast<float*>(static_cast<const float*>(output.get_ptr())),
            M, N, K);
  }
}

// ═══ CPU 回退 ═══
void matmul_int8_kernel_cpu(const tensor::Tensor& input,
                             const tensor::Tensor& weight,
                             const tensor::Tensor& scales,
                             const float* bias,
                             const tensor::Tensor& output,
                             void* stream) {
  (void)stream;

  const int32_t N = static_cast<int32_t>(weight.get_dim(0));
  const int32_t K = static_cast<int32_t>(weight.get_dim(1));
  int32_t M = 1;
  if (input.get_dims_size() == 2)
    M = static_cast<int32_t>(input.get_dim(0));

  const float* in = static_cast<const float*>(input.get_ptr());
  const int8_t* w = static_cast<const int8_t*>(weight.get_ptr());
  const float* s = static_cast<const float*>(scales.get_ptr());
  float* out = const_cast<float*>(static_cast<const float*>(output.get_ptr()));

  for (int32_t m = 0; m < M; ++m) {
    for (int32_t n = 0; n < N; ++n) {
      float sum = 0.0f;
      for (int32_t k = 0; k < K; ++k) {
        sum += static_cast<float>(w[n * K + k]) * in[m * K + k];
      }
      out[m * N + n] = sum * s[n] + (bias ? bias[n] : 0.0f);
    }
  }
}

}  // namespace kernel
