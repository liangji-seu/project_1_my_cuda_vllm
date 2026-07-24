#include "matmul_int8_kernel.cuh"
#include <glog/logging.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

namespace kernel {

/**
 * INT8 权重矩阵乘 kernel — 完全匹配 teacher's KuiperLLama 实现
 *
 * 计算: output[n] = Σ_k input[k] * weight[n*K+k]_int8 * scale[(n*K+k)/group_size] + bias[n]
 *
 * 数据布局:
 *   input  [K]    fp32
 *   weight [N, K] int8 行优先, 扁平化为 [N*K]
 *   scales [N*K/group_size] fp32, 全局 group 索引
 *   output [N]    fp32
 *
 * 每个 block 计算一个 output[n] (ROW_PER_BLOCK=1)
 */
template <int THREAD_PER_BLOCK, int ROW_PER_BLOCK>
__global__ void matmul_int8_kernel_cuda_fp32(
    const float* __restrict__ input,
    const int8_t* __restrict__ weight,
    const float* __restrict__ scales,
    const float* __restrict__ bias,
    const int32_t group_size,
    float* __restrict__ output,
    int M, int K) {

  __shared__ float sdata[THREAD_PER_BLOCK];
  unsigned int tid = threadIdx.x;

  int start_row = blockIdx.x * ROW_PER_BLOCK;
  int end_row = start_row + ROW_PER_BLOCK;
  if (start_row >= K) return;

  for (int p = start_row; p < end_row; ++p) {
    sdata[tid] = 0.0f;
    for (int i = tid; i < M; i += THREAD_PER_BLOCK) {
      const int weight_idx = p * M + i;
      const int group_idx = weight_idx / group_size;
      sdata[tid] += input[i] * scales[group_idx] * static_cast<float>(weight[weight_idx]);
    }
    __syncthreads();

    using BlockReduce = cub::BlockReduce<float, THREAD_PER_BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp;
    float part_sum = BlockReduce(temp).Sum(sdata[tid]);
    __syncthreads();

    if (tid == 0) {
      float val = part_sum;
      if (bias) val += bias[p];
      output[p] = val;
    }
    __syncthreads();
  }
}

// ═══ GPU 对外接口 ═══
void matmul_int8_kernel_cuda(const tensor::Tensor& input,
                              const tensor::Tensor& weight,
                              const tensor::Tensor& scales,
                              const float* bias,
                              const tensor::Tensor& output,
                              int32_t group_size,
                              void* stream) {
  CHECK(!input.is_empty());
  CHECK(!weight.is_empty());
  CHECK(!scales.is_empty());
  CHECK(!output.is_empty());
  CHECK(weight.get_data_type() == tensor::DataType_t::int8)
      << "Weight must be INT8 for quantized matmul";

  // weight 形状: [N, K] — N=output rows, K=inner dim (=M in kernel)
  const int32_t N = static_cast<int32_t>(weight.get_dim(0));
  const int32_t K_inner = static_cast<int32_t>(weight.get_dim(1));

  CHECK_EQ(static_cast<int32_t>(input.get_dim(input.get_dims_size() - 1)), K_inner)
      << "Input last dim must equal weight inner dim";

  // input: 1D [K] or 2D [M_batch, K]
  int32_t M = K_inner;
  if (input.get_dims_size() == 2)
    M = static_cast<int32_t>(input.get_dim(0)) * K_inner;

  constexpr int THREADS = 128;
  int grid = N;  // one block per output row

  if (stream) {
    matmul_int8_kernel_cuda_fp32<THREADS, 1>
        <<<grid, THREADS, 0, static_cast<cudaStream_t>(stream)>>>(
            static_cast<const float*>(input.get_ptr()),
            static_cast<const int8_t*>(weight.get_ptr()),
            static_cast<const float*>(scales.get_ptr()),
            bias, group_size,
            const_cast<float*>(static_cast<const float*>(output.get_ptr())),
            M, N);
  } else {
    matmul_int8_kernel_cuda_fp32<THREADS, 1>
        <<<grid, THREADS>>>(
            static_cast<const float*>(input.get_ptr()),
            static_cast<const int8_t*>(weight.get_ptr()),
            static_cast<const float*>(scales.get_ptr()),
            bias, group_size,
            const_cast<float*>(static_cast<const float*>(output.get_ptr())),
            M, N);
  }
}

// ═══ CPU 回退 ═══
void matmul_int8_kernel_cpu(const tensor::Tensor& input,
                             const tensor::Tensor& weight,
                             const tensor::Tensor& scales,
                             const float* bias,
                             const tensor::Tensor& output,
                             int32_t group_size,
                             void* stream) {
  (void)stream;

  const int32_t N = static_cast<int32_t>(weight.get_dim(0));
  const int32_t K = static_cast<int32_t>(weight.get_dim(1));

  const float* in = static_cast<const float*>(input.get_ptr());
  const int8_t* w = static_cast<const int8_t*>(weight.get_ptr());
  const float* s = static_cast<const float*>(scales.get_ptr());
  float* out = const_cast<float*>(static_cast<const float*>(output.get_ptr()));

  for (int32_t n = 0; n < N; ++n) {
    float sum = 0.0f;
    for (int32_t k = 0; k < K; ++k) {
      int weight_idx = n * K + k;
      int g = weight_idx / group_size;
      sum += in[k] * s[g] * static_cast<float>(w[weight_idx]);
    }
    if (bias) sum += bias[n];
    out[n] = sum;
  }
}

}  // namespace kernel
