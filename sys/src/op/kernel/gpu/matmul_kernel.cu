#include "matmul_kernel.cuh"
#include <glog/logging.h>
#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>

namespace kernel {

/**
 * float4向量化 matmul — 每个block算一个输出元素
 * 计算: output[m, n] = scale * Σ_k input[m, k] * weight[n, k] + bias[n]
 * weight [N, K] 行优先, input 2D [M, K] 或 1D [K]
 */
template <int THREAD_PER_BLOCK, int ROW_PER_BLOCK>
__global__ void matmul_kernel_cuda_fp32(const float* __restrict__ input,
                                         const float* __restrict__ weight,
                                         const float* __restrict__ bias,
                                         float* __restrict__ output,
                                         float scale, int M, int N, int K) {
  const unsigned int tid = threadIdx.x;
  const unsigned int bid = blockIdx.x;

  // ── block 负责 output[m, n]: m = bid / N, n = bid % N ──
  int m = bid / N;
  int n = bid % N;
  if (m >= M || n >= N) return;

  const float* input_row = input + m * K;

  // ── float4 对齐部分与余数 ──
  constexpr int pack_size = 4;
  const int pack_num = K / pack_size;
  const int pack_off = pack_size * pack_num;

  __shared__ float sdata[THREAD_PER_BLOCK];

  sdata[tid] = 0.0f;

  // float4 向量化点积 (4元素一批)
  const float4* input_float4_ptr  = reinterpret_cast<const float4*>(input_row);
  const float4* weight_float4_ptr = reinterpret_cast<const float4*>(weight + n * K);

#pragma unroll
  for (int i = tid; i < pack_num; i += blockDim.x) {
    float4 in  = *(input_float4_ptr  + i);
    float4 wgt = *(weight_float4_ptr + i);
    sdata[tid] += in.x * wgt.x + in.y * wgt.y + in.z * wgt.z + in.w * wgt.w;
  }

  // 处理余数
  for (int i = pack_off + tid; i < K; i += blockDim.x) {
    sdata[tid] += input_row[i] * weight[n * K + i];
  }

  __syncthreads();

  // cub::BlockReduce — warp shuffle 高效规约
  using BlockReduce = cub::BlockReduce<float, THREAD_PER_BLOCK>;
  __shared__ typename BlockReduce::TempStorage temp;
  float sum = BlockReduce(temp).Sum(sdata[tid]);

  if (tid == 0) {
    float val = sum * scale;
    if (bias) val += bias[n];
    output[m * N + n] = val;
  }
}

// ═══ 对外接口 ═══
void matmul_kernel_cuda(const tensor::Tensor& input,
                        const tensor::Tensor& weight,
                        const float* bias, float scale,
                        const tensor::Tensor& output,
                        void* stream) {
  CHECK(!input.is_empty());
  CHECK(!weight.is_empty());
  CHECK(!output.is_empty());

  // weight 形状: [N, K] — N=输出维度, K=输入维度
  const int32_t N = static_cast<int32_t>(weight.get_dim(0));
  const int32_t K = static_cast<int32_t>(weight.get_dim(1));

  // input: 1D [K] 或 2D [M, K]
  int32_t M = 1;
  if (input.get_dims_size() == 2)
    M = static_cast<int32_t>(input.get_dim(0));

  CHECK_EQ(static_cast<int32_t>(input.get_dim(input.get_dims_size() - 1)), K)
      << "Input last dim must equal weight inner dim";

  constexpr int THREADS = 128;
  int grid = M * N;  // 每个输出元素一个 block

  if (stream) {
    matmul_kernel_cuda_fp32<THREADS, 1>
        <<<grid, THREADS, 0, static_cast<cudaStream_t>(stream)>>>(
            static_cast<const float*>(input.get_ptr()),
            static_cast<const float*>(weight.get_ptr()),
            bias, const_cast<float*>(static_cast<const float*>(output.get_ptr())),
            scale, M, N, K);
  } else {
    matmul_kernel_cuda_fp32<THREADS, 1>
        <<<grid, THREADS>>>(
            static_cast<const float*>(input.get_ptr()),
            static_cast<const float*>(weight.get_ptr()),
            bias, const_cast<float*>(static_cast<const float*>(output.get_ptr())),
            scale, M, N, K);
  }
}

}  // namespace kernel
