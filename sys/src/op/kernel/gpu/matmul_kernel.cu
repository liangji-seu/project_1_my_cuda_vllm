#include "matmul_kernel.cuh"
#include <glog/logging.h>
#include <cuda_runtime.h>

namespace kernel {

// 矩阵乘: input[M, K] × weight[K, N] + bias[N] → output[M, N]
// 每个 block 负责 output 的一个元素
__global__ void matmul_kernel_cuda_fp32(int32_t M, int32_t K, int32_t N,
                                        const float* input,
                                        const float* weight,
                                        const float* bias,
                                        float* output, float scale) {
  int32_t block_id = blockIdx.x;
  int32_t total_elems = M * N;
  if (block_id >= total_elems) return;

  // output 的行号 m 和列号 n
  int32_t m = block_id / N;
  int32_t n = block_id % N;

  // input 第 m 行: input[m * K + k]
  const float* input_row = input + m * K;


  //规约计算 乘加
  float sum = 0.0f;
  for (int32_t k = threadIdx.x; k < K; k += blockDim.x) {
    sum += input_row[k] * weight[n * K + k];
  }

  // block 内归约
  extern __shared__ float sdata[];
  sdata[threadIdx.x] = sum;
  __syncthreads();

  for (int32_t s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      sdata[threadIdx.x] += sdata[threadIdx.x + s];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    float val = sdata[0] * scale;
    // Add bias if present
    if (bias) val += bias[n];
    output[block_id] = val;
  }
}

void matmul_kernel_cuda(const tensor::Tensor& input, const tensor::Tensor& weight,
                        const float* bias, float scale,
                        const tensor::Tensor& output, void* stream) {
  CHECK(!input.is_empty());
  CHECK(!weight.is_empty());
  CHECK(!output.is_empty());

  // weight 形状: [N, K]
  const int32_t N = static_cast<int32_t>(weight.get_dim(0));  // 输出维度
  const int32_t K = static_cast<int32_t>(weight.get_dim(1));  // 输入特征维度（内积维度）

  // input 形状: [M=1, K] 或 [M, K]
  int32_t M = 1;
  if (input.get_dims_size() == 2) {
    M = static_cast<int32_t>(input.get_dim(0));
  }
  CHECK_EQ(static_cast<int32_t>(input.get_dim(input.get_dims_size() - 1)), K)
      << "Input last dimension must match weight inner dimension K";

  size_t block_size = 256;
  size_t grid_size = static_cast<size_t>(M * N);//一个block算一个元素
  size_t shared_mem = block_size * sizeof(float);

  if (stream) {
    cudaStream_t _stream = static_cast<cudaStream_t>(stream);
    matmul_kernel_cuda_fp32<<<grid_size, block_size, shared_mem, _stream>>>(
        M, K, N,
        static_cast<const float*>(input.get_ptr()),
        static_cast<const float*>(weight.get_ptr()),
        bias,
        const_cast<float*>(static_cast<const float*>(output.get_ptr())),
        scale);
  } else {
    matmul_kernel_cuda_fp32<<<grid_size, block_size, shared_mem>>>(
        M, K, N,
        static_cast<const float*>(input.get_ptr()),
        static_cast<const float*>(weight.get_ptr()),
        bias,
        const_cast<float*>(static_cast<const float*>(output.get_ptr())),
        scale);
  }
}

}
