#include "matmul_kernel.cuh"
#include <glog/logging.h>
#include <cuda_runtime.h>

namespace kernel {

// Simple matmul: each block handles one row of weight
__global__ void matmul_kernel_cuda_fp32(int32_t M, int32_t K, const float* input,
                                        const float* weight, float* output, float scale) {
  int32_t row = blockIdx.x;
  if (row >= K) return;

  float sum = 0.0f;
  for (int32_t i = threadIdx.x; i < M; i += blockDim.x) {
    sum += input[i] * weight[row * M + i];
  }

  // Block-level reduction
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
    output[row] = sdata[0] * scale;
  }
}

void matmul_kernel_cuda(const tensor::Tensor& input, const tensor::Tensor& weight,
                        float scale, const tensor::Tensor& output, void* stream) {
  CHECK(!input.is_empty());
  CHECK(!weight.is_empty());
  CHECK(!output.is_empty());

  const int32_t K = static_cast<int32_t>(weight.get_dim(0));  // output dim
  const int32_t M = static_cast<int32_t>(weight.get_dim(1));  // input dim

  size_t block_size = 256;
  size_t grid_size = static_cast<size_t>(K);
  size_t shared_mem = block_size * sizeof(float);

  if (stream) {
    cudaStream_t _stream = static_cast<cudaStream_t>(stream);
    matmul_kernel_cuda_fp32<<<grid_size, block_size, shared_mem, _stream>>>(
        M, K,
        static_cast<const float*>(input.get_ptr()),
        static_cast<const float*>(weight.get_ptr()),
        const_cast<float*>(static_cast<const float*>(output.get_ptr())),
        scale);
  } else {
    matmul_kernel_cuda_fp32<<<grid_size, block_size, shared_mem>>>(
        M, K,
        static_cast<const float*>(input.get_ptr()),
        static_cast<const float*>(weight.get_ptr()),
        const_cast<float*>(static_cast<const float*>(output.get_ptr())),
        scale);
  }
}

}
