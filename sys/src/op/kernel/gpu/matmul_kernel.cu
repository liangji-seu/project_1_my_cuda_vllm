#include "matmul_kernel.cuh"
#include <glog/logging.h>
#include <cuda_runtime.h>

namespace kernel {

// Simple matmul: each block handles one output element
// Supports:
//   1D input [M]          -> output [K]
//   2D input [batch, M]   -> output [batch, K]
__global__ void matmul_kernel_cuda_fp32(int32_t M, int32_t K, int32_t batch,
                                        const float* input,
                                        const float* weight, float* output, float scale) {
  int32_t block_id = blockIdx.x;
  int32_t total_rows = batch * K;
  if (block_id >= total_rows) return;

  int32_t b = block_id / K;
  int32_t row = block_id % K;

  float sum = 0.0f;
  const float* input_row = input + b * M;
  for (int32_t i = threadIdx.x; i < M; i += blockDim.x) {
    sum += input_row[i] * weight[row * M + i];
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
    output[block_id] = sdata[0] * scale;
  }
}

void matmul_kernel_cuda(const tensor::Tensor& input, const tensor::Tensor& weight,
                        float scale, const tensor::Tensor& output, void* stream) {
  CHECK(!input.is_empty());
  CHECK(!weight.is_empty());
  CHECK(!output.is_empty());

  const int32_t K = static_cast<int32_t>(weight.get_dim(0));  // output dim
  const int32_t M = static_cast<int32_t>(weight.get_dim(1));  // input dim

  // Determine batch dimension from input
  int32_t batch = 1;
  if (input.get_dims_size() == 2) {
    batch = static_cast<int32_t>(input.get_dim(0));
  }
  CHECK_EQ(static_cast<int32_t>(input.get_dim(input.get_dims_size() - 1)), M)
      << "Input last dimension must match weight input dimension";

  size_t block_size = 256;
  size_t grid_size = static_cast<size_t>(batch * K);
  size_t shared_mem = block_size * sizeof(float);

  if (stream) {
    cudaStream_t _stream = static_cast<cudaStream_t>(stream);
    matmul_kernel_cuda_fp32<<<grid_size, block_size, shared_mem, _stream>>>(
        M, K, batch,
        static_cast<const float*>(input.get_ptr()),
        static_cast<const float*>(weight.get_ptr()),
        const_cast<float*>(static_cast<const float*>(output.get_ptr())),
        scale);
  } else {
    matmul_kernel_cuda_fp32<<<grid_size, block_size, shared_mem>>>(
        M, K, batch,
        static_cast<const float*>(input.get_ptr()),
        static_cast<const float*>(weight.get_ptr()),
        const_cast<float*>(static_cast<const float*>(output.get_ptr())),
        scale);
  }
}

}
