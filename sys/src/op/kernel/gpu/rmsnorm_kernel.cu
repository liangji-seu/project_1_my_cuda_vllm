#include "rmsnorm_kernel.cuh"
#include <glog/logging.h>
#include <cuda_runtime.h>

namespace kernel {

__global__ void rmsnorm_kernel_cuda_fp32(int32_t dim, const float* input,
                                         const float* weight, float* output,
                                         float eps) {
  // Each block handles one RMSNorm vector
  extern __shared__ float shared_buf[];
  float* s_mean_sq = shared_buf;

  int32_t tid = threadIdx.x;
  float val = 0.0f;
  if (tid < dim) {
    val = input[tid];
  }
  float sq = val * val;

  // Block-level reduction for mean(x^2)
  for (int32_t offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    s_mean_sq[tid] = sq;
    __syncthreads();
    if (tid < offset && (tid + offset) < dim) {
      sq += s_mean_sq[tid + offset];
    }
  }

  float rsqrt = rsqrtf(sq / static_cast<float>(dim) + eps);

  if (tid < dim) {
    output[tid] = weight[tid] * rsqrt * val;
  }
}

void rmsnorm_kernel_cuda(const tensor::Tensor& input, const tensor::Tensor& weight,
                         const tensor::Tensor& output, void* stream) {
  CHECK(!input.is_empty());
  CHECK(!weight.is_empty());
  CHECK(!output.is_empty());

  int32_t dim = static_cast<int32_t>(input.get_size());
  const float eps = 1e-5f;

  // block_size: power of 2 >= dim, max 1024
  size_t block_size = 32;
  while (block_size < static_cast<size_t>(dim)) {
    block_size <<= 1;
  }
  if (block_size > 1024) block_size = 1024;

  size_t shared_mem_bytes = block_size * sizeof(float);

  if (stream) {
    cudaStream_t _stream = static_cast<cudaStream_t>(stream);
    rmsnorm_kernel_cuda_fp32<<<1, block_size, shared_mem_bytes, _stream>>>(
        dim,
        static_cast<const float*>(input.get_ptr()),
        static_cast<const float*>(weight.get_ptr()),
        const_cast<float*>(static_cast<const float*>(output.get_ptr())),
        eps);
  } else {
    rmsnorm_kernel_cuda_fp32<<<1, block_size, shared_mem_bytes>>>(
        dim,
        static_cast<const float*>(input.get_ptr()),
        static_cast<const float*>(weight.get_ptr()),
        const_cast<float*>(static_cast<const float*>(output.get_ptr())),
        eps);
  }
}

}
