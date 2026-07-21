#include "rmsnorm_kernel.cuh"
#include <glog/logging.h>
#include <cuda_runtime.h>

namespace kernel {

__global__ void rmsnorm_kernel_cuda_fp32(int32_t dim, const float* input,
                                         const float* weight, float* output,
                                         float eps) {
  extern __shared__ float shared_buf[];
  float* s_sum_sq = shared_buf;

  int32_t tid = threadIdx.x;
  float sq = 0.0f;

  // grid-stride loop: each thread reads multiple elements
  // e.g. dim=4096, block_size=1024 → each thread reads 4 elements
  for (int32_t i = tid; i < dim; i += blockDim.x) {
    float val = input[i];
    sq += val * val;
  }

  // store partial sum to shared memory
  s_sum_sq[tid] = sq;
  __syncthreads();

  // tree reduction within the block
  for (int32_t offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_sum_sq[tid] += s_sum_sq[tid + offset];
    }
    __syncthreads();
  }

  // s_sum_sq[0] now holds sum(x_i^2) for the entire vector
  float rsqrt = rsqrtf(s_sum_sq[0] / static_cast<float>(dim) + eps);

  // grid-stride loop: each thread writes back all its elements
  for (int32_t i = tid; i < dim; i += blockDim.x) {
    output[i] = weight[i] * rsqrt * input[i];
  }
}

void rmsnorm_kernel_cuda(const tensor::Tensor& input, const tensor::Tensor& weight,
                         const tensor::Tensor& output, void* stream) {
  CHECK(!input.is_empty());
  CHECK(!weight.is_empty());
  CHECK(!output.is_empty());

  int32_t dim = static_cast<int32_t>(input.get_size());
  const float eps = 1e-6f;  // Qwen2.5 rms_norm_eps = 1e-6

  // block_size: power of 2 >= dim/stride, capped at 1024
  // For dim=4096, 1024 threads each process 4 elements
  size_t block_size = 32;
  while (block_size * 2 <= static_cast<size_t>(dim) && block_size * 2 <= 1024) {
    block_size <<= 1;
  }

  // shared memory: one float per thread for partial sum
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
