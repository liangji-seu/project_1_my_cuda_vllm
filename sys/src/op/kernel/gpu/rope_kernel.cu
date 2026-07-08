#include "rope_kernel.cuh"
#include <glog/logging.h>
#include <cuda_runtime.h>

namespace kernel {

__global__ void rope_kernel_cuda_fp32(int32_t pos, int32_t dim, int32_t kv_dim,
                                       int32_t head_size, float* input_q, float* input_k,
                                       const float* sin_cache, const float* cos_cache) {
  int32_t idx = (threadIdx.x + blockDim.x * blockIdx.x) * 2;
  if (idx >= dim) return;

  int32_t head_dim = idx % head_size;
  float fci = sin_cache[pos * head_size + head_dim];
  float fcr = cos_cache[pos * head_size + head_dim];

  // Rotate Q
  float q0 = input_q[idx];
  float q1 = input_q[idx + 1];
  input_q[idx]     = q0 * fcr - q1 * fci;
  input_q[idx + 1] = q0 * fci + q1 * fcr;

  // Rotate K only if within kv_dim
  if (idx < kv_dim) {
    float k0 = input_k[idx];
    float k1 = input_k[idx + 1];
    input_k[idx]     = k0 * fcr - k1 * fci;
    input_k[idx + 1] = k0 * fci + k1 * fcr;
  }
}

void rope_kernel_cuda(int32_t dim, int32_t kv_dim, int32_t head_size,
                      const tensor::Tensor& input_q, const tensor::Tensor& input_k,
                      const tensor::Tensor& input_pos, const tensor::Tensor& sin_cache,
                      const tensor::Tensor& cos_cache, void* stream) {
  const int32_t pos = *static_cast<const int32_t*>(input_pos.get_ptr());

  size_t threads = 128;
  size_t blocks = (static_cast<size_t>(dim) / 2 + threads - 1) / threads;

  if (stream) {
    cudaStream_t _stream = static_cast<cudaStream_t>(stream);
    rope_kernel_cuda_fp32<<<blocks, threads, 0, _stream>>>(
        pos, dim, kv_dim, head_size,
        const_cast<float*>(static_cast<const float*>(input_q.get_ptr())),
        const_cast<float*>(static_cast<const float*>(input_k.get_ptr())),
        static_cast<const float*>(sin_cache.get_ptr()),
        static_cast<const float*>(cos_cache.get_ptr()));
  } else {
    rope_kernel_cuda_fp32<<<blocks, threads>>>(
        pos, dim, kv_dim, head_size,
        const_cast<float*>(static_cast<const float*>(input_q.get_ptr())),
        const_cast<float*>(static_cast<const float*>(input_k.get_ptr())),
        static_cast<const float*>(sin_cache.get_ptr()),
        static_cast<const float*>(cos_cache.get_ptr()));
  }
}

}
