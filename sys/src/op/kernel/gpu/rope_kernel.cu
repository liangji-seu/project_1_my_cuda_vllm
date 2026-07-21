#include "rope_kernel.cuh"
#include <glog/logging.h>
#include <cuda_runtime.h>

namespace kernel {

// LLaMA-style RoPE: rotate (i, i + head_size/2) for each head.
// Matches HuggingFace Qwen2/Qwen3 apply_rotary_pos_emb + rotate_half.
//
// Each thread handles one (head, pair_idx) combination.
__global__ void rope_kernel_cuda_fp32(int32_t pos, int32_t dim, int32_t kv_dim,
                                       int32_t head_size,
                                       float* input_q, float* input_k,
                                       const float* sin_cache,
                                       const float* cos_cache) {
  int32_t half_dim = head_size / 2;
  int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  int32_t total_q_pairs = (dim / head_size) * half_dim;
  if (tid >= total_q_pairs) return;

  int32_t head_idx = tid / half_dim;
  int32_t pair_idx = tid % half_dim;
  int32_t head_off = head_idx * head_size;

  // Cache index: each (pos, pair_idx) has a unique cos/sin value
  float fci = sin_cache[pos * head_size + pair_idx];
  float fcr = cos_cache[pos * head_size + pair_idx];

  // Rotate Q: (q[pair_idx], q[pair_idx + half_dim])
  float q0 = input_q[head_off + pair_idx];
  float q1 = input_q[head_off + pair_idx + half_dim];
  input_q[head_off + pair_idx] = q0 * fcr - q1 * fci;
  input_q[head_off + pair_idx + half_dim] = q0 * fci + q1 * fcr;

  // Rotate K only for KV heads (first kv_dim/head_size heads)
  int32_t kv_heads = kv_dim / head_size;
  if (head_idx < kv_heads) {
    float k0 = input_k[head_off + pair_idx];
    float k1 = input_k[head_off + pair_idx + half_dim];
    input_k[head_off + pair_idx] = k0 * fcr - k1 * fci;
    input_k[head_off + pair_idx + half_dim] = k0 * fci + k1 * fcr;
  }
}

void rope_kernel_cuda(int32_t dim, int32_t kv_dim, int32_t head_size,
                      const tensor::Tensor& input_q, const tensor::Tensor& input_k,
                      const tensor::Tensor& input_pos, const tensor::Tensor& sin_cache,
                      const tensor::Tensor& cos_cache, void* stream) {
  const int32_t pos = *static_cast<const int32_t*>(input_pos.get_ptr());

  int32_t half_dim = head_size / 2;
  int32_t total_q_pairs = (dim / head_size) * half_dim;

  size_t threads = 128;
  size_t blocks = (static_cast<size_t>(total_q_pairs) + threads - 1) / threads;

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

}  // namespace kernel
