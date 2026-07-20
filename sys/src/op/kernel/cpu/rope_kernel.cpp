#include "rope_kernel.h"
#include <glog/logging.h>
#include <cmath>
#include <cstring>

namespace kernel {

#if defined(QWEN3_SUPPORT)

void sin_cos_cache_calc_cpu(int head_size, int max_seq_len, float* sin_cache, float* cos_cache) {
  // LLaMA-style: compute [max_seq_len * head_size] with values repeated
  // (matches HuggingFace Qwen2/Qwen3 RoPE convention)
  for (int pos = 0; pos < max_seq_len; ++pos) {
    for (int i = 0; i < head_size; ++i) {
      // inv_freq for the i-th dimension pair
      int pair_idx = i % (head_size / 2);
      float freq =
          1.0f / std::pow(1000000.0f, 2.0f * static_cast<float>(pair_idx) / static_cast<float>(head_size));
      float val = static_cast<float>(pos) * freq;
      float fcr = cosf(val);
      float fci = sinf(val);
      *(sin_cache + pos * head_size + i) = fci;
      *(cos_cache + pos * head_size + i) = fcr;
    }
  }
}

void rope_kernel_cpu(int32_t dim, int32_t kv_dim, int32_t head_size,
                     const tensor::Tensor& input_q, const tensor::Tensor& input_k,
                     const tensor::Tensor& input_pos, const tensor::Tensor& sin_cache,
                     const tensor::Tensor& cos_cache, void* stream) {
  (void)stream;
  // LLaMA-style RoPE: rotate (i, i + half_dim) pairs within each head
  const int32_t pos = *static_cast<const int32_t*>(input_pos.get_ptr());
  const int32_t half_dim = head_size / 2;

  float* q_ptr = const_cast<float*>(static_cast<const float*>(input_q.get_ptr()));
  float* k_ptr = const_cast<float*>(static_cast<const float*>(input_k.get_ptr()));
  const float* sin_ptr = static_cast<const float*>(sin_cache.get_ptr());
  const float* cos_ptr = static_cast<const float*>(cos_cache.get_ptr());

  // Iterate over all heads in q_dim
  int32_t q_heads = dim / head_size;
  int32_t kv_heads = kv_dim / head_size;

  for (int32_t h = 0; h < q_heads; ++h) {
    int32_t head_off = h * head_size;
    // Apply RoPE to this Q head
    for (int32_t i = 0; i < half_dim; ++i) {
      float fci = sin_ptr[pos * head_size + i];
      float fcr = cos_ptr[pos * head_size + i];

      float* q = q_ptr + head_off;
      float q0 = q[i];
      float q1 = q[i + half_dim];
      q[i] = q0 * fcr - q1 * fci;
      q[i + half_dim] = q0 * fci + q1 * fcr;
    }
    // Apply to K head if this Q head maps to a KV head
    if (h < kv_heads) {
      int32_t k_off = h * head_size;
      for (int32_t i = 0; i < half_dim; ++i) {
        float fci = sin_ptr[pos * head_size + i];
        float fcr = cos_ptr[pos * head_size + i];

        float* k = k_ptr + k_off;
        float k0 = k[i];
        float k1 = k[i + half_dim];
        k[i] = k0 * fcr - k1 * fci;
        k[i + half_dim] = k0 * fci + k1 * fcr;
      }
    }
  }
}

#else  // Default (LLAMA3_SUPPORT or generic)

void sin_cos_cache_calc_cpu(int head_size, int max_seq_len, float* sin_cache, float* cos_cache) {
  for (int pos = 0; pos < max_seq_len; ++pos) {
    for (int i = 0; i < head_size; ++i) {
      int pair_idx = i % (head_size / 2);
      float freq =
          1.0f / std::pow(10000.0f, 2.0f * static_cast<float>(pair_idx) / static_cast<float>(head_size));
      float val = static_cast<float>(pos) * freq;
      float fcr = cosf(val);
      float fci = sinf(val);
      *(sin_cache + pos * head_size + i) = fci;
      *(cos_cache + pos * head_size + i) = fcr;
    }
  }
}

void rope_kernel_cpu(int32_t dim, int32_t kv_dim, int32_t head_size,
                     const tensor::Tensor& input_q, const tensor::Tensor& input_k,
                     const tensor::Tensor& input_pos, const tensor::Tensor& sin_cache,
                     const tensor::Tensor& cos_cache, void* stream) {
  (void)stream;
  const int32_t pos = *static_cast<const int32_t*>(input_pos.get_ptr());
  const int32_t half_dim = head_size / 2;

  float* q_ptr = const_cast<float*>(static_cast<const float*>(input_q.get_ptr()));
  float* k_ptr = const_cast<float*>(static_cast<const float*>(input_k.get_ptr()));
  const float* sin_ptr = static_cast<const float*>(sin_cache.get_ptr());
  const float* cos_ptr = static_cast<const float*>(cos_cache.get_ptr());

  int32_t q_heads = dim / head_size;
  int32_t kv_heads = kv_dim / head_size;

  for (int32_t h = 0; h < q_heads; ++h) {
    int32_t head_off = h * head_size;
    for (int32_t i = 0; i < half_dim; ++i) {
      float fci = sin_ptr[pos * head_size + i];
      float fcr = cos_ptr[pos * head_size + i];

      float* q = q_ptr + head_off;
      float q0 = q[i];
      float q1 = q[i + half_dim];
      q[i] = q0 * fcr - q1 * fci;
      q[i + half_dim] = q0 * fci + q1 * fcr;
    }
    if (h < kv_heads) {
      int32_t k_off = h * head_size;
      for (int32_t i = 0; i < half_dim; ++i) {
        float fci = sin_ptr[pos * head_size + i];
        float fcr = cos_ptr[pos * head_size + i];

        float* k = k_ptr + k_off;
        float k0 = k[i];
        float k1 = k[i + half_dim];
        k[i] = k0 * fcr - k1 * fci;
        k[i + half_dim] = k0 * fci + k1 * fcr;
      }
    }
  }
}

#endif

}  // namespace kernel
