#include "mha_kernel.h"
#include <glog/logging.h>
#include <cmath>
#include <cstring>

namespace kernel {

void mha_kernel_cpu(
    int32_t pos, int32_t head_num, int32_t layer_index,
    int32_t seq_len, int32_t kv_dim, int32_t kv_mul, int32_t head_size,
    const tensor::Tensor& mha_out, const tensor::Tensor& query_tensor,
    const tensor::Tensor& score_tensor, const tensor::Tensor& key_cache_tensor,
    const tensor::Tensor& value_cache_tensor, void* stream) {
  (void)stream;

  int32_t layer_offset = layer_index * seq_len * kv_dim;
  float scale = 1.0f / std::sqrt(static_cast<float>(head_size));

  const float* query_ptr = static_cast<const float*>(query_tensor.get_ptr());
  const float* key_ptr = static_cast<const float*>(key_cache_tensor.get_ptr());
  const float* value_ptr = static_cast<const float*>(value_cache_tensor.get_ptr());
  float* mha_out_ptr = const_cast<float*>(static_cast<const float*>(mha_out.get_ptr()));
  float* score_ptr = const_cast<float*>(static_cast<const float*>(score_tensor.get_ptr()));

  for (int32_t h = 0; h < head_num; ++h) {
    const float* query_head = query_ptr + h * head_size;
    float* score_head = score_ptr + h * seq_len;
    int32_t kv_head = h / kv_mul;

    // Step 1: Q * K^T / sqrt(d_k)
    for (int32_t t = 0; t <= pos; ++t) {
      int32_t cache_offset = t * kv_dim + kv_head * head_size;
      const float* key_head = key_ptr + layer_offset + cache_offset;

      float dot = 0.0f;
      for (int32_t d = 0; d < head_size; ++d) {
        dot += query_head[d] * key_head[d];
      }
      score_head[t] = dot * scale;
    }

    // Step 2: Softmax
    float max_score = score_head[0];
    for (int32_t t = 1; t <= pos; ++t) {
      if (score_head[t] > max_score) max_score = score_head[t];
    }
    float sum_exp = 0.0f;
    for (int32_t t = 0; t <= pos; ++t) {
      score_head[t] = std::exp(score_head[t] - max_score);
      sum_exp += score_head[t];
    }
    for (int32_t t = 0; t <= pos; ++t) {
      score_head[t] /= sum_exp;
    }

    // Step 3: Weighted sum with V
    float* out_head = mha_out_ptr + h * head_size;
    std::memset(out_head, 0, head_size * sizeof(float));
    for (int32_t t = 0; t <= pos; ++t) {
      int32_t cache_offset = t * kv_dim + kv_head * head_size;
      const float* value_head = value_ptr + layer_offset + cache_offset;
      for (int32_t d = 0; d < head_size; ++d) {
        out_head[d] += score_head[t] * value_head[d];
      }
    }
  }
}

}
