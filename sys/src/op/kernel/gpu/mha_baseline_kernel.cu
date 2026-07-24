#include "mha_baseline_kernel.cuh"
#include <glog/logging.h>
#include <cuda_runtime.h>
#include <cfloat>

namespace kernel {

/**
 * Simple 1-thread-per-head MHA — baseline (no optimization).
 * All loops are sequential within each thread.
 */
__global__ void mha_baseline_kernel(
    int32_t pos, int32_t head_num, int32_t layer_offset,
    int32_t seq_len, int32_t kv_dim, int32_t kv_mul, int32_t head_size,
    const float* query, const float* key_cache, const float* value_cache,
    float* score, float* output) {

  int head = blockIdx.x;
  if (head >= head_num) return;

  const float* query_head = query + head * head_size;
  float* score_head = score + head * seq_len;
  float* output_head = output + head * head_size;
  int head_offset = (head / kv_mul) * head_size;

  float scale = rsqrtf(static_cast<float>(head_size));

  // 1. QK dot product — sequential, no vectorization
  for (int t = 0; t <= pos; t++) {
    const float* key_head = key_cache + layer_offset + t * kv_dim + head_offset;
    float s = 0.0f;
    for (int d = 0; d < head_size; d++) {
      s += query_head[d] * key_head[d];
    }
    score_head[t] = s * scale;
  }

  // 2. Softmax — sequential
  float max_val = -FLT_MAX;
  for (int t = 0; t <= pos; t++) {
    if (score_head[t] > max_val) max_val = score_head[t];
  }
  float sum = 0.0f;
  for (int t = 0; t <= pos; t++) {
    score_head[t] = expf(score_head[t] - max_val);
    sum += score_head[t];
  }
  for (int t = 0; t <= pos; t++) {
    score_head[t] /= sum;
  }

  // 3. V weighted sum — sequential
  for (int d = 0; d < head_size; d++) {
    float val = 0.0f;
    for (int t = 0; t <= pos; t++) {
      const float* value_head = value_cache + layer_offset + t * kv_dim + head_offset;
      val += score_head[t] * value_head[d];
    }
    output_head[d] = val;
  }
}

void mha_baseline_kernel_cuda(
    int32_t pos, int32_t head_num, int32_t layer_index,
    int32_t seq_len, int32_t kv_dim, int32_t kv_mul, int32_t head_size,
    const tensor::Tensor& mha_out, const tensor::Tensor& query_tensor,
    const tensor::Tensor& score_tensor, const tensor::Tensor& key_cache_tensor,
    const tensor::Tensor& value_cache_tensor, void* stream) {

  int32_t layer_offset = layer_index * seq_len * kv_dim;
  int grid = head_num;  // one block per head, 1 thread per block

  if (stream) {
    mha_baseline_kernel<<<grid, 1, 0, static_cast<cudaStream_t>(stream)>>>(
        pos, head_num, layer_offset, seq_len, kv_dim, kv_mul, head_size,
        static_cast<const float*>(query_tensor.get_ptr()),
        static_cast<const float*>(key_cache_tensor.get_ptr()),
        static_cast<const float*>(value_cache_tensor.get_ptr()),
        const_cast<float*>(static_cast<const float*>(score_tensor.get_ptr())),
        const_cast<float*>(static_cast<const float*>(mha_out.get_ptr())));
  } else {
    mha_baseline_kernel<<<grid, 1>>>(
        pos, head_num, layer_offset, seq_len, kv_dim, kv_mul, head_size,
        static_cast<const float*>(query_tensor.get_ptr()),
        static_cast<const float*>(key_cache_tensor.get_ptr()),
        static_cast<const float*>(value_cache_tensor.get_ptr()),
        const_cast<float*>(static_cast<const float*>(score_tensor.get_ptr())),
        const_cast<float*>(static_cast<const float*>(mha_out.get_ptr())));
  }
}

}  // namespace kernel
