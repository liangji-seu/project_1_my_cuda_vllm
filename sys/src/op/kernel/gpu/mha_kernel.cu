#include "mha_kernel.cuh"
#include <glog/logging.h>
#include <cuda_runtime.h>

namespace kernel {

  /**
   * 朴素实现，一个thread负责计算一个q头的注意力输出
   */
__global__ void mha_kernel_cuda_fp32(
    int32_t pos, int32_t head_num, int32_t layer_offset,
    int32_t seq_len, int32_t kv_dim, int32_t kv_mul, int32_t head_size,
    const float* query, const float* key_cache, const float* value_cache,
    float* score, float* output, float scale) {

  //thread的grid全局id，一个grid负责一个q向量的注意力输出计算
  int32_t h = blockDim.x * blockIdx.x + threadIdx.x;
  if (h >= head_num) return;

  int32_t kv_head = h / kv_mul;//几个q头一组
  const float* query_head = query + h * head_size;//偏移到该thread负责处理的头
  float* score_head = score + h * seq_len;//偏移到这个thread的q头针对这个k头的分数位置

  //只计算因果score
  for (int32_t t = 0; t <= pos; ++t) {
    int32_t cache_offset = t * kv_dim + kv_head * head_size;
    const float* key_head = key_cache + layer_offset + cache_offset;//找到pos位置的k头

    //q头和k头做点积
    float dot = 0.0f;
    for (int32_t d = 0; d < head_size; ++d) {
      dot += query_head[d] * key_head[d];
    }
    //得到q头和这个k头的分数
    score_head[t] = dot * scale;
  }

  // Softmax
  float max_score = score_head[0];
  for (int32_t t = 1; t <= pos; ++t) {
    if (score_head[t] > max_score) max_score = score_head[t];
  }
  float sum_exp = 0.0f;
  for (int32_t t = 0; t <= pos; ++t) {
    float exp_val = expf(score_head[t] - max_score);
    score_head[t] = exp_val;
    sum_exp += exp_val;
  }
  for (int32_t t = 0; t <= pos; ++t) {
    score_head[t] /= sum_exp;
  }

  // Weighted sum with V
  float* out_head = output + h * head_size;
  for (int32_t d = 0; d < head_size; ++d) {
    out_head[d] = 0.0f;
  }
  for (int32_t t = 0; t <= pos; ++t) {
    int32_t cache_offset = t * kv_dim + kv_head * head_size;
    const float* value_head = value_cache + layer_offset + cache_offset;
    for (int32_t d = 0; d < head_size; ++d) {
      out_head[d] += score_head[t] * value_head[d];
    }
  }
}

void mha_kernel_cuda(
    int32_t pos, int32_t head_num, int32_t layer_index,
    int32_t seq_len, int32_t kv_dim, int32_t kv_mul, int32_t head_size,
    const tensor::Tensor& mha_out, const tensor::Tensor& query_tensor,
    const tensor::Tensor& score_tensor, const tensor::Tensor& key_cache_tensor,
    const tensor::Tensor& value_cache_tensor, void* stream) {

  int32_t layer_offset = layer_index * seq_len * kv_dim;
  float scale = 1.0f / std::sqrt(static_cast<float>(head_size));//根号d的系数

  //一个thread负责一个头
  size_t block_size = 256;
  size_t grid_size = (static_cast<size_t>(head_num) + block_size - 1) / block_size;

  if (stream) {
    cudaStream_t _stream = static_cast<cudaStream_t>(stream);
    mha_kernel_cuda_fp32<<<grid_size, block_size, 0, _stream>>>(
        pos, head_num, layer_offset, seq_len, kv_dim, kv_mul, head_size,
        static_cast<const float*>(query_tensor.get_ptr()),
        static_cast<const float*>(key_cache_tensor.get_ptr()),
        static_cast<const float*>(value_cache_tensor.get_ptr()),
        const_cast<float*>(static_cast<const float*>(score_tensor.get_ptr())),
        const_cast<float*>(static_cast<const float*>(mha_out.get_ptr())),
        scale);
  } else {
    mha_kernel_cuda_fp32<<<grid_size, block_size>>>(
        pos, head_num, layer_offset, seq_len, kv_dim, kv_mul, head_size,
        static_cast<const float*>(query_tensor.get_ptr()),
        static_cast<const float*>(key_cache_tensor.get_ptr()),
        static_cast<const float*>(value_cache_tensor.get_ptr()),
        const_cast<float*>(static_cast<const float*>(score_tensor.get_ptr())),
        const_cast<float*>(static_cast<const float*>(mha_out.get_ptr())),
        scale);
  }
}

}
