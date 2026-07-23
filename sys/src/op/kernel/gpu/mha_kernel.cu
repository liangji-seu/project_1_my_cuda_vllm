#include "mha_kernel.cuh"
#include <glog/logging.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <cub/cub.cuh>

namespace kernel {

constexpr int MHA_THREADS = 256;

/**
 * Block级别 in-place softmax (数值稳定)
 * 256 线程协作完成, 使用 cub::BlockReduce
 */
__device__ void softmax_block(float* __restrict__ x, int size) {
  int tid = threadIdx.x;

  // 找最大值 (数值稳定)
  float max_val = tid < size ? x[tid] : -FLT_MAX;
  for (int i = tid + blockDim.x; i < size; i += blockDim.x) {
    if (x[i] > max_val) max_val = x[i];
  }

  using BlockReduce = cub::BlockReduce<float, MHA_THREADS>;
  __shared__ BlockReduce::TempStorage temp;
  __shared__ float s_val;
  max_val = BlockReduce(temp).Reduce(max_val, cub::Max());
  if (tid == 0) s_val = max_val;
  __syncthreads();
  max_val = s_val;

  // exp + 求和
  float sum = 0.0f;
  for (int i = tid; i < size; i += blockDim.x) {
    x[i] = expf(x[i] - max_val);
    sum += x[i];
  }
  sum = BlockReduce(temp).Sum(sum);
  if (tid == 0) s_val = sum;
  __syncthreads();
  sum = s_val;

  // 归一化
  for (int i = tid; i < size; i += blockDim.x) {
    x[i] /= sum;
  }
}

/**
 * 优化版 MHA kernel — 一个 block 负责一个 head
 *
 * 优化点:
 *   1. query 预加载到 shared memory (256 线程协作, 只读一次)
 *   2. QK 点积: float4 向量化 + grid-stride 分布到所有线程
 *   3. Softmax: cub::BlockReduce 高效规约
 *   4. V 加权求和: grid-stride 分布
 *
 * 与朴素实现的对比:
 *   原: 1 thread/head, 所有K/V从global memory逐标量读取
 *   新: 256 threads/head, query在shared memory, float4向量化K读取
 */
__global__ void mha_kernel_cuda_fp32(
    int32_t pos, int32_t head_num, int32_t layer_offset,
    int32_t seq_len, int32_t kv_dim, int32_t kv_mul, int32_t head_size,
    const float* query, const float* key_cache, const float* value_cache,
    float* score, float* output) {

  int head = blockIdx.x;
  if (head >= head_num) return;

  // ── query 预加载到 shared memory ──
  extern __shared__ float s_query[];
  const float* query_head = query + head * head_size;
  for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
    s_query[i] = query_head[i];
  }
  __syncthreads();

  // ── QK 点积 (分数计算) ──
  float scale = rsqrtf(static_cast<float>(head_size));
  float* score_head = score + head * seq_len;
  int head_offset = (head / kv_mul) * head_size;  // GQA: 多个Q头共享一个KV头

  for (int t = threadIdx.x; t <= pos; t += blockDim.x) {
    const float* key_head = key_cache + layer_offset + t * kv_dim + head_offset;

    float s = 0.0f;
    // float4 向量化点积
    for (int i = 0; i < head_size; i += 4) {
      float4 k = *reinterpret_cast<const float4*>(key_head + i);
      float4 q = *reinterpret_cast<const float4*>(s_query + i);
      s += k.x * q.x + k.y * q.y + k.z * q.z + k.w * q.w;
    }
    score_head[t] = s * scale;
  }
  __syncthreads();

  // ── Softmax (in-place) ──
  softmax_block(score_head, pos + 1);
  __syncthreads();

  // ── V 加权求和 ──
  float* output_head = output + head * head_size;
  for (int d = threadIdx.x; d < head_size; d += blockDim.x) {
    float val = 0.0f;
    for (int t = 0; t <= pos; t++) {
      const float* value_head = value_cache + layer_offset + t * kv_dim + head_offset;
      val += score_head[t] * value_head[d];
    }
    output_head[d] = val;
  }
}

// ═══ 对外接口 ═══
void mha_kernel_cuda(
    int32_t pos, int32_t head_num, int32_t layer_index,
    int32_t seq_len, int32_t kv_dim, int32_t kv_mul, int32_t head_size,
    const tensor::Tensor& mha_out, const tensor::Tensor& query_tensor,
    const tensor::Tensor& score_tensor, const tensor::Tensor& key_cache_tensor,
    const tensor::Tensor& value_cache_tensor, void* stream) {

  int32_t layer_offset = layer_index * seq_len * kv_dim;

  // 每个 head 一个 block, 256 线程协作
  int grid = head_num;
  int block = MHA_THREADS;
  size_t shared_mem = head_size * sizeof(float);  // query 缓存

  if (stream) {
    mha_kernel_cuda_fp32<<<grid, block, shared_mem, static_cast<cudaStream_t>(stream)>>>(
        pos, head_num, layer_offset, seq_len, kv_dim, kv_mul, head_size,
        static_cast<const float*>(query_tensor.get_ptr()),
        static_cast<const float*>(key_cache_tensor.get_ptr()),
        static_cast<const float*>(value_cache_tensor.get_ptr()),
        const_cast<float*>(static_cast<const float*>(score_tensor.get_ptr())),
        const_cast<float*>(static_cast<const float*>(mha_out.get_ptr())));
  } else {
    mha_kernel_cuda_fp32<<<grid, block, shared_mem>>>(
        pos, head_num, layer_offset, seq_len, kv_dim, kv_mul, head_size,
        static_cast<const float*>(query_tensor.get_ptr()),
        static_cast<const float*>(key_cache_tensor.get_ptr()),
        static_cast<const float*>(value_cache_tensor.get_ptr()),
        const_cast<float*>(static_cast<const float*>(score_tensor.get_ptr())),
        const_cast<float*>(static_cast<const float*>(mha_out.get_ptr())));
  }
}

}  // namespace kernel
