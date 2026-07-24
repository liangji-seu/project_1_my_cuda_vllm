#include "rope_kernel.cuh"
#include "mha_kernel.cuh"
#include <glog/logging.h>
#include <cuda_runtime.h>

namespace kernel {

// 引用 mha_kernel.cu 中的 graph 全局变量
extern bool g_graph_mode;
extern int32_t* g_graph_d_pos;

// LLaMA-style RoPE: rotate (i, i + head_size/2) for each head.
// Matches HuggingFace Qwen2/Qwen3 apply_rotary_pos_emb + rotate_half.
//
// Each thread handles one (head, pair_idx) combination.
__global__ void rope_kernel_cuda_fp32(const int32_t* __restrict__ d_pos,
                                       int32_t dim, int32_t kv_dim,
                                       int32_t head_size,
                                       float* input_q, float* input_k,
                                       const float* sin_cache,
                                       const float* cos_cache) {
  int32_t pos = *d_pos;  // GPU 端读取 (支持 CUDA Graph)
  int32_t half_dim = head_size / 2;
  int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  int32_t total_q_pairs = (dim / head_size) * half_dim;
  if (tid >= total_q_pairs) return;

  int32_t head_idx = tid / half_dim;
  int32_t pair_idx = tid % half_dim;
  int32_t head_off = head_idx * head_size;

  float fci = sin_cache[pos * head_size + pair_idx];
  float fcr = cos_cache[pos * head_size + pair_idx];



  
  // Rotate Q: (q[pair_idx], q[pair_idx + half_dim])
  //全局内存，获取q向量的一个旋转对
  float q0 = input_q[head_off + pair_idx];
  float q1 = input_q[head_off + pair_idx + half_dim];

  //直接写全局内存
  input_q[head_off + pair_idx] = q0 * fcr - q1 * fci;
  input_q[head_off + pair_idx + half_dim] = q0 * fci + q1 * fcr;





  // Rotate K only for KV heads (first kv_dim/head_size heads)
  //K向量也来一遍旋转
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
  static int32_t* d_pos_fallback = nullptr;
  if (!d_pos_fallback) cudaMalloc(&d_pos_fallback, sizeof(int32_t));

  const int32_t pos = *static_cast<const int32_t*>(input_pos.get_ptr());

  const int32_t* pos_ptr;
  if (g_graph_mode) {
    pos_ptr = g_graph_d_pos;  // graph 模式: 外部已更新
  } else {
    pos_ptr = d_pos_fallback;
    cudaMemcpyAsync(d_pos_fallback, &pos, sizeof(int32_t), cudaMemcpyHostToDevice,
                    static_cast<cudaStream_t>(stream));
  }

  size_t threads = 128;
  size_t blocks = ((dim / head_size) * (head_size / 2) + threads - 1) / threads;

  if (stream) {
    rope_kernel_cuda_fp32<<<blocks, threads, 0, static_cast<cudaStream_t>(stream)>>>(
        pos_ptr, dim, kv_dim, head_size,
        const_cast<float*>(static_cast<const float*>(input_q.get_ptr())),
        const_cast<float*>(static_cast<const float*>(input_k.get_ptr())),
        static_cast<const float*>(sin_cache.get_ptr()),
        static_cast<const float*>(cos_cache.get_ptr()));
  } else {
    rope_kernel_cuda_fp32<<<blocks, threads>>>(
        pos_ptr, dim, kv_dim, head_size,
        const_cast<float*>(static_cast<const float*>(input_q.get_ptr())),
        const_cast<float*>(static_cast<const float*>(input_k.get_ptr())),
        static_cast<const float*>(sin_cache.get_ptr()),
        static_cast<const float*>(cos_cache.get_ptr()));
  }
}

}  // namespace kernel
