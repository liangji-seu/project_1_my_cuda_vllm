#include "sampler/gpu_argmax.h"
#include <cuda_runtime.h>
#include <cfloat>
#include <cub/cub.cuh>

namespace sampler {

// 每个线程找自己负责范围内的局部 argmax, 再用 cub::BlockReduce 全局规约
template <int BLOCK_SIZE>
__global__ void argmax_kernel(const float* __restrict__ input, size_t size,
                               float* __restrict__ out_val, size_t* __restrict__ out_idx) {
  int tid = threadIdx.x;

  // grid-stride: 每个线程找自己负责元素中的最大值
  float max_val = -FLT_MAX;
  size_t max_idx = SIZE_MAX;
  for (size_t i = tid; i < size; i += BLOCK_SIZE) {
    if (input[i] > max_val) {
      max_val = input[i];
      max_idx = i;
    }
  }

  // cub::BlockReduce — KeyValuePair 做 argmax 规约
  using BlockReduce = cub::BlockReduce<cub::KeyValuePair<size_t, float>, BLOCK_SIZE>;
  __shared__ typename BlockReduce::TempStorage temp;
  cub::KeyValuePair<size_t, float> local{max_idx, max_val};
  auto result = BlockReduce(temp).Reduce(local, cub::ArgMax());

  if (tid == 0) {
    *out_val = result.value;
    *out_idx = result.key;
  }
}

size_t gpu_argmax(const float* logits_gpu, size_t size, void* stream) {
  constexpr int BLOCK_SIZE = 256;

  // 使用静态预分配的 GPU buffer (避免每次 cudaMalloc)
  static float* d_val = nullptr;
  static size_t* d_idx = nullptr;
  if (!d_val) {
    cudaMalloc(&d_val, sizeof(float));
    cudaMalloc(&d_idx, sizeof(size_t));
  }

  cudaStream_t s = static_cast<cudaStream_t>(stream);
  argmax_kernel<BLOCK_SIZE><<<1, BLOCK_SIZE, 0, s>>>(logits_gpu, size, d_val, d_idx);

  size_t result_idx;
  cudaMemcpyAsync(&result_idx, d_idx, sizeof(size_t), cudaMemcpyDeviceToHost, s);
  cudaStreamSynchronize(s);

  return result_idx;
}

}  // namespace sampler
