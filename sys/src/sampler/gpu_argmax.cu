#include "sampler/gpu_argmax.h"
#include <cuda_runtime.h>
#include <cfloat>
#include <cub/cub.cuh>

namespace sampler {

// 每个线程找局部 argmax, cub::BlockReduce 全局规约
// 输出: 写 int32 token_id 到 gpu_out (GPU buffer), 同时异步拷到 *cpu_out
template <int BLOCK_SIZE>
__global__ void argmax_kernel(const float* __restrict__ input, size_t size,
                               int32_t* __restrict__ gpu_out) {
  int tid = threadIdx.x;

  float max_val = -FLT_MAX;
  size_t max_idx = SIZE_MAX;
  for (size_t i = tid; i < size; i += BLOCK_SIZE) {
    if (input[i] > max_val) {
      max_val = input[i];
      max_idx = i;
    }
  }

  using BlockReduce = cub::BlockReduce<cub::KeyValuePair<size_t, float>, BLOCK_SIZE>;
  __shared__ typename BlockReduce::TempStorage temp;
  cub::KeyValuePair<size_t, float> local{max_idx, max_val};
  auto result = BlockReduce(temp).Reduce(local, cub::ArgMax());

  if (tid == 0) {
    *gpu_out = static_cast<int32_t>(result.key);
  }
}

void gpu_argmax_async(const float* logits_gpu, size_t size,
                      int32_t* token_gpu, int32_t* token_cpu, void* stream) {
  constexpr int BLOCK_SIZE = 256;
  cudaStream_t s = static_cast<cudaStream_t>(stream);

  argmax_kernel<BLOCK_SIZE><<<1, BLOCK_SIZE, 0, s>>>(logits_gpu, size, token_gpu);

  // 异步拷贝到 CPU, 不 sync — 调用方在需要时自行同步
  if (token_cpu) {
    cudaMemcpyAsync(token_cpu, token_gpu, sizeof(int32_t), cudaMemcpyDeviceToHost, s);
  }
}

}  // namespace sampler
