#include "softmax_kernel.cuh"
#include <glog/logging.h>
#include <cuda_runtime.h>
#include <cmath>

namespace kernel {

__global__ void softmax_inplace_cuda_fp32(int32_t size, float* data) {
  __shared__ float s_max;
  __shared__ float s_sum;

  // Step 1: find max (single block, all threads)
  int32_t tid = threadIdx.x;
  float local_max = -1e30f;
  for (int32_t i = tid; i < size; i += blockDim.x) {
    if (data[i] > local_max) local_max = data[i];
  }

  // Reduce to find global max within the block
  extern __shared__ float shared_buf[];
  shared_buf[tid] = local_max;
  __syncthreads();

  for (int32_t s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      if (shared_buf[tid + s] > shared_buf[tid]) {
        shared_buf[tid] = shared_buf[tid + s];
      }
    }
    __syncthreads();
  }

  if (tid == 0) s_max = shared_buf[0];
  __syncthreads();

  float global_max = s_max;

  // Step 2: exp and sum
  float local_sum = 0.0f;
  for (int32_t i = tid; i < size; i += blockDim.x) {
    float val = expf(data[i] - global_max);
    data[i] = val;
    local_sum += val;
  }

  shared_buf[tid] = local_sum;
  __syncthreads();

  for (int32_t s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      shared_buf[tid] += shared_buf[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) s_sum = shared_buf[0];
  __syncthreads();

  float global_sum = s_sum;

  // Step 3: normalize
  for (int32_t i = tid; i < size; i += blockDim.x) {
    data[i] /= global_sum;
  }
}

void softmax_inplace_cuda(const tensor::Tensor& input, void* stream) {
  CHECK(!input.is_empty());

  int32_t size = static_cast<int32_t>(input.get_size());
  float* data = const_cast<float*>(static_cast<const float*>(input.get_ptr()));

  size_t block_size = 256;
  size_t shared_mem = block_size * sizeof(float);

  if (stream) {
    cudaStream_t _stream = static_cast<cudaStream_t>(stream);
    softmax_inplace_cuda_fp32<<<1, block_size, shared_mem, _stream>>>(size, data);
  } else {
    softmax_inplace_cuda_fp32<<<1, block_size, shared_mem>>>(size, data);
  }
}

}
