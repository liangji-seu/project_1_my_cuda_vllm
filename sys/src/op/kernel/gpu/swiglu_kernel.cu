#include "swiglu_kernel.cuh"
#include <glog/logging.h>
#include <cuda_runtime.h>

namespace kernel {

__global__ void swiglu_kernel_cuda_fp32(int32_t size, const float* in1,
                                         const float* in2, float* out) {
  int32_t tid = threadIdx.x;
  int32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= size) return;

  extern __shared__ float shared_mem[];
  float* smem1 = shared_mem;
  float* smem2 = shared_mem + blockDim.x;

  smem1[tid] = in1[idx];
  smem2[tid] = in2[idx];
  __syncthreads();

  float gate_val = smem1[tid];
  float sigmoid_val = 1.0f / (1.0f + expf(-gate_val));
  smem1[tid] = gate_val * sigmoid_val;

  out[idx] = smem1[tid] * smem2[tid];
}

void swiglu_kernel_cuda(const tensor::Tensor& input1, const tensor::Tensor& input2,
                        const tensor::Tensor& output, void* stream) {
  CHECK(!input1.is_empty());
  CHECK(!input2.is_empty());
  CHECK(!output.is_empty());

  int32_t size = static_cast<int32_t>(input1.get_size());
  int32_t threads = 128;
  int32_t blocks = (size + threads - 1) / threads;
  size_t shmem = threads * sizeof(float) * 2;

  if (stream) {
    cudaStream_t _stream = static_cast<cudaStream_t>(stream);
    swiglu_kernel_cuda_fp32<<<blocks, threads, shmem, _stream>>>(
        size,
        static_cast<const float*>(input1.get_ptr()),
        static_cast<const float*>(input2.get_ptr()),
        const_cast<float*>(static_cast<const float*>(output.get_ptr())));
  } else {
    swiglu_kernel_cuda_fp32<<<blocks, threads, shmem>>>(
        size,
        static_cast<const float*>(input1.get_ptr()),
        static_cast<const float*>(input2.get_ptr()),
        const_cast<float*>(static_cast<const float*>(output.get_ptr())));
  }
}

}
