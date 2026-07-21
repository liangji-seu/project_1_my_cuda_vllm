#include "emb_kernel.cuh"
#include <glog/logging.h>
#include <cuda_runtime.h>

namespace kernel {

//核函数，来实现张量的计算。
__global__ void emb_kernel_cuda_fp32(
  int32_t token_num,//token ids的个数 = 序列长度
  int32_t dim,    //模型维数，weight权重的一行的维度长度
  const int* input,//输入张量
  const float* weight,//权重张量
  float* output) {//输出张量地址


  /**
   * 朴素实现:一个thread处理一个token id的查表映射
   */
  int32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid >= token_num) return;

  int32_t token = input[tid];
  const float* src = weight + token * dim;
  float* dst = output + tid * dim;
  for (int32_t i = 0; i < dim; ++i) {
    dst[i] = src[i];
  }
}



void emb_kernel_cuda(
  const tensor::Tensor& input,
  const tensor::Tensor& weight,
  const tensor::Tensor& output, 
  size_t vocab_size, 
  void* stream) {

  (void)vocab_size;
  CHECK(!input.is_empty());
  CHECK(!weight.is_empty());
  CHECK(!output.is_empty());

  //这里直接用的input的张量size，但是我们的推理框架是分成固定长度的token ids张量 + 有效长度两个输入的。
  //所以，如果要使用embedding算子，输入token_ids张量的时候，要reshape他的形状。
  int32_t token_num = static_cast<int32_t>(input.get_size());
  int32_t dim = static_cast<int32_t>(weight.get_dim(1));

  size_t block_size = 256;
  size_t grid_size = (static_cast<size_t>(token_num) + block_size - 1) / block_size;

  if (stream) {
    cudaStream_t _stream = static_cast<cudaStream_t>(stream);
    emb_kernel_cuda_fp32<<<grid_size, block_size, 0, _stream>>>(
        token_num, dim,
        static_cast<const int*>(input.get_ptr()),
        static_cast<const float*>(weight.get_ptr()),
        const_cast<float*>(static_cast<const float*>(output.get_ptr())));
  } else {
    emb_kernel_cuda_fp32<<<grid_size, block_size>>>(
        token_num, dim,
        static_cast<const int*>(input.get_ptr()),
        static_cast<const float*>(weight.get_ptr()),
        const_cast<float*>(static_cast<const float*>(output.get_ptr())));
  }
}

}
