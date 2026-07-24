#pragma once
#include "tensor/tensor.h"

namespace kernel {

void matmul_int8_kernel_cuda(const tensor::Tensor& input,
                              const tensor::Tensor& weight,
                              const tensor::Tensor& scales,
                              const float* bias,
                              const tensor::Tensor& output,
                              void* stream = nullptr);

void matmul_int8_kernel_cpu(const tensor::Tensor& input,
                             const tensor::Tensor& weight,
                             const tensor::Tensor& scales,
                             const float* bias,
                             const tensor::Tensor& output,
                             void* stream = nullptr);

}  // namespace kernel
