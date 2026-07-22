#pragma once
#include "tensor/tensor.h"

namespace kernel {

void matmul_kernel_cuda(const tensor::Tensor& input, const tensor::Tensor& weight,
                        const float* bias, float scale,
                        const tensor::Tensor& output,
                        void* stream = nullptr);

}
