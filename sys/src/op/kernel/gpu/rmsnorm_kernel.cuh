#pragma once
#include "tensor/tensor.h"

namespace kernel {

void rmsnorm_kernel_cuda(const tensor::Tensor& input, const tensor::Tensor& weight,
                         const tensor::Tensor& output, void* stream = nullptr);

}
