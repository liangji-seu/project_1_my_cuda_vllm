#pragma once
#include "tensor/tensor.h"

namespace kernel {

void swiglu_kernel_cuda(const tensor::Tensor& input1, const tensor::Tensor& input2,
                        const tensor::Tensor& output, void* stream = nullptr);

}
