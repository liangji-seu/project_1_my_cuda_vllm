#pragma once
#include "tensor/tensor.h"

namespace kernel {

void softmax_inplace_cuda(const tensor::Tensor& input, void* stream = nullptr);

}
