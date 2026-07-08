#pragma once
#include "tensor/tensor.h"

namespace kernel {

void softmax_inplace_cpu(const tensor::Tensor& input, void* stream = nullptr);

}
