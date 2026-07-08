#pragma once
#include "tensor/tensor.h"

namespace kernel {

void emb_kernel_cuda(const tensor::Tensor& input, const tensor::Tensor& weight,
                     const tensor::Tensor& output, size_t vocab_size,
                     void* stream = nullptr);

}
