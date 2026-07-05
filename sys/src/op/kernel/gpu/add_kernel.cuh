#pragma once

#include "tensor/tensor.h"

namespace kernel{

    void add_kernel_cuda(
        const tensor::Tensor& x1,
        const tensor::Tensor& x2,
        const tensor::Tensor& y,
        void* stream
    );
}