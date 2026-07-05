#pragma once
#include "tensor/tensor.h"

namespace kernel{

    //加法算子的CPU后端的内核
    void add_kernel_cpu(
        const tensor::Tensor& x1,
        const tensor::Tensor& x2,
        tensor::Tensor& y,
        void* stream
    );
}

