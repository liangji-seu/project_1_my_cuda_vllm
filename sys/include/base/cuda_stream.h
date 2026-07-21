#pragma once

#include <cuda_runtime.h>

namespace kernel {

// 将cudaStream的生命周期和CudaStream类对象绑定，实现RAII
class CudaStream {
public:
    cudaStream_t stream = nullptr; // cuda工作流

    CudaStream() {
        cudaStreamCreate(&stream);
    }

    ~CudaStream();
};

} // namespace kernel
