#include "base/cuda_stream.h"

namespace kernel {

CudaStream::~CudaStream() {
    if (stream != nullptr) {
        cudaStreamDestroy(stream);
    }
}

} // namespace kernel
