#include "op/softmax.h"
#include "kernel/kernel_interface.h"
#include <cstring>
#include <cuda_runtime.h>

namespace op {

SoftmaxLayer::SoftmaxLayer(base::DeviceType_t device_type)
    : Layer(device_type, LayerType_t::Softmax, tensor::DataType_t::fp32, "Softmax") {
  reset_input_tensor_num(1);
  reset_output_tensor_num(1);
}

base::error::Status SoftmaxLayer::check_layer() {
  CHECK(check_tensor(get_input(0)) == base::error::Status());
  return base::error::Status();
}

base::error::Status SoftmaxLayer::forward() {
  CHECK(this->check_layer() == base::error::Status());

  auto input = this->get_input(0);

  void* stream_ptr = nullptr;
  if (device_type == base::DeviceType_t::GPU) {
    CHECK(cuda_stream != nullptr);
    stream_ptr = cuda_stream->stream;
  }

  kernel::get_softmax_interface(device_type)(input, stream_ptr);

  // Copy result to output (if different buffer)
  auto output = this->get_output(0);
  if (output.get_ptr() != input.get_ptr()) {
    size_t byte_size = input.get_byte_size();
    if (device_type == base::DeviceType_t::GPU) {
      cudaStream_t _stream = stream_ptr ? static_cast<cudaStream_t>(stream_ptr) : nullptr;
      if (_stream) {
        cudaMemcpyAsync(output.get_ptr(), input.get_ptr(), byte_size,
                        cudaMemcpyDeviceToDevice, _stream);
      } else {
        cudaMemcpy(output.get_ptr(), input.get_ptr(), byte_size,
                   cudaMemcpyDeviceToDevice);
      }
    } else {
      std::memcpy(output.get_ptr(), input.get_ptr(), byte_size);
    }
  }
  return base::error::Status();
}

}  // namespace op
