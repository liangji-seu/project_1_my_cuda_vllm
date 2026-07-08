#include "op/softmax.h"
#include "kernel/kernel_interface.h"
#include <cstring>

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

  // Copy result to output
  auto output = this->get_output(0);
  if (output.get_ptr() != input.get_ptr()) {
    size_t byte_size = input.get_byte_size();
    std::memcpy(output.get_ptr(), input.get_ptr(), byte_size);
  }
  return base::error::Status();
}

}  // namespace op
