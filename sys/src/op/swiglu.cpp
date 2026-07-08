#include "op/swiglu.h"
#include "kernel/kernel_interface.h"

namespace op {

SwiGLULayer::SwiGLULayer(base::DeviceType_t device_type)
    : Layer(device_type, LayerType_t::SwiGLU, tensor::DataType_t::fp32, "SwiGLU") {
  reset_input_tensor_num(2);  // gate, up
  reset_output_tensor_num(1);
}

base::error::Status SwiGLULayer::check_layer() {
  CHECK(check_tensor(get_input(0)) == base::error::Status());
  CHECK(check_tensor(get_input(1)) == base::error::Status());
  CHECK(check_tensor(get_output(0)) == base::error::Status());
  return base::error::Status();
}

base::error::Status SwiGLULayer::forward() {
  CHECK(this->check_layer() == base::error::Status());

  auto input1 = this->get_input(0);  // gate
  auto input2 = this->get_input(1);  // up
  auto output = this->get_output(0);

  void* stream_ptr = nullptr;
  if (device_type == base::DeviceType_t::GPU) {
    CHECK(cuda_stream != nullptr);
    stream_ptr = cuda_stream->stream;
  }

  kernel::get_swiglu_interface(device_type)(input1, input2, output, stream_ptr);
  return base::error::Status();
}

}  // namespace op
