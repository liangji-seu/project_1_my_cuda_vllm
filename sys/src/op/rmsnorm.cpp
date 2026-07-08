#include "op/rmsnorm.h"
#include "kernel/kernel_interface.h"

namespace op {

RmsNormLayer::RmsNormLayer(base::DeviceType_t device_type, int32_t dim)
    : dim_(dim),
      LayerParam(device_type, LayerType_t::RMSNorm, false, "RMSNorm") {
  reset_input_tensor_num(1);
  reset_output_tensor_num(1);
  reset_weight_tensor_num(1);
}

base::error::Status RmsNormLayer::check_layer() {
  CHECK(check_tensor(get_input(0)) == base::error::Status());
  CHECK(check_tensor(get_weight(0)) == base::error::Status());
  CHECK(check_tensor(get_output(0)) == base::error::Status());
  return base::error::Status();
}

base::error::Status RmsNormLayer::forward() {
  CHECK(this->check_layer() == base::error::Status());

  auto input = this->get_input(0);
  auto weight = this->get_weight(0);
  auto output = this->get_output(0);

  void* stream_ptr = nullptr;
  if (device_type == base::DeviceType_t::GPU) {
    CHECK(cuda_stream != nullptr);
    stream_ptr = cuda_stream->stream;
  }

  kernel::get_rmsnorm_interface(device_type)(input, weight, output, stream_ptr);
  return base::error::Status();
}

}  // namespace op
