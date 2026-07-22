#include "op/matmul.h"
#include "kernel/kernel_interface.h"

namespace op {

MatmulLayer::MatmulLayer(base::DeviceType_t device_type, float scale)
    : scale_(scale),
      LayerParam(device_type, LayerType_t::Matmul, false, "Matmul") {
  reset_input_tensor_num(1);//一个输入张量
  reset_output_tensor_num(1);//一个输出张量
  reset_weight_tensor_num(1);  // 一个权重矩阵
}

base::error::Status MatmulLayer::check_layer() {
  CHECK(check_tensor(get_input(0)) == base::error::Status());
  CHECK(check_tensor(get_weight(0)) == base::error::Status());
  CHECK(check_tensor(get_output(0)) == base::error::Status());
  return base::error::Status();
}

base::error::Status MatmulLayer::forward() {
  CHECK(this->check_layer() == base::error::Status());

  auto input = this->get_input(0);
  auto weight = this->get_weight(0);
  auto output = this->get_output(0);

  void* stream_ptr = nullptr;
  if (device_type == base::DeviceType_t::GPU) {
    CHECK(cuda_stream != nullptr);
    stream_ptr = cuda_stream->stream;
  }

  kernel::get_matmul_interface(device_type)(input, weight, scale_, output, stream_ptr);
  return base::error::Status();
}

}  // namespace op
