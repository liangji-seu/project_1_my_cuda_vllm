#include "op/matmul.h"
#include "kernel/kernel_interface.h"

namespace op {

MatmulLayer::MatmulLayer(base::DeviceType_t device_type, float scale, bool has_bias,
                         bool is_quant)
    : scale_(scale),
      has_bias_(has_bias),
      LayerParam(device_type, LayerType_t::Matmul, is_quant, "Matmul") {
  reset_input_tensor_num(1);
  reset_output_tensor_num(1);
  reset_weight_tensor_num(has_bias ? 2 : 1);  // weight[0]=W, weight[1]=bias
}

base::error::Status MatmulLayer::check_layer() {
  CHECK(check_tensor(get_input(0)) == base::error::Status());
  CHECK(check_tensor(get_weight(0)) == base::error::Status());
  CHECK(check_tensor(get_output(0)) == base::error::Status());
  if (is_quant_layer) {
    CHECK(!scales.is_empty()) << "Quantized matmul layer must have scales";
    CHECK(get_weight(0).get_data_type() == tensor::DataType_t::int8)
        << "Quantized matmul weight must be INT8";
  }
  return base::error::Status();
}

base::error::Status MatmulLayer::forward() {
  CHECK(this->check_layer() == base::error::Status());

  auto input = this->get_input(0);
  auto weight = this->get_weight(0);
  auto output = this->get_output(0);

  const float* bias = nullptr;
  if (has_bias_ && !get_weight(1).is_empty()) {
    bias = static_cast<const float*>(get_weight(1).get_ptr());
  }

  void* stream_ptr = nullptr;
  if (device_type == base::DeviceType_t::GPU) {
    CHECK(cuda_stream != nullptr);
    stream_ptr = cuda_stream->stream;
  }

  if (is_quant_layer) {
    CHECK(!scales.is_empty()) << "Quantized matmul missing scales";
    kernel::get_matmul_int8_interface(device_type)(input, weight, scales, bias, output, stream_ptr);
  } else {
    kernel::get_matmul_interface(device_type)(input, weight, bias, scale_, output, stream_ptr);
  }
  return base::error::Status();
}

}  // namespace op
