#include "softmax_kernel.h"
#include <glog/logging.h>
#include <armadillo>

namespace kernel {

void softmax_inplace_cpu(const tensor::Tensor& input, void* stream) {
  (void)stream;
  CHECK(!input.is_empty());
  CHECK(input.get_device_type() == base::DeviceType_t::CPU);

  int32_t size = static_cast<int32_t>(input.get_size());
  float* input_ptr = const_cast<float*>(static_cast<const float*>(input.get_ptr()));

  float max_value = input_ptr[0];
  for (int32_t i = 1; i < size; ++i) {
    if (input_ptr[i] > max_value) max_value = input_ptr[i];
  }

  float sum_exp = 0.0f;
  for (int32_t i = 0; i < size; ++i) {
    input_ptr[i] = std::exp(input_ptr[i] - max_value);
    sum_exp += input_ptr[i];
  }

  for (int32_t i = 0; i < size; ++i) {
    input_ptr[i] /= sum_exp;
  }
}

}
