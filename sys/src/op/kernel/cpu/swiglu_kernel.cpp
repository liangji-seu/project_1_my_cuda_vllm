#include "swiglu_kernel.h"
#include <glog/logging.h>
#include <armadillo>

namespace kernel {

void swiglu_kernel_cpu(const tensor::Tensor& input1, const tensor::Tensor& input2,
                       const tensor::Tensor& output, void* stream) {
  (void)stream;
  CHECK(!input1.is_empty());
  CHECK(!input2.is_empty());
  CHECK(!output.is_empty());

  CHECK(input1.get_device_type() == base::DeviceType_t::CPU);
  CHECK(input2.get_device_type() == base::DeviceType_t::CPU);
  CHECK(output.get_device_type() == base::DeviceType_t::CPU);

  const int32_t size = static_cast<int32_t>(input1.get_size());
  CHECK_EQ(size, static_cast<int32_t>(input2.get_size()));
  CHECK_EQ(size, static_cast<int32_t>(output.get_size()));

  float* in1_ptr = const_cast<float*>(static_cast<const float*>(input1.get_ptr()));
  const float* in2_ptr = static_cast<const float*>(input2.get_ptr());
  float* out_ptr = const_cast<float*>(static_cast<const float*>(output.get_ptr()));

  arma::fvec gate_vec(in1_ptr, size, false, true);
  arma::fvec up_vec(const_cast<float*>(in2_ptr), size, false, true);
  arma::fvec out_vec(out_ptr, size, false, true);

  gate_vec %= (1.0f / (1.0f + arma::exp(-gate_vec)));
  out_vec = gate_vec % up_vec;
}

}
