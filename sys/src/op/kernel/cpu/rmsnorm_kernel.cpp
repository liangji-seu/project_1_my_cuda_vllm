#include "rmsnorm_kernel.h"
#include <glog/logging.h>
#include <armadillo>

namespace kernel {

void rmsnorm_kernel_cpu(const tensor::Tensor& input, const tensor::Tensor& weight,
                        const tensor::Tensor& output, void* stream) {
  (void)stream;
  CHECK(!input.is_empty());
  CHECK(!weight.is_empty());
  CHECK(!output.is_empty());

  CHECK(input.get_device_type() == base::DeviceType_t::CPU);

  const float* in_ptr = static_cast<const float*>(input.get_ptr());
  const float* wei_ptr = static_cast<const float*>(weight.get_ptr());
  float* out_ptr = const_cast<float*>(static_cast<const float*>(output.get_ptr()));
  const int32_t dim = static_cast<int32_t>(input.get_size());

  arma::fvec in_tensor(const_cast<float*>(in_ptr), dim, false, true);
  arma::fvec out_tensor(out_ptr, dim, false, true);
  arma::fvec wei_tensor(const_cast<float*>(wei_ptr), dim, false, true);

  const float eps = 1e-6f;  // Qwen2.5 rms_norm_eps = 1e-6

  const float mean = arma::as_scalar(arma::mean(arma::pow(in_tensor, 2))) + eps;
  const float rsqrt = 1.0f / std::sqrt(mean);
  out_tensor = wei_tensor % (rsqrt * in_tensor);
}

}
