#include "matmul_kernel.h"
#include <glog/logging.h>
#include <armadillo>

namespace kernel {

void matmul_kernel_cpu(const tensor::Tensor& input, const tensor::Tensor& weight,
                       float scale, const tensor::Tensor& output, void* stream) {
  (void)stream;
  CHECK(!input.is_empty());
  CHECK(!weight.is_empty());
  CHECK(!output.is_empty());
  CHECK(input.get_device_type() == base::DeviceType_t::CPU);

  const float* input_ptr = static_cast<const float*>(input.get_ptr());
  const float* weight_ptr = static_cast<const float*>(weight.get_ptr());
  float* output_ptr = const_cast<float*>(static_cast<const float*>(output.get_ptr()));

  // input shape: [in_dim0] or [in_dim0, in_dim1]
  int32_t in_dim0 = 1;
  int32_t in_dim1 = 1;
  if (input.get_dims_size() == 2) {
    in_dim0 = static_cast<int32_t>(input.get_dim(0));
    in_dim1 = static_cast<int32_t>(input.get_dim(1));
  } else if (input.get_dims_size() == 1) {
    in_dim0 = static_cast<int32_t>(input.get_dim(0));
  }

  CHECK_EQ(weight.get_dims_size(), 2);
  const int32_t wei_dim0 = static_cast<int32_t>(weight.get_dim(0));  // K
  const int32_t wei_dim1 = static_cast<int32_t>(weight.get_dim(1));  // M
  CHECK_EQ(in_dim0, wei_dim1);

  CHECK_EQ(output.get_size(), static_cast<size_t>(wei_dim0 * in_dim1));

  arma::fmat input_mat(const_cast<float*>(input_ptr), in_dim1, in_dim0, false, true);
  arma::fmat weight_mat(const_cast<float*>(weight_ptr), wei_dim1, wei_dim0, false, true);
  arma::fmat output_mat(output_ptr, in_dim1, wei_dim0, false, true);

  output_mat = (input_mat * weight_mat) * scale;
}

}
