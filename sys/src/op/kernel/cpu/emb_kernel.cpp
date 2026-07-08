#include "emb_kernel.h"
#include <glog/logging.h>
#include <cstring>

namespace kernel {

void emb_kernel_cpu(const tensor::Tensor& input, const tensor::Tensor& weight,
                    const tensor::Tensor& output, size_t vocab_size, void* stream) {
  (void)stream;
  CHECK(!input.is_empty());
  CHECK(!weight.is_empty());
  CHECK(!output.is_empty());

  const int32_t token_num = static_cast<int32_t>(input.get_size());
  const int32_t dim = static_cast<int32_t>(weight.get_dim(1));

  CHECK(weight.get_device_type() == output.get_device_type());
  CHECK(input.get_device_type() == base::DeviceType_t::CPU);

  const int32_t* input_ptr = static_cast<const int32_t*>(input.get_ptr());
  const float* weight_ptr = static_cast<const float*>(weight.get_ptr());
  float* output_ptr = const_cast<float*>(static_cast<const float*>(output.get_ptr()));

  for (int32_t i = 0; i < token_num; ++i) {
    int32_t token = input_ptr[i];
    CHECK(token < static_cast<int32_t>(vocab_size))
        << "Token index " << token << " exceeds vocab size " << vocab_size;

    const float* src = weight_ptr + token * dim;
    float* dst = output_ptr + i * dim;
    std::memcpy(dst, src, dim * sizeof(float));
  }
}

}  // namespace kernel
