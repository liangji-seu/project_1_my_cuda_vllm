#include "rope_kernel.h"
#include <glog/logging.h>
#include <cmath>
#include <cstring>

namespace kernel {

void rope_kernel_cpu(int32_t dim, int32_t kv_dim, int32_t head_size,
                     const tensor::Tensor& input_q, const tensor::Tensor& input_k,
                     const tensor::Tensor& input_pos, const tensor::Tensor& sin_cache,
                     const tensor::Tensor& cos_cache, void* stream) {
  (void)stream;

  const int32_t pos = *static_cast<const int32_t*>(input_pos.get_ptr());

  float* q_ptr = const_cast<float*>(static_cast<const float*>(input_q.get_ptr()));
  float* k_ptr = const_cast<float*>(static_cast<const float*>(input_k.get_ptr()));
  const float* sin_ptr = static_cast<const float*>(sin_cache.get_ptr());
  const float* cos_ptr = static_cast<const float*>(cos_cache.get_ptr());

  for (int32_t i = 0; i < dim; i += 2) {
    int32_t head_dim = i % head_size;
    float fci = sin_ptr[pos * head_size + head_dim];
    float fcr = cos_ptr[pos * head_size + head_dim];

    int32_t rotn = i < kv_dim ? 2 : 1;
    for (int32_t v = 0; v < rotn; ++v) {
      float* vec = (v == 0) ? q_ptr : k_ptr;
      float v0 = vec[i];
      float v1 = vec[i + 1];
      vec[i] = v0 * fcr - v1 * fci;
      vec[i + 1] = v0 * fci + v1 * fcr;
    }
  }
}

}
