#include "op/rope.h"
#include "kernel/kernel_interface.h"

namespace op {

RoPELayer::RoPELayer(base::DeviceType_t device_type, int32_t dim, int32_t kv_dim,
                     int32_t head_size)
    : dim_(dim),
      kv_dim_(kv_dim),
      head_size_(head_size),
      Layer(device_type, LayerType_t::RoPe, tensor::DataType_t::fp32, "RoPE") {
  reset_input_tensor_num(5);  // Q, K, pos, sin_cache, cos_cache//cache的长度是上下文长度的向量
  reset_output_tensor_num(1);
}

base::error::Status RoPELayer::check_layer() {
  CHECK(check_tensor(get_input(0)) == base::error::Status());  // Q
  CHECK(check_tensor(get_input(1)) == base::error::Status());  // K
  CHECK(check_tensor(get_input(2)) == base::error::Status());  // pos
  CHECK(check_tensor(get_input(3)) == base::error::Status());  // sin_cache
  CHECK(check_tensor(get_input(4)) == base::error::Status());  // cos_cache
  return base::error::Status();
}

base::error::Status RoPELayer::forward() {
  CHECK(this->check_layer() == base::error::Status());

  auto input_q = this->get_input(0);
  auto input_k = this->get_input(1);
  auto input_pos = this->get_input(2);
  auto sin_cache = this->get_input(3);
  auto cos_cache = this->get_input(4);

  void* stream_ptr = nullptr;
  if (device_type == base::DeviceType_t::GPU) {
    CHECK(cuda_stream != nullptr);
    stream_ptr = cuda_stream->stream;
  }

  kernel::get_rope_interface(device_type)(dim_, kv_dim_, head_size_,
                                          input_q, input_k, input_pos,
                                          sin_cache, cos_cache, stream_ptr);
  return base::error::Status();
}

}  // namespace op
