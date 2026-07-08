#include "op/mha.h"
#include "kernel/kernel_interface.h"

namespace op {

MultiHeadAttentionLayer::MultiHeadAttentionLayer(
    base::DeviceType_t device_type,
    int32_t layer_index,
    int32_t kv_mul,
    int32_t kv_dim,
    int32_t seq_len,
    int32_t head_num,
    int32_t head_size)
    : Layer(device_type, LayerType_t::MHA, tensor::DataType_t::fp32, "MultiHeadAttention"),
      layer_index_(layer_index),
      kv_mul_(kv_mul),
      kv_dim_(kv_dim),
      seq_len_(seq_len),
      head_num_(head_num),
      head_size_(head_size) {
  reset_input_tensor_num(5);
  reset_output_tensor_num(1);
}

void MultiHeadAttentionLayer::set_pos(int32_t pos) { pos_ = pos; }

void MultiHeadAttentionLayer::set_layer_index(int32_t layer_index) { layer_index_ = layer_index; }

base::error::Status MultiHeadAttentionLayer::check_layer() {
  const int32_t input_tensor_num = 4;
  for (int32_t i = 0; i < input_tensor_num; ++i) {
    CHECK(check_tensor(get_input(i)) == base::error::Status());
  }
  CHECK(check_tensor(get_output(0)) == base::error::Status());
  return base::error::Status();
}

base::error::Status MultiHeadAttentionLayer::forward() {
  CHECK(this->check_layer() == base::error::Status());

  const auto& mha_out = this->get_output(0);
  const auto& query_tensor = this->get_input(0);
  const auto& score_tensor = this->get_input(1);
  const auto& key_cache_tensor = this->get_input(2);
  const auto& value_cache_tensor = this->get_input(3);

  void* stream_ptr = nullptr;
  if (device_type == base::DeviceType_t::GPU) {
    CHECK(cuda_stream != nullptr);
    stream_ptr = cuda_stream->stream;
  }

  kernel::get_mha_interface(device_type)(
      pos_, head_num_, layer_index_, seq_len_, kv_dim_, kv_mul_, head_size_,
      mha_out, query_tensor, score_tensor, key_cache_tensor, value_cache_tensor,
      stream_ptr);
  return base::error::Status();
}

}  // namespace op
