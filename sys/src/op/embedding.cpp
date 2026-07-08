#include "op/embedding.h"
#include "kernel/kernel_interface.h"

namespace op{

EmbeddingLayer::EmbeddingLayer(
    base::DeviceType_t device_type,
    int32_t dim,
    int32_t seq_len,
    int32_t vocab_size)
    : dim_(dim),
      seq_len_(seq_len),
      vocab_size_(vocab_size),
      LayerParam(device_type, LayerType_t::Embedding, false, "Embedding") {
  reset_input_tensor_num(2);
  reset_output_tensor_num(1);
  reset_weight_tensor_num(1);
}

base::error::Status EmbeddingLayer::check_layer() {
  const auto& input_tokens = get_input(0);
  const auto& token_size = get_input(1).get_size();

  if (token_size > input_tokens.get_size()) {
    return base::error::Status(base::error::kInvalidArgument,
                               "The number of input tokens exceeds the tensor size.");
  }

  return base::error::Status();
}

base::error::Status EmbeddingLayer::forward() {
  CHECK(this->check_layer() == base::error::Status());

  auto input = this->get_input(0);
  auto weight = this->get_weight(0);
  auto output = this->get_output(0);

  void* stream_ptr = nullptr;
  if (device_type == base::DeviceType_t::GPU) {
    CHECK(cuda_stream != nullptr);
    stream_ptr = cuda_stream->stream;
  }

  kernel::get_emb_interface(device_type)(input, weight, output,
                                         static_cast<size_t>(vocab_size_), stream_ptr);
  return base::error::Status();
}

}  // namespace op
