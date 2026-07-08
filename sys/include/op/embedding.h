#pragma once

#include "layer.h"
#include "base/base.h"

namespace op{

class EmbeddingLayer : public LayerParam {
private:
  int32_t dim_ = 0;
  int32_t seq_len_ = 0;
  int32_t vocab_size_ = 0;

public:
  explicit EmbeddingLayer(
      base::DeviceType_t device_type,
      int32_t dim,
      int32_t seq_len,
      int32_t vocab_size);

  base::error::Status check_layer() override;

  base::error::Status forward() override;
};

}  // namespace op
