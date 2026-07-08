#pragma once

#include "layer.h"
#include "base/base.h"

namespace op {

class RoPELayer : public Layer {
private:
  int32_t dim_ = 0;
  int32_t kv_dim_ = 0;
  int32_t head_size_ = 0;

public:
  explicit RoPELayer(base::DeviceType_t device_type, int32_t dim, int32_t kv_dim,
                     int32_t head_size);

  base::error::Status check_layer() override;

  base::error::Status forward() override;
};

}  // namespace op
