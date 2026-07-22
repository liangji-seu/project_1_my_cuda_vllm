#pragma once

#include "layer.h"
#include "base/base.h"

namespace op {

class MatmulLayer : public LayerParam {
private:
  float scale_ = 1.0f;
  bool has_bias_ = false;

public:
  explicit MatmulLayer(base::DeviceType_t device_type, float scale = 1.0f,
                       bool has_bias = false);

  base::error::Status check_layer() override;

  base::error::Status forward() override;
};

}  // namespace op
