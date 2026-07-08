#pragma once

#include "layer.h"
#include "base/base.h"

namespace op {

class SwiGLULayer : public Layer {
public:
  explicit SwiGLULayer(base::DeviceType_t device_type = base::DeviceType_t::Unknown);

  base::error::Status check_layer() override;

  base::error::Status forward() override;
};

}  // namespace op
