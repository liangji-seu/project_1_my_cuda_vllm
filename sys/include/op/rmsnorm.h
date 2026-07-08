#pragma once

#include "layer.h"
#include "base/base.h"

namespace op {

class RmsNormLayer : public LayerParam {
private:
  int32_t dim_ = 0;

public:
  explicit RmsNormLayer(base::DeviceType_t device_type, int32_t dim);

  base::error::Status check_layer() override;

  base::error::Status forward() override;
};

}  // namespace op
