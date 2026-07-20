#pragma once

#include <cstddef>
#include <cstdint>

#include "base/base.h"

namespace sampler {

class Sampler {
 public:
  explicit Sampler(base::DeviceType_t device_type) : device_type_(device_type) {}

  virtual size_t sample(const float* logits, size_t size, void* stream = nullptr) = 0;

 protected:
  base::DeviceType_t device_type_;
};

}  // namespace sampler
