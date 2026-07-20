#pragma once

#include <random>
#include <vector>

#include "sampler/sampler.h"

namespace sampler {

class TopKSampler : public Sampler {
 private:
  float temperature_;
  int32_t top_k_;
  float top_p_;
  std::mt19937 rng_;

 public:
  explicit TopKSampler(base::DeviceType_t device_type, float temperature = 1.0f,
                       int32_t top_k = 0, float top_p = 1.0f);

  size_t sample(const float* logits, size_t size, void* stream = nullptr) override;
};

}  // namespace sampler
