#include "sampler/argmax_sampler.h"

#include <algorithm>

namespace sampler {

size_t ArgmaxSampler::sample(const float* logits, size_t size, void* stream) {
  if (device_type_ == base::DeviceType_t::CPU) {
    return static_cast<size_t>(std::distance(logits, std::max_element(logits, logits + size)));
  }
  // GPU: fallback to CPU for now
  return static_cast<size_t>(std::distance(logits, std::max_element(logits, logits + size)));
}

}  // namespace sampler
