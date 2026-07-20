#include "sampler/topk_sampler.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>

namespace sampler {

TopKSampler::TopKSampler(base::DeviceType_t device_type, float temperature,
                         int32_t top_k, float top_p)
    : Sampler(device_type),
      temperature_(temperature),
      top_k_(top_k),
      top_p_(top_p) {
  std::random_device rd;
  rng_ = std::mt19937(rd());
}

size_t TopKSampler::sample(const float* logits, size_t size, void* stream) {
  (void)stream;

  // Step 1: temperature scaling + find max for numerical stability
  std::vector<float> probs(size);
  float max_logit = logits[0] / temperature_;
  for (size_t i = 1; i < size; ++i) {
    float scaled = logits[i] / temperature_;
    if (scaled > max_logit) max_logit = scaled;
  }

  // Step 2: softmax
  float sum_exp = 0.0f;
  for (size_t i = 0; i < size; ++i) {
    probs[i] = std::exp(logits[i] / temperature_ - max_logit);
    sum_exp += probs[i];
  }
  for (size_t i = 0; i < size; ++i) {
    probs[i] /= sum_exp;
  }

  // Step 3: top-k filtering — keep top_k indices, zero the rest
  if (top_k_ > 0 && static_cast<size_t>(top_k_) < size) {
    std::vector<std::pair<float, size_t>> indexed;
    indexed.reserve(size);
    for (size_t i = 0; i < size; ++i) {
      indexed.emplace_back(probs[i], i);
    }
    std::nth_element(indexed.begin(), indexed.begin() + top_k_, indexed.end(),
                     std::greater<std::pair<float, size_t>>{});

    // Zero out probabilities below top-k
    float threshold = indexed[top_k_ - 1].first;
    for (size_t i = 0; i < size; ++i) {
      if (probs[i] < threshold) probs[i] = 0.0f;
    }
    // Re-normalize
    float new_sum =
        std::accumulate(probs.begin(), probs.end(), 0.0f);
    if (new_sum > 0.0f) {
      for (size_t i = 0; i < size; ++i) {
        probs[i] /= new_sum;
      }
    }
  }

  // Step 4: top-p (nucleus) filtering
  if (top_p_ < 1.0f) {
    std::vector<std::pair<float, size_t>> indexed;
    indexed.reserve(size);
    for (size_t i = 0; i < size; ++i) {
      if (probs[i] > 0.0f) {
        indexed.emplace_back(probs[i], i);
      }
    }
    std::sort(indexed.begin(), indexed.end(),
              std::greater<std::pair<float, size_t>>{});

    float cumsum = 0.0f;
    std::vector<bool> keep(size, false);
    for (const auto& [p, idx] : indexed) {
      cumsum += p;
      keep[idx] = true;
      if (cumsum >= top_p_) break;
    }

    float new_sum = 0.0f;
    for (size_t i = 0; i < size; ++i) {
      if (!keep[i]) probs[i] = 0.0f;
      new_sum += probs[i];
    }
    if (new_sum > 0.0f) {
      for (size_t i = 0; i < size; ++i) {
        probs[i] /= new_sum;
      }
    } else {
      // Fallback: all filtered, use argmax
      size_t best = 0;
      float best_p = probs[0];
      for (size_t i = 1; i < size; ++i) {
        if (probs[i] > best_p) {
          best_p = probs[i];
          best = i;
        }
      }
      return best;
    }
  }

  // Step 5: sample from the filtered distribution
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  float r = dist(rng_);
  float cumsum = 0.0f;
  for (size_t i = 0; i < size; ++i) {
    cumsum += probs[i];
    if (r < cumsum) {
      return i;
    }
  }

  // Fallback: return the max probability index
  return std::distance(probs.begin(),
                       std::max_element(probs.begin(), probs.end()));
}

}  // namespace sampler
