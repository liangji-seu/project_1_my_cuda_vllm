#pragma once

#include <string>
#include <vector>
#include <memory>

#include "op/layer.h"

#if defined(LLAMA3_SUPPORT) || defined(QWEN3_SUPPORT)
#include <absl/strings/str_join.h>
#include <absl/strings/str_replace.h>
#include <absl/strings/str_split.h>
#include "base/tiktoken.h"
#include "base/unordered_dense.h"
#include "base/unicode.h"
#include "nlohmann/json.hpp"
#endif

namespace op {

class EncodeLayerBase : public Layer {
protected:
  bool has_bos_ = true;
  bool has_eos_ = false;
  std::string token_model_path_;


public:
  explicit EncodeLayerBase(std::string token_model_path, bool has_bos, bool has_eos)
      : Layer(base::DeviceType_t::CPU, LayerType_t::Encode, tensor::DataType_t::fp32, "Encode"),
        has_bos_(has_bos),
        has_eos_(has_eos),
        token_model_path_(std::move(token_model_path)) {}

  virtual std::vector<int32_t> encode(const std::string& sentence) const = 0;
  virtual std::string decode(int32_t token_id) const = 0;
  virtual std::string decode(const std::vector<int32_t>& token_ids) const = 0;
  virtual bool is_sentence_ending(int32_t token_id) const = 0;
  virtual int32_t vocab_size() const = 0;


};





#if defined(LLAMA3_SUPPORT) || defined(QWEN3_SUPPORT)
class BpeEncodeLayer : public EncodeLayerBase {
protected:
  int32_t bos_id_ = -1;
  int32_t eos_id_ = -1;
  int32_t stop_token1_ = -1;
  int32_t stop_token2_ = -1;
  int32_t num_token_ = 0;
  std::unique_ptr<tiktoken::tiktoken> tiktoken_;



public:
  explicit BpeEncodeLayer(std::string token_model_path, bool has_bos, bool has_eos);

  std::vector<int32_t> encode(const std::string& sentence) const override;
  std::string decode(int32_t token_id) const override;
  std::string decode(const std::vector<int32_t>& token_ids) const override;
  bool is_sentence_ending(int32_t token_id) const override;
  int32_t vocab_size() const override;
};

class QwenEncodeLayer : public BpeEncodeLayer {
 public:
  explicit QwenEncodeLayer(std::string token_model_path, bool has_bos, bool has_eos);
};
#endif

}  // namespace op
