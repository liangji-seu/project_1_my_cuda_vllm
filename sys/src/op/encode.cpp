#include "op/encode.h"
#include <fstream>
#include <glog/logging.h>

namespace op {

#if defined(LLAMA3_SUPPORT) || defined(QWEN3_SUPPORT)
static const std::string PAT_STR =
    R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?:$|[^\S])|\s+)";

BpeEncodeLayer::BpeEncodeLayer(std::string token_model_path, bool has_bos, bool has_eos)
    : EncodeLayerBase(std::move(token_model_path), has_bos, has_eos) {
  using json = nlohmann::json;
  std::ifstream f(token_model_path_);
  CHECK(f.is_open())
      << "The token model path is not valid, please check the path and type of token model.";

  json data;
  try {
    data = json::parse(f);
  } catch (json::parse_error&) {
    LOG(FATAL)
        << "The token model path is not valid, please check the path and type of token model.";
  }

  const auto& datas = data["added_tokens"];
  ankerl::unordered_dense::map<std::string, int> special_tokens;
  for (const auto& data1 : datas) {
    int id = data1["id"];
    std::string content = data1["content"];
    special_tokens.insert({content, id});
  }

  ankerl::unordered_dense::map<std::string, int> encoder;
  const auto& vocabs = data["model"]["vocab"];
  const auto& vocab_items = vocabs.items();
  for (const auto& v : vocab_items) {
    const auto cpts = unicode_cpts_from_utf8(v.key());
    std::string key;
    for (const auto cpt : cpts) {
      const auto utf8 = unicode_cpt_to_utf8(cpt);
      key += unicode_utf8_to_byte(utf8);
    }
    const int32_t id = v.value();
    encoder[key] = id;
  }

  //默认的特殊tokens
  bos_id_ = special_tokens["<|begin_of_text|>"];
  eos_id_ = special_tokens["<|end_of_text|>"];
  stop_token1_ = eos_id_;
  stop_token2_ = special_tokens["<|eot_id|>"];

  num_token_ = encoder.size() + special_tokens.size();
  tiktoken_ = std::make_unique<tiktoken::tiktoken>(encoder, special_tokens, PAT_STR);
}

std::vector<int32_t> BpeEncodeLayer::encode(const std::string& sentence) const {
  CHECK(this->tiktoken_ != nullptr);
  std::map<std::string, std::string> replacements;
  replacements[" "] = "Ġ";
  std::string s = absl::StrReplaceAll(sentence, replacements);
  auto input_ids = this->tiktoken_->encode(s);

  //构造好主动添加tokenid的bos, eos
  if (has_bos_) {
    input_ids.insert(input_ids.begin(), bos_id_);
  }
  if (has_eos_) {
    input_ids.push_back(eos_id_);
  }
  return input_ids;
}

std::string BpeEncodeLayer::decode(int32_t token_id) const {
  CHECK(this->tiktoken_ != nullptr);
  return decode(std::vector<int32_t>{token_id});
}

std::string BpeEncodeLayer::decode(const std::vector<int32_t>& token_ids) const {
  CHECK(this->tiktoken_ != nullptr);
  try {
    auto s = tiktoken_->decode(token_ids);
    std::map<std::string, std::string> reverse_replacements;
    reverse_replacements["Ġ"] = " ";
    const std::string& sentence = absl::StrReplaceAll(s, reverse_replacements);
    return sentence;
  } catch (const std::exception& e) {
    // Some token IDs may not be individually decodable (e.g. special tokens
    // or continuation bytes). Return empty string for those.
    return std::string();
  }
}

bool BpeEncodeLayer::is_sentence_ending(int32_t token_id) const {
  return token_id == stop_token1_ || token_id == stop_token2_;
}

int32_t BpeEncodeLayer::vocab_size() const {
  CHECK(this->tiktoken_ != nullptr);
  return num_token_;
}

QwenEncodeLayer::QwenEncodeLayer(std::string token_model_path, bool has_bos, bool has_eos)
    : BpeEncodeLayer(std::move(token_model_path), has_bos, has_eos) {
  // BpeEncodeLayer already parsed the JSON and created the tiktoken engine.
  // Re-parse to get Qwen3-specific special token IDs.
  using json = nlohmann::json;
  std::ifstream f(token_model_path_);
  CHECK(f.is_open()) << "Failed to open token model path: " << token_model_path_;
  json data;
  try {
    data = json::parse(f);
  } catch (json::parse_error&) {
    LOG(FATAL) << "Failed to parse token model JSON: " << token_model_path_;
  }

  const auto& datas = data["added_tokens"];//特殊词
  ankerl::unordered_dense::map<std::string, int> special_tokens;
  for (const auto& data1 : datas) {
    int id = data1["id"];
    std::string content = data1["content"];
    special_tokens.insert({content, id});
  }

  bos_id_ = special_tokens["<|im_start|>"];
  eos_id_ = special_tokens["<|im_end|>"];
  stop_token1_ = eos_id_;
  stop_token2_ = special_tokens["<|endoftext|>"];
}
#endif

}  // namespace op
