#include "model/model.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include <glog/logging.h>

namespace model {

Model::Model(base::TokenizerType tokenizer_type, base::ModelType model_type,
             std::string token_path, std::string model_path, bool is_quant_model)
    : is_quant_model_(is_quant_model),
      model_type_(model_type),
      token_path_(std::move(token_path)),
      model_path_(std::move(model_path)),
      tokenizer_type_(tokenizer_type) {}

base::ModelType Model::model_type() const { return model_type_; }

const std::string& Model::token_path() const { return token_path_; }

const std::string& Model::model_path() const { return model_path_; }

base::error::Status Model::insert_buffer(ModelBufferType buffer_idx,
                                         const tensor::Tensor& tensor) {
  if (buffers_.count(buffer_idx) > 0) {
    return base::error::Status(base::error::kKeyValueHasExist,
                               std::to_string(static_cast<int>(buffer_idx)) +
                                   " has exists in the buffers");
  }
  if (tensor.is_empty()) {
    return base::error::Status(base::error::kInvalidArgument,
                               "The tensor is empty for inserting buffer.");
  }
  buffers_.insert({buffer_idx, tensor});
  return base::error::Status();
}

tensor::Tensor& Model::get_buffer(ModelBufferType buffer_idx) {
  CHECK_GT(buffers_.count(buffer_idx), 0) << static_cast<int>(buffer_idx);
  return buffers_.at(buffer_idx);
}

const tensor::Tensor& Model::get_buffer(ModelBufferType buffer_idx) const {
  CHECK_GT(buffers_.count(buffer_idx), 0);
  return buffers_.at(buffer_idx);
}

base::error::Status Model::read_model_file() {
  if (model_path_.empty()) {
    return base::error::Status(base::error::kPathNotValid,
                               "Failed to open the weight file, the model path is empty!");
  }

  int32_t fd = open(model_path_.data(), O_RDONLY);
  if (fd == -1) {
    return base::error::Status(base::error::kPathNotValid,
                               "Failed to open the weight file " + model_path_ +
                                   " may be the path does not exist!");
  }

  FILE* file = fopen(model_path_.data(), "rb");
  if (!file) {
    return base::error::Status(base::error::kPathNotValid,
                               "Failed to open the file. The path may be invalid.");
  }

  auto config = ModelConfig{};
  if (fread(&config, sizeof(ModelConfig), 1, file) != 1) {
    return base::error::Status(base::error::kModelParseError,
                               "Failed to retrieve the configuration from the model file.");
  }

  // Read flags field (4 bytes, after ModelConfig, before weights)
  int32_t flags = FLAG_NONE;
  size_t flags_read = fread(&flags, sizeof(int32_t), 1, file);
  if (flags_read != 1) {
    // Old format without flags — infer from legacy fields
    flags = FLAG_TIED_WEIGHTS;  // Old format only had tied-weight models
    LOG(WARNING) << "No flags field in model file, using legacy inference (may be wrong!)";
  }

  if (is_quant_model_) {
    if (fread(&group_size_, sizeof(int32_t), 1, file) != 1) {
      return base::error::Status(base::error::kModelParseError,
                                 "Failed to retrieve the group size from the model file.");
    }
  }
  fclose(file);

  auto gen_status = generate_model_infos(config, flags);
  if (!gen_status) {
    return gen_status;
  }

  if (!is_quant_model_) {
    raw_model_data_ = std::make_shared<RawModelDataFp32>();
  } else {
    raw_model_data_ = std::make_shared<RawModelDataInt8>();
  }

  struct stat sb;
  if (fstat(fd, &sb) == -1) {
    close(fd);
    return base::error::Status(base::error::kModelParseError,
                               "Failed to retrieve the file size from the model file.");
  }
  raw_model_data_->file_size = sb.st_size;
  raw_model_data_->fd = fd;

  raw_model_data_->data =
      mmap(nullptr, raw_model_data_->file_size, PROT_READ, MAP_PRIVATE, fd, 0);

  if (raw_model_data_->data == MAP_FAILED || raw_model_data_->data == nullptr) {
    return base::error::Status(base::error::kModelParseError,
                               "Failed to map the weight file " + model_path_ +
                                   " into memory.");
  }

  // Header layout: ModelConfig (32 bytes) + flags (4 bytes) [+ group_size for quant]
  constexpr size_t kHeaderSize = sizeof(ModelConfig) + sizeof(int32_t);  // flags field

  if (!is_quant_model_) {
    raw_model_data_->weight_data =
        static_cast<int8_t*>(raw_model_data_->data) + kHeaderSize;
  } else {
    raw_model_data_->weight_data =
        static_cast<int8_t*>(raw_model_data_->data) + kHeaderSize +
        sizeof(group_size_);
  }

  return base::error::Status();
}

base::error::Status Model::generate_model_infos(const ModelConfig& config,
                                                int32_t flags) const {
  config_->dim_ = config.dim;
  config_->hidden_dim_ = config.hidden_dim;
  config_->layer_num_ = config.layer_num;
  config_->head_num_ = config.head_num;
  config_->kv_head_num_ = config.kv_head_num;
  config_->seq_len_ = config.seq_len;

  config_->head_size_ = config.head_dim > 0 ? config.head_dim : config.dim / config.head_num;
  config_->kv_dim_ = config_->head_size_ * config.kv_head_num;
  config_->kv_mul_ = config.head_num / config.kv_head_num;
  config_->q_dim_ = config_->head_size_ * config.head_num;

  config_->flags_ = flags;
  config_->is_shared_weight_ = (flags & FLAG_TIED_WEIGHTS) != 0;
  config_->vocab_size_ = std::abs(config.vocab_size);
  return base::error::Status();
}

base::error::Status Model::create_encode_layer() {
#if defined(QWEN3_SUPPORT)
  encode_layer_ = std::make_unique<op::QwenEncodeLayer>(token_path_, true, false);
#elif defined(LLAMA3_SUPPORT)
  encode_layer_ = std::make_unique<op::BpeEncodeLayer>(token_path_, true, false);
#endif
  if (!encode_layer_) {
    return base::error::Status(base::error::kInternalError,
                               "Create the encode layer failed.");
  }
  config_->vocab_size_ = encode_layer_->vocab_size();
  if (config_->vocab_size_ <= 0) {
    return base::error::Status(base::error::kInternalError,
                               "The vocab size param read error from the model file!");
  }
  return base::error::Status();
}

base::error::Status Model::gen_model_from_file() {
  config_ = std::make_unique<TransformerConfig>();

  auto create_encode_status = create_encode_layer();
  if (!create_encode_status) {
    LOG(ERROR) << "Create the encode layer failed!";
    return create_encode_status;
  }

  auto mmap_status = read_model_file();
  if (!mmap_status) {
    LOG(ERROR) << "Handle model file " << model_path_ << " failed!";
    return mmap_status;
  }

  auto layer_create_status = create_layers();
  if (!layer_create_status) {
    LOG(ERROR) << "Create layers for the model file " << model_path_ << " failed!";
    return layer_create_status;
  }

  return base::error::Status();
}

std::vector<int32_t> Model::encode(const std::string& sentence) const {
  CHECK(encode_layer_ != nullptr);
  return encode_layer_->encode(sentence);
}

bool Model::is_sentence_ending(int32_t token_idx) const {
  CHECK(encode_layer_ != nullptr);
  return encode_layer_->is_sentence_ending(token_idx);
}

std::string Model::decode(int32_t token_idx) const {
  CHECK(encode_layer_ != nullptr);
  return encode_layer_->decode(token_idx);
}

std::string Model::decode(const std::vector<int32_t>& token_idxs) const {
  CHECK(encode_layer_ != nullptr);
  return encode_layer_->decode(token_idxs);
}

std::pair<tensor::Tensor, tensor::Tensor> Model::slice_kv_cache(
    int32_t layer_idx, int32_t token_pos) const {
  int32_t layer_offset = layer_idx * config_->seq_len_ * config_->kv_dim_;
  int32_t cache_offset = layer_offset + token_pos * config_->kv_dim_;

  float* key_ptr = static_cast<float*>(
      const_cast<void*>(get_buffer(ModelBufferType::kKeyCache).get_ptr())) +
      cache_offset;
  float* val_ptr = static_cast<float*>(
      const_cast<void*>(get_buffer(ModelBufferType::kValueCache).get_ptr())) +
      cache_offset;

  tensor::Tensor key(tensor::DataType_t::fp32, {static_cast<size_t>(config_->kv_dim_)},
                     false, nullptr, key_ptr);
  tensor::Tensor val(tensor::DataType_t::fp32, {static_cast<size_t>(config_->kv_dim_)},
                     false, nullptr, val_ptr);

  key.set_device_type(device_type_);
  val.set_device_type(device_type_);
  return {key, val};
}

tensor::Tensor Model::fill_input(const tensor::Tensor& pos_tensor,
                                 const op::EmbeddingOutput& embedding_output,
                                 bool is_prompt) const {
  const int32_t pos = static_cast<const int32_t*>(pos_tensor.get_ptr())[0];

  auto [input_tokens, input_embeddings, input_token_num] = embedding_output;

  int32_t index = 0;
  if (is_prompt) {
    index = pos;
  }

  tensor::Tensor input(tensor::DataType_t::fp32, {static_cast<size_t>(config_->dim_)},
                       false, nullptr,
                       const_cast<float*>(
                           static_cast<const float*>(input_embeddings.get_ptr()) +
                           index * config_->dim_));
  input.set_device_type(device_type_);
  return input;
}

}  // namespace model
