#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "base/base.h"
#include "buffer_type.h"
#include "config.h"
#include "op/embedding.h"
#include "op/encode.h"
#include "op/layer.h"
#include "raw_model_data.h"
#include "sampler/sampler.h"
#include "tensor/tensor.h"

namespace model {

class Model {
 protected:
  int32_t group_size_ = 1;
  bool is_quant_model_ = false;

  base::DeviceType_t device_type_ = base::DeviceType_t::Unknown;
  base::ModelType model_type_ = base::ModelType::kModelTypeUnknown;

  std::unique_ptr<op::EncodeLayerBase> encode_layer_;
  std::unique_ptr<TransformerConfig> config_;

  std::string model_path_;
  std::shared_ptr<RawModelData> raw_model_data_;

  std::string token_path_;
  base::TokenizerType tokenizer_type_ = base::TokenizerType::kEncodeUnknown;

  std::map<ModelBufferType, tensor::Tensor> buffers_;
  std::unique_ptr<sampler::Sampler> sampler_;

 public:
  explicit Model(base::TokenizerType tokenizer_type, base::ModelType model_type,
                 std::string token_path, std::string model_path, bool is_quant_model);

  virtual ~Model() = default;

  virtual base::error::Status init(base::DeviceType_t device_type) = 0;

  virtual base::error::Status forward(const tensor::Tensor& input,
                                      const tensor::Tensor& pos_tensor,
                                      int& next) = 0;

  virtual base::error::Status predict(const tensor::Tensor& input,
                                      const tensor::Tensor& pos_tensor,
                                      bool is_prompt, int& next) = 0;

  // Tokenizer forwarding
  virtual std::vector<int32_t> encode(const std::string& sentence) const;
  virtual std::string decode(int32_t token_idx) const;
  virtual std::string decode(const std::vector<int32_t>& token_idxs) const;
  virtual bool is_sentence_ending(int32_t token_idx) const;

  // Embedding
  virtual op::EmbeddingOutput embedding(const std::vector<int>& tokens) = 0;

  virtual tensor::Tensor fill_input(const tensor::Tensor& pos_tensor,
                                    const op::EmbeddingOutput& embedding_output,
                                    bool is_prompt) const;

  // Buffer and KV Cache
  virtual tensor::Tensor& get_buffer(ModelBufferType buffer_idx);
  virtual const tensor::Tensor& get_buffer(ModelBufferType buffer_idx) const;

  virtual std::pair<tensor::Tensor, tensor::Tensor> slice_kv_cache(
      int32_t layer_idx, int32_t token_pos) const;

  // Query
  base::ModelType model_type() const;
  const std::string& token_path() const;
  const std::string& model_path() const;

 protected:
  virtual base::error::Status insert_buffer(ModelBufferType buffer_idx,
                                            const tensor::Tensor& tensor);

  virtual base::error::Status read_model_file();

  virtual base::error::Status create_encode_layer();

  virtual base::error::Status gen_model_from_file();

  virtual base::error::Status generate_model_infos(const ModelConfig& config,
                                                  int32_t flags) const;

  virtual int32_t post_processing(const tensor::Tensor& pos, bool is_prompt) = 0;

 private:
  virtual void init_mem() = 0;
  virtual base::error::Status create_layers() = 0;
  virtual void create_param_layers() = 0;
  virtual void create_nonparam_layers() = 0;
  virtual void create_param_quant_layers() = 0;
};

}  // namespace model
