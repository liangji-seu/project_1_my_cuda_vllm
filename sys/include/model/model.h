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
#include "profile/profiler.h"
#include "raw_model_data.h"
#include "sampler/sampler.h"
#include "tensor/tensor.h"

namespace model {

/**
 * 抽象模型的基类
 */
class Model {
 protected:
  //量化相关
  int32_t group_size_ = 1;
  bool is_quant_model_ = false;

  //基础信息
  base::DeviceType_t device_type_ = base::DeviceType_t::Unknown;
  base::ModelType model_type_ = base::ModelType::kModelTypeUnknown;

  std::unique_ptr<op::EncodeLayerBase> encode_layer_;//分词器层
  std::unique_ptr<TransformerConfig> config_;//模型超参

  std::string model_path_;//模型参数权重
  std::shared_ptr<RawModelData> raw_model_data_;//权重文件

  std::string token_path_;//词表文件
  base::TokenizerType tokenizer_type_ = base::TokenizerType::kEncodeUnknown;//分词器类型

  std::map<ModelBufferType, tensor::Tensor> buffers_;//张量内存 + cache内存（提前分配张量内存）
  std::unique_ptr<sampler::Sampler> sampler_;//采样器

  // Optional profiler for benchmark/layer-profiling (not owned)
  profile::Profiler* profiler_ = nullptr;

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

  // Profiler access (set by demo/benchmark for optional profiling)
  void set_profiler(profile::Profiler* p) { profiler_ = p; }
  profile::Profiler* profiler() const { return profiler_; }

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

  //构造一个分词器类对象实例
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
