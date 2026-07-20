#include "model/llama3.h"

#include <glog/logging.h>

#include "base/DeviceController.h"
#include "../op/kernel/cpu/rope_kernel.h"
#include "sampler/argmax_sampler.h"

namespace model {

LLama2Model::LLama2Model(base::TokenizerType tokenizer_type, std::string token_path,
                         std::string model_path, bool is_quant_model)
    : Model(tokenizer_type, base::ModelType::kModelTypeLLama2,
            std::move(token_path), std::move(model_path), is_quant_model) {}

base::error::Status LLama2Model::init(base::DeviceType_t device_type) {
  if (token_path_.empty()) {
    return base::error::Status(base::error::kPathNotValid, token_path_);
  }

  device_type_ = device_type;

  auto read_status = gen_model_from_file();
  if (!read_status) {
    return read_status;
  }

  init_mem();

  sampler_ = std::make_unique<sampler::ArgmaxSampler>(device_type_);
  return base::error::Status();
}

base::error::Status LLama2Model::forward(const tensor::Tensor& input,
                                         const tensor::Tensor& pos_tensor,
                                         int& next) {
  (void)next;
  if (input.is_empty()) {
    return base::error::Status(base::error::kInvalidArgument,
                               "The input tensor is empty.");
  }

  for (int32_t layer_idx = 0; layer_idx < config_->layer_num_; ++layer_idx) {
    attention_rms(layer_idx, input);
    attention_qkv(layer_idx, pos_tensor);
    attention_mha(layer_idx, pos_tensor);
    feed_forward(layer_idx, input);
  }
  cls_logits(input);
  return base::error::Status();
}

base::error::Status LLama2Model::predict(const tensor::Tensor& input,
                                         const tensor::Tensor& pos_tensor,
                                         bool is_prompt, int& next) {
  auto status = forward(input, pos_tensor, next);
  if (!status) {
    return status;
  }
  next = post_processing(pos_tensor, is_prompt);
  return base::error::Status();
}

void LLama2Model::create_nonparam_layers() {
  CHECK(llama_layers_ != nullptr);
  llama_layers_->rope_layer_ = std::make_shared<op::RoPELayer>(
      device_type_, config_->q_dim_, config_->kv_dim_, config_->head_size_);

  llama_layers_->mha_layer_ = std::make_shared<op::MultiHeadAttentionLayer>(
      device_type_, 0, config_->kv_mul_, config_->kv_dim_,
      config_->seq_len_, config_->head_num_, config_->head_size_);

  llama_layers_->add_layer_ = std::make_shared<op::VecAddLayer>(device_type_);

  llama_layers_->swiglu_layer_ =
      std::make_shared<op::SwiGLULayer>(device_type_);
}

void LLama2Model::create_param_quant_layers() {
  // Quantized model not yet supported
  CHECK(false) << "Quantized model is not supported yet.";
}

void LLama2Model::create_param_layers() {
  CHECK(!is_quant_model_);
  CHECK(llama_layers_ != nullptr);

  auto cpu_device_type = base::DeviceType_t::CPU;
  int32_t dim = config_->dim_;

  // Embedding layer
  llama_layers_->embedding_layer_ = std::make_shared<op::EmbeddingLayer>(
      device_type_, dim, config_->seq_len_, config_->vocab_size_);

  const void* weight_embedding = raw_model_data_->weight(0);
  std::dynamic_pointer_cast<op::LayerParam>(llama_layers_->embedding_layer_)
      ->set_weight(0, {static_cast<size_t>(config_->vocab_size_), static_cast<size_t>(dim)},
                   weight_embedding, cpu_device_type);

  // Skip embedding weight: vocab_size * dim, plus rmsnorm weights before QKV
  size_t pos = static_cast<size_t>(dim) * config_->vocab_size_ + dim * config_->layer_num_;

  // Wq layers
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wq = std::make_shared<op::MatmulLayer>(device_type_);
    wq->set_weight(0, {static_cast<size_t>(config_->q_dim_), static_cast<size_t>(dim)},
                   raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->wq_layers_.push_back(wq);
    pos += config_->q_dim_ * dim;
  }

  // Wk layers
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wk = std::make_shared<op::MatmulLayer>(device_type_);
    wk->set_weight(0, {static_cast<size_t>(config_->kv_dim_), static_cast<size_t>(dim)},
                   raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->wk_layers_.push_back(wk);
    pos += config_->kv_dim_ * dim;
  }

  // Wv layers
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wv = std::make_shared<op::MatmulLayer>(device_type_);
    wv->set_weight(0, {static_cast<size_t>(config_->kv_dim_), static_cast<size_t>(dim)},
                   raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->wv_layers_.push_back(wv);
    pos += config_->kv_dim_ * dim;
  }

  // Wo layers
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wo = std::make_shared<op::MatmulLayer>(device_type_);
    wo->set_weight(0, {static_cast<size_t>(dim), static_cast<size_t>(config_->q_dim_)},
                   raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->wo_layers_.push_back(wo);
    pos += dim * config_->q_dim_;
  }

  // Skip ffn rmsnorm
  pos += config_->layer_num_ * dim;

  // W1 layers (gate)
  int32_t hidden_dim = config_->hidden_dim_;
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto w1 = std::make_shared<op::MatmulLayer>(device_type_);
    w1->set_weight(0, {static_cast<size_t>(hidden_dim), static_cast<size_t>(dim)},
                   raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->w1_layers_.push_back(w1);
    pos += dim * hidden_dim;
  }

  // W2 layers (down)
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto w2 = std::make_shared<op::MatmulLayer>(device_type_);
    w2->set_weight(0, {static_cast<size_t>(dim), static_cast<size_t>(hidden_dim)},
                   raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->w2_layers_.push_back(w2);
    pos += dim * hidden_dim;
  }

  // W3 layers (up)
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto w3 = std::make_shared<op::MatmulLayer>(device_type_);
    w3->set_weight(0, {static_cast<size_t>(hidden_dim), static_cast<size_t>(dim)},
                   raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->w3_layers_.push_back(w3);
    pos += dim * hidden_dim;
  }

  // Skip final rmsnorm weight
  pos += dim;
  // Skip freqs_cos and freqs_sin weight (2 arrays)
  pos += 2 * config_->seq_len_ * config_->head_size_;

  // CLS layer
  llama_layers_->cls_layer_ =
      std::make_shared<op::MatmulLayer>(device_type_);
  if (config_->is_shared_weight_) {
    std::dynamic_pointer_cast<op::LayerParam>(llama_layers_->cls_layer_)
        ->set_weight(0, {static_cast<size_t>(config_->vocab_size_), static_cast<size_t>(dim)},
                     raw_model_data_->weight(0), cpu_device_type);
  } else {
    std::dynamic_pointer_cast<op::LayerParam>(llama_layers_->cls_layer_)
        ->set_weight(0, {static_cast<size_t>(config_->vocab_size_), static_cast<size_t>(dim)},
                     raw_model_data_->weight(pos), cpu_device_type);
  }

  // RMSNorm layers: attention rmsnorm (layer_num) + ffn rmsnorm (layer_num) + final (1)
  size_t rmsnorm_pos = static_cast<size_t>(dim) * config_->vocab_size_;
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto rms = std::make_shared<op::RmsNormLayer>(device_type_, dim);
    rms->set_weight(0, {static_cast<size_t>(dim)}, raw_model_data_->weight(rmsnorm_pos),
                    cpu_device_type);
    llama_layers_->rmsnorm_layers_.push_back(rms);
    rmsnorm_pos += dim;
  }

  // Skip attention QKV weights to reach FFN rmsnorm
  rmsnorm_pos += config_->layer_num_ * config_->q_dim_ * dim;    // wq
  rmsnorm_pos += config_->layer_num_ * config_->kv_dim_ * dim;  // wk
  rmsnorm_pos += config_->layer_num_ * config_->kv_dim_ * dim;  // wv
  rmsnorm_pos += config_->layer_num_ * dim * config_->q_dim_;    // wo

  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto rms = std::make_shared<op::RmsNormLayer>(device_type_, dim);
    rms->set_weight(0, {static_cast<size_t>(dim)}, raw_model_data_->weight(rmsnorm_pos),
                    cpu_device_type);
    llama_layers_->rmsnorm_layers_.push_back(rms);
    rmsnorm_pos += dim;
  }

  // Skip FFN weights to reach final rmsnorm
  rmsnorm_pos += config_->layer_num_ * hidden_dim * dim;   // w1
  rmsnorm_pos += config_->layer_num_ * dim * hidden_dim;   // w2
  rmsnorm_pos += config_->layer_num_ * hidden_dim * dim;   // w3

  auto rms_final = std::make_shared<op::RmsNormLayer>(device_type_, dim);
  rms_final->set_weight(0, {static_cast<size_t>(dim)},
                        raw_model_data_->weight(rmsnorm_pos), cpu_device_type);
  llama_layers_->rmsnorm_layers_.push_back(rms_final);
}

void LLama2Model::init_mem() {
  auto cpu_alloc = base::CPUDeviceControllerFactory::get_instance();

  // Sin/Cos cache for RoPE
  tensor::Tensor sin_cache(tensor::DataType_t::fp32,
                           {static_cast<size_t>(config_->head_size_ * config_->seq_len_)},
                           true, cpu_alloc);
  tensor::Tensor cos_cache(tensor::DataType_t::fp32,
                           {static_cast<size_t>(config_->head_size_ * config_->seq_len_)},
                           true, cpu_alloc);
  CHECK(insert_buffer(ModelBufferType::kSinCache, sin_cache));
  CHECK(insert_buffer(ModelBufferType::kCosCache, cos_cache));

  kernel::sin_cos_cache_calc_cpu(config_->head_size_, config_->seq_len_,
                                 static_cast<float*>(sin_cache.get_ptr()),
                                 static_cast<float*>(cos_cache.get_ptr()));

  // Input tokens and embeddings
  tensor::Tensor input_tokens(tensor::DataType_t::int32, {1}, true, cpu_alloc);
  tensor::Tensor input_embeddings(tensor::DataType_t::fp32,
                                  {1, static_cast<size_t>(config_->dim_)},
                                  true, cpu_alloc);
  CHECK(insert_buffer(ModelBufferType::kInputTokens, input_tokens));
  CHECK(insert_buffer(ModelBufferType::kInputEmbeddings, input_embeddings));

  // RMSNorm output
  tensor::Tensor rms_output(tensor::DataType_t::fp32, {static_cast<size_t>(config_->dim_)},
                            true, cpu_alloc);
  CHECK(insert_buffer(ModelBufferType::kOutputRMSNorm, rms_output));
  CHECK(insert_buffer(ModelBufferType::kW2Output, rms_output));
  CHECK(insert_buffer(ModelBufferType::kFFNRMSNorm, rms_output));

  // MHA output (before Wo projection)
  tensor::Tensor mha_output(tensor::DataType_t::fp32,
                            {static_cast<size_t>(config_->q_dim_)}, true, cpu_alloc);
  CHECK(insert_buffer(ModelBufferType::kOutputMHA, mha_output));

  // SwiGLU outputs
  tensor::Tensor w1_output(tensor::DataType_t::fp32,
                           {static_cast<size_t>(config_->hidden_dim_)}, true, cpu_alloc);
  tensor::Tensor w3_output(tensor::DataType_t::fp32,
                           {static_cast<size_t>(config_->hidden_dim_)}, true, cpu_alloc);
  CHECK(insert_buffer(ModelBufferType::kW1Output, w1_output));
  CHECK(insert_buffer(ModelBufferType::kW3Output, w3_output));

  // KV cache: [layer_num, seq_len, kv_dim]
  tensor::Tensor key_cache(tensor::DataType_t::fp32,
                           {static_cast<size_t>(config_->layer_num_),
                            static_cast<size_t>(config_->seq_len_),
                            static_cast<size_t>(config_->kv_dim_)},
                           true, cpu_alloc);
  tensor::Tensor value_cache(tensor::DataType_t::fp32,
                             {static_cast<size_t>(config_->layer_num_),
                              static_cast<size_t>(config_->seq_len_),
                              static_cast<size_t>(config_->kv_dim_)},
                             true, cpu_alloc);
  CHECK(insert_buffer(ModelBufferType::kKeyCache, key_cache));
  CHECK(insert_buffer(ModelBufferType::kValueCache, value_cache));

  // Query buffer
  tensor::Tensor query(tensor::DataType_t::fp32, {static_cast<size_t>(config_->q_dim_)},
                       true, cpu_alloc);
  CHECK(insert_buffer(ModelBufferType::kQuery, query));

  // Attention output (after Wo projection, back to residual dim)
  tensor::Tensor attn_output(tensor::DataType_t::fp32, {static_cast<size_t>(config_->dim_)},
                             true, cpu_alloc);
  CHECK(insert_buffer(ModelBufferType::kAttnOutput, attn_output));

  // Position tensor
  tensor::Tensor pos_tensor(tensor::DataType_t::int32, {1}, true, cpu_alloc);
  CHECK(insert_buffer(ModelBufferType::kInputPos, pos_tensor));

  // Attention scores
  tensor::Tensor attn(tensor::DataType_t::fp32,
                      {static_cast<size_t>(config_->head_num_),
                       static_cast<size_t>(config_->seq_len_)},
                      true, cpu_alloc);
  CHECK(insert_buffer(ModelBufferType::kScoreStorage, attn));

  // Forward output
  tensor::Tensor forward_output(tensor::DataType_t::fp32,
                                {static_cast<size_t>(config_->vocab_size_)},
                                true, cpu_alloc);
  CHECK(insert_buffer(ModelBufferType::kForwardOutput, forward_output));
}

base::error::Status LLama2Model::create_layers() {
  if (!llama_layers_) {
    llama_layers_ = std::make_unique<LLama2Layers>();
  }

  if (!is_quant_model_) {
    create_param_layers();
  } else {
    create_param_quant_layers();
  }
  create_nonparam_layers();

  if (!llama_layers_->embedding_layer_) {
    return base::error::Status(base::error::kInternalError,
                               "Create the embedding layer failed!");
  }

  if (llama_layers_->rmsnorm_layers_.size() !=
      static_cast<size_t>(2 * config_->layer_num_ + 1)) {
    return base::error::Status(base::error::kInternalError,
                               "Create the rmsnorm layers failed!");
  }

  return base::error::Status();
}

op::EmbeddingOutput LLama2Model::embedding(const std::vector<int>& tokens) {
  auto& input_tokens = get_buffer(ModelBufferType::kInputTokens);
  auto& input_embeddings = get_buffer(ModelBufferType::kInputEmbeddings);

  size_t token_count = tokens.size();
  if (input_tokens.get_size() != token_count) {
    input_tokens.reshape({token_count});
    input_embeddings.reshape({token_count, static_cast<size_t>(config_->dim_)});
  }

  for (size_t i = 0; i < token_count; ++i) {
    const_cast<int32_t&>(
        static_cast<const int32_t*>(input_tokens.get_ptr())[i]) = tokens.at(i);
  }

  auto cpu_alloc = base::CPUDeviceControllerFactory::get_instance();
  tensor::Tensor input_token_num(tensor::DataType_t::int32, {1}, true, cpu_alloc);
  *const_cast<int32_t*>(static_cast<const int32_t*>(input_token_num.get_ptr())) =
      static_cast<int32_t>(token_count);

  CHECK(llama_layers_->embedding_layer_ != nullptr);
  STATUS_CHECK(llama_layers_->embedding_layer_->forward(
      input_tokens, input_token_num, input_embeddings));

  op::EmbeddingOutput output(input_tokens, input_embeddings, input_token_num);
  return output;
}

void LLama2Model::attention_rms(int32_t layer_idx, const tensor::Tensor& input) {
  CHECK(llama_layers_ != nullptr);

  tensor::Tensor& rmsnorm_output = get_buffer(ModelBufferType::kOutputRMSNorm);
  const auto& rmsnorm_layer = llama_layers_->rmsnorm_layers_.at(layer_idx);
  CHECK(rmsnorm_layer != nullptr);
  STATUS_CHECK(rmsnorm_layer->forward(input, rmsnorm_output));
}

void LLama2Model::attention_qkv(int32_t layer_idx, const tensor::Tensor& pos_tensor) {
  CHECK(llama_layers_ != nullptr);

  tensor::Tensor& query = get_buffer(ModelBufferType::kQuery);
  tensor::Tensor& rmsnorm_output = get_buffer(ModelBufferType::kOutputRMSNorm);
  int32_t pos = static_cast<const int32_t*>(pos_tensor.get_ptr())[0];

  auto [key, val] = slice_kv_cache(layer_idx, pos);

  // Q projection
  const auto& query_layer = llama_layers_->wq_layers_.at(layer_idx);
  CHECK(query_layer != nullptr);
  STATUS_CHECK(query_layer->forward(rmsnorm_output, query));

  // K projection
  const auto& key_layer = llama_layers_->wk_layers_.at(layer_idx);
  CHECK(key_layer != nullptr);
  STATUS_CHECK(key_layer->forward(rmsnorm_output, key));

  // V projection
  const auto& value_layer = llama_layers_->wv_layers_.at(layer_idx);
  CHECK(value_layer != nullptr);
  STATUS_CHECK(value_layer->forward(rmsnorm_output, val));

  // RoPE
  CHECK(llama_layers_->rope_layer_ != nullptr);
  const tensor::Tensor& sin_cache = get_buffer(ModelBufferType::kSinCache);
  const tensor::Tensor& cos_cache = get_buffer(ModelBufferType::kCosCache);
  STATUS_CHECK(llama_layers_->rope_layer_->forward(
      query, key, pos_tensor, sin_cache, cos_cache, tensor::Tensor{}));
}

void LLama2Model::attention_mha(int32_t layer_idx, const tensor::Tensor& pos_tensor) {
  CHECK(llama_layers_ != nullptr);

  tensor::Tensor& query = get_buffer(ModelBufferType::kQuery);
  tensor::Tensor& key_cache = get_buffer(ModelBufferType::kKeyCache);
  tensor::Tensor& val_cache = get_buffer(ModelBufferType::kValueCache);
  tensor::Tensor& mha_output = get_buffer(ModelBufferType::kOutputMHA);
  tensor::Tensor& score_storage = get_buffer(ModelBufferType::kScoreStorage);

  int pos = static_cast<const int32_t*>(pos_tensor.get_ptr())[0];
  const auto& mha_layer = llama_layers_->mha_layer_;
  CHECK(mha_layer != nullptr);

  auto mha = std::dynamic_pointer_cast<op::MultiHeadAttentionLayer>(mha_layer);
  mha->set_pos(pos);
  mha->set_layer_index(layer_idx);
  STATUS_CHECK(mha_layer->forward(query, score_storage, key_cache, val_cache, mha_output));

  // Wo projection
  tensor::Tensor& attn_output = get_buffer(ModelBufferType::kAttnOutput);
  const auto& wo_layer = llama_layers_->wo_layers_.at(layer_idx);
  CHECK(wo_layer != nullptr);
  STATUS_CHECK(wo_layer->forward(mha_output, attn_output));
}

void LLama2Model::feed_forward(int32_t layer_idx, const tensor::Tensor& input) {
  CHECK(llama_layers_ != nullptr);

  // Residual add: input = input + attn_output
  tensor::Tensor& attn_output = get_buffer(ModelBufferType::kAttnOutput);
  CHECK(llama_layers_->add_layer_ != nullptr);
  STATUS_CHECK(llama_layers_->add_layer_->forward(input, attn_output, input));

  // FFN RMSNorm
  tensor::Tensor& ffn_norm_output = get_buffer(ModelBufferType::kFFNRMSNorm);
  const auto& ffn_rmsnorm = llama_layers_->rmsnorm_layers_.at(layer_idx + config_->layer_num_);
  CHECK(ffn_rmsnorm != nullptr);
  STATUS_CHECK(ffn_rmsnorm->forward(input, ffn_norm_output));

  // W1 (gate)
  tensor::Tensor& w1_output = get_buffer(ModelBufferType::kW1Output);
  const auto& w1_layer = llama_layers_->w1_layers_.at(layer_idx);
  CHECK(w1_layer != nullptr);
  STATUS_CHECK(w1_layer->forward(ffn_norm_output, w1_output));

  // W3 (up)
  tensor::Tensor& w3_output = get_buffer(ModelBufferType::kW3Output);
  const auto& w3_layer = llama_layers_->w3_layers_.at(layer_idx);
  CHECK(w3_layer != nullptr);
  STATUS_CHECK(w3_layer->forward(ffn_norm_output, w3_output));

  // SwiGLU
  CHECK(llama_layers_->swiglu_layer_ != nullptr);
  STATUS_CHECK(llama_layers_->swiglu_layer_->forward(w1_output, w3_output, w1_output));

  // W2 (down)
  tensor::Tensor& w2_output = get_buffer(ModelBufferType::kW2Output);
  const auto& w2_layer = llama_layers_->w2_layers_.at(layer_idx);
  CHECK(w2_layer != nullptr);
  STATUS_CHECK(w2_layer->forward(w1_output, w2_output));

  // Residual add: input = input + w2_output
  CHECK(llama_layers_->add_layer_ != nullptr);
  STATUS_CHECK(llama_layers_->add_layer_->forward(input, w2_output, input));
}

void LLama2Model::cls_logits(const tensor::Tensor& input) {
  CHECK(llama_layers_ != nullptr);

  const auto& norm = llama_layers_->rmsnorm_layers_.at(2 * config_->layer_num_);
  CHECK(norm != nullptr);

  tensor::Tensor& input_mut = const_cast<tensor::Tensor&>(input);
  STATUS_CHECK(norm->forward(input, input_mut));

  tensor::Tensor& forward_output = get_buffer(ModelBufferType::kForwardOutput);
  CHECK(llama_layers_->cls_layer_ != nullptr);
  STATUS_CHECK(llama_layers_->cls_layer_->forward(input, forward_output));
}

int32_t LLama2Model::post_processing(const tensor::Tensor& pos, bool is_prompt) {
  if (is_prompt) {
    return -1;
  }
  const tensor::Tensor& forward_output = get_buffer(ModelBufferType::kForwardOutput);
  const float* forward_logits = static_cast<const float*>(forward_output.get_ptr());

  return static_cast<int32_t>(sampler_->sample(forward_logits, forward_output.get_size()));
}

}  // namespace model
