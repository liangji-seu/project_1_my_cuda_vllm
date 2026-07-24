#include "model/llama3.h"

#include <glog/logging.h>

#include "base/Buffer.h"
#include "base/DeviceController.h"
#include "profile/nvtx_utils.h"
#include "../op/kernel/cpu/rope_kernel.h"
#include "sampler/topk_sampler.h"
#include "sampler/argmax_sampler.h"
#include "sampler/gpu_argmax.h"

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

//模型超参config_在基类方法里面读出来
  auto read_status = gen_model_from_file();
  if (!read_status) {
    return read_status;
  }

  
  //在cpu测分配好内存空间（输入输出张量+cache）
  init_mem();

  // Set up CUDA stream and transfer to GPU
  if (device_type_ == base::DeviceType_t::GPU) {
    cuda_stream_ = std::make_shared<kernel::CudaStream>();
    set_cuda_stream_on_all_layers();
    transfer_to_device();
  }

  //构造采样器实例
  sampler_ = std::make_unique<sampler::ArgmaxSampler>(device_type_);
  // sampler_ = std::make_unique<sampler::TopKSampler>(device_type_, 0.6f, 20, 0.95f);
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

  // Extract pos value on CPU — pos_tensor stays on CPU, never goes to GPU.
  int32_t pos = static_cast<const int32_t*>(pos_tensor.get_ptr())[0];
  current_forward_pos_ = pos;
  uint32_t nvtx_color = is_prefill_phase_ ? profile::nvtx_color::kPrefill
                                          : profile::nvtx_color::kDecode;
  std::string phase = is_prefill_phase_ ? "prefill" + std::to_string(pos)
                                        : "decode"  + std::to_string(pos);

  bool layer_prof = (profiler_ && profiler_->layer_profile_enabled());
  cudaStream_t stream = cuda_stream_ ? cuda_stream_->stream : nullptr;
  const std::string stage = (pos < 1) ? "prefill" : "decode";

  for (int32_t layer_idx = 0; layer_idx < config_->layer_num_; ++layer_idx) {
    std::string layer_label = nvtx_context_ + "/" + phase + "/L" + std::to_string(layer_idx);
    NVTX_RANGE_C(layer_label.c_str(), nvtx_color);

    // ---- input_rmsnorm ----
    if (layer_prof) {
      cudaEvent_t ev_start, ev_stop;
      cudaEventCreate(&ev_start);
      cudaEventCreate(&ev_stop);
      cudaEventRecord(ev_start, stream);
      attention_rms(layer_idx, input);
      cudaEventRecord(ev_stop, stream);
      cudaEventSynchronize(ev_stop);
      float ms = 0;
      cudaEventElapsedTime(&ms, ev_start, ev_stop);
      cudaEventDestroy(ev_start);
      cudaEventDestroy(ev_stop);
      profiler_->add_layer_record("input_rmsnorm", stage, layer_idx, ms);
    } else {
      attention_rms(layer_idx, input);
    }

    // ---- qkv_projection (includes RoPE + KV cache write + optional QK norm) ----
    if (layer_prof) {
      cudaEvent_t ev_start, ev_stop;
      cudaEventCreate(&ev_start);
      cudaEventCreate(&ev_stop);
      cudaEventRecord(ev_start, stream);
      attention_qkv(layer_idx, pos);
      cudaEventRecord(ev_stop, stream);
      cudaEventSynchronize(ev_stop);
      float ms = 0;
      cudaEventElapsedTime(&ms, ev_start, ev_stop);
      cudaEventDestroy(ev_start);
      cudaEventDestroy(ev_stop);
      profiler_->add_layer_record("qkv_projection", stage, layer_idx, ms);
    } else {
      attention_qkv(layer_idx, pos);
    }

    // ---- attention (MHA + Wo projection) ----
    if (layer_prof) {
      cudaEvent_t ev_start, ev_stop;
      cudaEventCreate(&ev_start);
      cudaEventCreate(&ev_stop);
      cudaEventRecord(ev_start, stream);
      attention_mha(layer_idx, pos);
      cudaEventRecord(ev_stop, stream);
      cudaEventSynchronize(ev_stop);
      float ms = 0;
      cudaEventElapsedTime(&ms, ev_start, ev_stop);
      cudaEventDestroy(ev_start);
      cudaEventDestroy(ev_stop);
      profiler_->add_layer_record("attention", stage, layer_idx, ms);
    } else {
      attention_mha(layer_idx, pos);
    }

    // ---- feed_forward (FFN rmsnorm + gate/up/swiglu/down + residuals) ----
    if (layer_prof) {
      cudaEvent_t ev_start, ev_stop;
      cudaEventCreate(&ev_start);
      cudaEventCreate(&ev_stop);
      cudaEventRecord(ev_start, stream);
      feed_forward(layer_idx, input);
      cudaEventRecord(ev_stop, stream);
      cudaEventSynchronize(ev_stop);
      float ms = 0;
      cudaEventElapsedTime(&ms, ev_start, ev_stop);
      cudaEventDestroy(ev_start);
      cudaEventDestroy(ev_stop);
      profiler_->add_layer_record("mlp", stage, layer_idx, ms);
    } else {
      feed_forward(layer_idx, input);
    }
  }

  // ---- final_rmsnorm + lm_head ----
  if (layer_prof) {
    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);
    cudaEventRecord(ev_start, stream);
    cls_logits(input);
    cudaEventRecord(ev_stop, stream);
    cudaEventSynchronize(ev_stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, ev_start, ev_stop);
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
    profiler_->add_layer_record("final_rmsnorm+lm_head", stage, -1, ms);
  } else {
    cls_logits(input);
  }

  return base::error::Status();
}

base::error::Status LLama2Model::predict(const tensor::Tensor& input,
                                         const tensor::Tensor& pos_tensor,
                                         bool is_prompt, int& next) {
  is_prefill_phase_ = is_prompt;
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
      device_type_, dim, config_->vocab_size_);

  const void* weight_embedding = raw_model_data_->weight(0);
  std::dynamic_pointer_cast<op::LayerParam>(llama_layers_->embedding_layer_)
      ->set_weight(0, {static_cast<size_t>(config_->vocab_size_), static_cast<size_t>(dim)},
                   weight_embedding, cpu_device_type);

  // Skip embedding weight: vocab_size * dim, plus rmsnorm weights before QKV
  size_t pos = static_cast<size_t>(dim) * config_->vocab_size_ + dim * config_->layer_num_;

  bool has_bias = (config_->flags_ & FLAG_HAS_QKV_BIAS) != 0;

  // Wq layers (with optional bias)
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wq = std::make_shared<op::MatmulLayer>(device_type_, 1.0f, has_bias);
    wq->set_weight(0, {static_cast<size_t>(config_->q_dim_), static_cast<size_t>(dim)},
                   raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->wq_layers_.push_back(wq);
    pos += config_->q_dim_ * dim;
  }
  if (has_bias) {
    for (int32_t i = 0; i < config_->layer_num_; ++i) {
      auto wq = std::static_pointer_cast<op::MatmulLayer>(llama_layers_->wq_layers_[i]);
      wq->set_weight(1, {static_cast<size_t>(config_->q_dim_)},
                     raw_model_data_->weight(pos), cpu_device_type);
      pos += config_->q_dim_;
    }
  }

  // Wk layers (with optional bias)
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wk = std::make_shared<op::MatmulLayer>(device_type_, 1.0f, has_bias);
    wk->set_weight(0, {static_cast<size_t>(config_->kv_dim_), static_cast<size_t>(dim)},
                   raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->wk_layers_.push_back(wk);
    pos += config_->kv_dim_ * dim;
  }
  if (has_bias) {
    for (int32_t i = 0; i < config_->layer_num_; ++i) {
      auto wk = std::static_pointer_cast<op::MatmulLayer>(llama_layers_->wk_layers_[i]);
      wk->set_weight(1, {static_cast<size_t>(config_->kv_dim_)},
                     raw_model_data_->weight(pos), cpu_device_type);
      pos += config_->kv_dim_;
    }
  }

  // Wv layers (with optional bias)
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wv = std::make_shared<op::MatmulLayer>(device_type_, 1.0f, has_bias);
    wv->set_weight(0, {static_cast<size_t>(config_->kv_dim_), static_cast<size_t>(dim)},
                   raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->wv_layers_.push_back(wv);
    pos += config_->kv_dim_ * dim;
  }
  if (has_bias) {
    for (int32_t i = 0; i < config_->layer_num_; ++i) {
      auto wv = std::static_pointer_cast<op::MatmulLayer>(llama_layers_->wv_layers_[i]);
      wv->set_weight(1, {static_cast<size_t>(config_->kv_dim_)},
                     raw_model_data_->weight(pos), cpu_device_type);
      pos += config_->kv_dim_;
    }
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
  // Skip freqs_cos and freqs_sin weight (each [seq_len, head_size] — LLaMA-style RoPE)
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
  if (has_bias) {
    rmsnorm_pos += config_->layer_num_ * config_->q_dim_;         // Wq bias
  }
  rmsnorm_pos += config_->layer_num_ * config_->kv_dim_ * dim;  // wk
  if (has_bias) {
    rmsnorm_pos += config_->layer_num_ * config_->kv_dim_;        // Wk bias
  }
  rmsnorm_pos += config_->layer_num_ * config_->kv_dim_ * dim;  // wv
  if (has_bias) {
    rmsnorm_pos += config_->layer_num_ * config_->kv_dim_;        // Wv bias
  }
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

  // Q/K normalization weights — determined by explicit FLAG, NOT file-size heuristic
  if (config_->flags_ & FLAG_HAS_QK_NORM) {
    size_t qk_norm_pos = static_cast<size_t>(dim) * config_->vocab_size_  // embedding
        + dim * config_->layer_num_                                        // attn rmsnorm
        + config_->layer_num_ * config_->q_dim_ * dim                      // wq
        + config_->layer_num_ * config_->kv_dim_ * dim                     // wk
        + config_->layer_num_ * config_->kv_dim_ * dim                     // wv
        + config_->layer_num_ * dim * config_->q_dim_                      // wo
        + config_->layer_num_ * dim                                        // ffn rmsnorm
        + config_->layer_num_ * dim * hidden_dim                           // w1
        + config_->layer_num_ * dim * hidden_dim                           // w2
        + config_->layer_num_ * dim * hidden_dim                           // w3
        + dim                                                              // final rmsnorm
        + 2 * config_->seq_len_ * config_->head_size_;                    // freqs

    if (config_->flags_ & FLAG_HAS_QKV_BIAS) {
      qk_norm_pos += config_->layer_num_ * config_->q_dim_;             // Wq bias
      qk_norm_pos += config_->layer_num_ * config_->kv_dim_;            // Wk bias
      qk_norm_pos += config_->layer_num_ * config_->kv_dim_;            // Wv bias
    }
    if (!config_->is_shared_weight_) {
      qk_norm_pos += config_->vocab_size_ * dim;  // CLS weight
    }

    const int32_t head_size = config_->head_size_;
    for (int32_t i = 0; i < config_->layer_num_; ++i) {
      llama_layers_->q_norm_weights_.push_back(
          static_cast<const float*>(raw_model_data_->weight(qk_norm_pos)));
      qk_norm_pos += head_size;
    }
    for (int32_t i = 0; i < config_->layer_num_; ++i) {
      llama_layers_->k_norm_weights_.push_back(
          static_cast<const float*>(raw_model_data_->weight(qk_norm_pos)));
      qk_norm_pos += head_size;
    }
  }
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


//构造所有算子层实例
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






//执行embedding推理
op::EmbeddingOutput LLama2Model::embedding(const std::vector<int>& tokens) {
  std::string label = nvtx_context_ + "/embedding";
  NVTX_RANGE_C(label.c_str(), profile::nvtx_color::kDefault);
  auto& input_tokens = get_buffer(ModelBufferType::kInputTokens);
  auto& input_embeddings = get_buffer(ModelBufferType::kInputEmbeddings);

  size_t token_count = tokens.size();
  if (input_tokens.get_size() != token_count) {
    input_tokens.reshape({token_count});
    input_embeddings.reshape({token_count, static_cast<size_t>(config_->dim_)});
  }

  // Ensure input_tokens is on CPU so we can write token IDs
  if (input_tokens.get_device_type() == base::DeviceType_t::GPU) {
    input_tokens.to("cpu", nullptr);
  }

  for (size_t i = 0; i < token_count; ++i) {
    const_cast<int32_t&>(
        static_cast<const int32_t*>(input_tokens.get_ptr())[i]) = tokens.at(i);
  }

  auto cpu_alloc = base::CPUDeviceControllerFactory::get_instance();
  tensor::Tensor input_token_num(tensor::DataType_t::int32, {1}, true, cpu_alloc);
  *const_cast<int32_t*>(static_cast<const int32_t*>(input_token_num.get_ptr())) =
      static_cast<int32_t>(token_count);

  // Transfer input data to GPU for the embedding kernel
  if (device_type_ == base::DeviceType_t::GPU) {
    input_tokens.to("cuda", nullptr);
    input_token_num.to("cuda", nullptr);
  }

  CHECK(llama_layers_->embedding_layer_ != nullptr);
  STATUS_CHECK(llama_layers_->embedding_layer_->forward(
      input_tokens, input_token_num, input_embeddings));

  op::EmbeddingOutput output(input_tokens, input_embeddings, input_token_num);
  return output;
}

// 零拷贝 embedding: token_id 已经由 post_processing 直接写入 GPU kInputTokens
// 跳过 embedding() 中的 GPU→CPU→GPU 来回拷贝
op::EmbeddingOutput LLama2Model::embed_next_token(int32_t token_id) {
  auto& input_tokens = get_buffer(ModelBufferType::kInputTokens);
  auto& input_embeddings = get_buffer(ModelBufferType::kInputEmbeddings);

  if (input_tokens.get_size() != 1) {
    input_tokens.reshape({1});
  }
  if (input_embeddings.get_dim(0) != 1 ||
      static_cast<int32_t>(input_embeddings.get_dim(1)) != config_->dim_) {
    input_embeddings.reshape({1, static_cast<size_t>(config_->dim_)});
  }

  // 持久 GPU int32 buffer: 单 token decode 时 token_num 恒为 1
  // 避免每次 cudaMalloc + 同步 cudaMemcpy + cudaFree
  static int32_t* d_one = nullptr;
  if (!d_one && device_type_ == base::DeviceType_t::GPU) {
    cudaMalloc(&d_one, sizeof(int32_t));
    int32_t one = 1;
    cudaMemcpy(d_one, &one, sizeof(int32_t), cudaMemcpyHostToDevice);
  }

  auto cpu_alloc = base::CPUDeviceControllerFactory::get_instance();
  tensor::Tensor input_token_num(tensor::DataType_t::int32, {1});
  if (device_type_ == base::DeviceType_t::GPU) {
    // 直接包装持久 GPU buffer, 零拷贝
    auto cu_alloc = base::GPUDeviceControllerFactory::get_instance();
    auto cu_buf = std::make_shared<base::Buffer>(
        sizeof(int32_t), d_one, base::DeviceType_t::GPU, cu_alloc, true);
    input_token_num.assign(cu_buf);
  } else {
    input_token_num = tensor::Tensor(tensor::DataType_t::int32, {1}, true, cpu_alloc);
    *const_cast<int32_t*>(static_cast<const int32_t*>(input_token_num.get_ptr())) = 1;
  }

  CHECK(llama_layers_->embedding_layer_ != nullptr);
  STATUS_CHECK(llama_layers_->embedding_layer_->forward(
      input_tokens, input_token_num, input_embeddings));

  (void)token_id;
  op::EmbeddingOutput output(input_tokens, input_embeddings, input_token_num);
  return output;
}

//执行rmsnorm算子
void LLama2Model::attention_rms(int32_t layer_idx, const tensor::Tensor& input) {
  std::string phase = is_prefill_phase_ ? "prefill" + std::to_string(current_forward_pos_)
                                        : "decode"  + std::to_string(current_forward_pos_);
  std::string label = nvtx_context_ + "/" + phase + "/L" + std::to_string(layer_idx) + "/rmsnorm";
  uint32_t c = is_prefill_phase_ ? profile::nvtx_color::kPrefill : profile::nvtx_color::kDecode;
  NVTX_RANGE_C(label.c_str(), c);
  CHECK(llama_layers_ != nullptr);

  tensor::Tensor& rmsnorm_output = get_buffer(ModelBufferType::kOutputRMSNorm);
  const auto& rmsnorm_layer = llama_layers_->rmsnorm_layers_.at(layer_idx);
  CHECK(rmsnorm_layer != nullptr);
  STATUS_CHECK(rmsnorm_layer->forward(input, rmsnorm_output));
}


//执行qkv投影计算
void LLama2Model::attention_qkv(int32_t layer_idx, int32_t pos) {
  std::string phase = is_prefill_phase_ ? "prefill" + std::to_string(pos)
                                        : "decode"  + std::to_string(pos);
  std::string label = nvtx_context_ + "/" + phase + "/L" + std::to_string(layer_idx) + "/qkv";
  uint32_t c = is_prefill_phase_ ? profile::nvtx_color::kPrefill : profile::nvtx_color::kDecode;
  NVTX_RANGE_C(label.c_str(), c);
  CHECK(llama_layers_ != nullptr);

  tensor::Tensor& query = get_buffer(ModelBufferType::kQuery);
  tensor::Tensor& rmsnorm_output = get_buffer(ModelBufferType::kOutputRMSNorm);

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

  // Bias is now fused into the MatmulLayer itself — Wq/Wk/Wv forward()
  // already added bias, so query/key/val are fully computed at this point.

  // Q/K normalization (Qwen3 specific, Qwen2.x skips this).
  // Same pattern: CPU staging buffer + mem_copy, not to().
  if (!llama_layers_->q_norm_weights_.empty()) {
    auto add_qk_norm = [this](tensor::Tensor& t, const float* weight,
                               int32_t head_num, int32_t head_size) {
      if (device_type_ == base::DeviceType_t::GPU) {
        size_t byte_size = static_cast<size_t>(head_num) * head_size * sizeof(float);
        auto cpu_alloc = base::CPUDeviceControllerFactory::get_instance();
        auto cpu_buf = std::make_shared<base::Buffer>(
            byte_size, nullptr, base::DeviceType_t::Unknown, cpu_alloc, false);
        auto cu_alloc = base::GPUDeviceControllerFactory::get_instance();
        void* gpu_ptr = const_cast<void*>(t.get_ptr());
        cu_alloc->mem_copy(gpu_ptr, cpu_buf->get_ptr(), byte_size,
                           base::DeviceType_t::GPU, base::DeviceType_t::CPU);
        float* ptr = static_cast<float*>(cpu_buf->get_ptr());
        const float eps = 1e-6f;
        for (int32_t h = 0; h < head_num; ++h) {
          float* head = ptr + h * head_size;
          float sum_sq = 0.0f;
          for (int32_t d = 0; d < head_size; ++d) sum_sq += head[d] * head[d];
          float rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(head_size) + eps);
          for (int32_t d = 0; d < head_size; ++d) head[d] = head[d] * rms * weight[d];
        }
        cu_alloc->mem_copy(cpu_buf->get_ptr(), gpu_ptr, byte_size,
                           base::DeviceType_t::CPU, base::DeviceType_t::GPU);
      } else {
        float* ptr = static_cast<float*>(const_cast<void*>(t.get_ptr()));
        const float eps = 1e-6f;
        for (int32_t h = 0; h < head_num; ++h) {
          float* head = ptr + h * head_size;
          float sum_sq = 0.0f;
          for (int32_t d = 0; d < head_size; ++d) sum_sq += head[d] * head[d];
          float rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(head_size) + eps);
          for (int32_t d = 0; d < head_size; ++d) head[d] = head[d] * rms * weight[d];
        }
      }
    };
    add_qk_norm(query, llama_layers_->q_norm_weights_[layer_idx],
                config_->head_num_, config_->head_size_);
    add_qk_norm(key,   llama_layers_->k_norm_weights_[layer_idx],
                config_->kv_head_num_, config_->head_size_);
  }

  // RoPE
  CHECK(llama_layers_->rope_layer_ != nullptr);
  const tensor::Tensor& sin_cache = get_buffer(ModelBufferType::kSinCache);
  const tensor::Tensor& cos_cache = get_buffer(ModelBufferType::kCosCache);

  // Create a tensor for pos. Data stays on CPU (RoPE host code reads it
  // before kernel launch), but device_type must match the layer's type for check_tensor.
  {
    auto cpu_alloc = base::CPUDeviceControllerFactory::get_instance();
    tensor::Tensor pos_tensor_local(tensor::DataType_t::int32, {1}, true, cpu_alloc);
    *const_cast<int32_t*>(static_cast<const int32_t*>(pos_tensor_local.get_ptr())) = pos;
    pos_tensor_local.set_device_type(device_type_);
    STATUS_CHECK(llama_layers_->rope_layer_->forward(
        query, key, pos_tensor_local, sin_cache, cos_cache, tensor::Tensor{}));
  }
}





void LLama2Model::attention_mha(int32_t layer_idx, int32_t pos) {
  (void)pos;
  std::string phase = is_prefill_phase_ ? "prefill" + std::to_string(current_forward_pos_)
                                        : "decode"  + std::to_string(current_forward_pos_);
  std::string label = nvtx_context_ + "/" + phase + "/L" + std::to_string(layer_idx) + "/attn";
  uint32_t c = is_prefill_phase_ ? profile::nvtx_color::kPrefill : profile::nvtx_color::kDecode;
  NVTX_RANGE_C(label.c_str(), c);
  CHECK(llama_layers_ != nullptr);

  tensor::Tensor& query = get_buffer(ModelBufferType::kQuery);
  tensor::Tensor& key_cache = get_buffer(ModelBufferType::kKeyCache);
  tensor::Tensor& val_cache = get_buffer(ModelBufferType::kValueCache);
  tensor::Tensor& mha_output = get_buffer(ModelBufferType::kOutputMHA);
  tensor::Tensor& score_storage = get_buffer(ModelBufferType::kScoreStorage);

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
  std::string phase = is_prefill_phase_ ? "prefill" + std::to_string(current_forward_pos_)
                                        : "decode"  + std::to_string(current_forward_pos_);
  std::string label = nvtx_context_ + "/" + phase + "/L" + std::to_string(layer_idx) + "/mlp";
  uint32_t c = is_prefill_phase_ ? profile::nvtx_color::kPrefill : profile::nvtx_color::kDecode;
  NVTX_RANGE_C(label.c_str(), c);
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
  std::string phase = is_prefill_phase_ ? "prefill" + std::to_string(current_forward_pos_)
                                        : "decode"  + std::to_string(current_forward_pos_);
  std::string label = nvtx_context_ + "/" + phase + "/cls_logits";
  uint32_t c = is_prefill_phase_ ? profile::nvtx_color::kPrefill : profile::nvtx_color::kDecode;
  NVTX_RANGE_C(label.c_str(), c);
  CHECK(llama_layers_ != nullptr);

  const auto& norm = llama_layers_->rmsnorm_layers_.at(2 * config_->layer_num_);
  CHECK(norm != nullptr);

  tensor::Tensor& input_mut = const_cast<tensor::Tensor&>(input);
  STATUS_CHECK(norm->forward(input, input_mut));

  tensor::Tensor& forward_output = get_buffer(ModelBufferType::kForwardOutput);
  CHECK(llama_layers_->cls_layer_ != nullptr);
  STATUS_CHECK(llama_layers_->cls_layer_->forward(input, forward_output));
}

void LLama2Model::set_cuda_stream_on_all_layers() {
  CHECK(llama_layers_ != nullptr);
  CHECK(cuda_stream_ != nullptr);

  auto set_stream = [this](auto& layers) {
    for (auto& layer : layers) {
      if (layer) layer->set_cuda_stream(cuda_stream_);
    }
  };

  set_stream(llama_layers_->wq_layers_);
  set_stream(llama_layers_->wk_layers_);
  set_stream(llama_layers_->wv_layers_);
  set_stream(llama_layers_->wo_layers_);
  set_stream(llama_layers_->w1_layers_);
  set_stream(llama_layers_->w2_layers_);
  set_stream(llama_layers_->w3_layers_);
  set_stream(llama_layers_->rmsnorm_layers_);

  if (llama_layers_->cls_layer_)
    llama_layers_->cls_layer_->set_cuda_stream(cuda_stream_);
  if (llama_layers_->embedding_layer_)
    llama_layers_->embedding_layer_->set_cuda_stream(cuda_stream_);
  if (llama_layers_->rope_layer_)
    llama_layers_->rope_layer_->set_cuda_stream(cuda_stream_);
  if (llama_layers_->mha_layer_)
    llama_layers_->mha_layer_->set_cuda_stream(cuda_stream_);
  if (llama_layers_->add_layer_)
    llama_layers_->add_layer_->set_cuda_stream(cuda_stream_);
  if (llama_layers_->swiglu_layer_)
    llama_layers_->swiglu_layer_->set_cuda_stream(cuda_stream_);
}

void LLama2Model::transfer_to_device() {
  // Transfer compute buffers to GPU, but keep I/O buffers on CPU
  // (I/O buffers are transferred explicitly at kernel boundaries)
  for (auto& [type, tensor] : buffers_) {
    (void)type;
    if (type == ModelBufferType::kInputTokens ||
        type == ModelBufferType::kInputPos) {
      continue;  // I/O buffers: kept on CPU, transferred on demand
    }
    if (!tensor.is_empty() && tensor.get_device_type() == base::DeviceType_t::CPU) {
      tensor.to("cuda", nullptr);
    }
  }

  // Transfer all layer weights to GPU
  if (llama_layers_) {
    for (auto& wq : llama_layers_->wq_layers_)
      std::dynamic_pointer_cast<op::LayerParam>(wq)->to("cuda");
    for (auto& wk : llama_layers_->wk_layers_)
      std::dynamic_pointer_cast<op::LayerParam>(wk)->to("cuda");
    for (auto& wv : llama_layers_->wv_layers_)
      std::dynamic_pointer_cast<op::LayerParam>(wv)->to("cuda");
    for (auto& wo : llama_layers_->wo_layers_)
      std::dynamic_pointer_cast<op::LayerParam>(wo)->to("cuda");
    for (auto& w1 : llama_layers_->w1_layers_)
      std::dynamic_pointer_cast<op::LayerParam>(w1)->to("cuda");
    for (auto& w2 : llama_layers_->w2_layers_)
      std::dynamic_pointer_cast<op::LayerParam>(w2)->to("cuda");
    for (auto& w3 : llama_layers_->w3_layers_)
      std::dynamic_pointer_cast<op::LayerParam>(w3)->to("cuda");
    for (auto& rms : llama_layers_->rmsnorm_layers_)
      std::dynamic_pointer_cast<op::LayerParam>(rms)->to("cuda");
    if (llama_layers_->cls_layer_)
      std::dynamic_pointer_cast<op::LayerParam>(llama_layers_->cls_layer_)->to("cuda");
    if (llama_layers_->embedding_layer_)
      std::dynamic_pointer_cast<op::LayerParam>(llama_layers_->embedding_layer_)->to("cuda");
  }
}

int32_t LLama2Model::post_processing(const tensor::Tensor& pos, bool is_prompt) {
  if (is_prompt) {
    return -1;
  }
  tensor::Tensor& forward_output = get_buffer(ModelBufferType::kForwardOutput);

  if (forward_output.get_device_type() == base::DeviceType_t::GPU) {
    // 零拷贝闭环: GPU argmax → token_id 直接写入 GPU kInputTokens buffer
    // CPU 侧异步拷贝 (不 sync), 利用 demo 循环中的 cudaEventSynchronize 自然同步
    auto& input_tokens = get_buffer(ModelBufferType::kInputTokens);
    if (input_tokens.get_size() != 1) {
      input_tokens.reshape({1});
    }
    int32_t* token_gpu = const_cast<int32_t*>(
        static_cast<const int32_t*>(input_tokens.get_ptr()));

    const float* logits_gpu = static_cast<const float*>(forward_output.get_ptr());
    sampler::gpu_argmax_async(logits_gpu, forward_output.get_size(),
                              token_gpu, &async_next_token_,
                              cuda_stream_ ? cuda_stream_->stream : nullptr);
    return async_next_token_;
  }

  // CPU fallback
  const float* forward_logits = static_cast<const float*>(forward_output.get_ptr());
  return static_cast<int32_t>(
      sampler_->sample(forward_logits, forward_output.get_size()));
}

}  // namespace model
