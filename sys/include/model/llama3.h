#pragma once

#include <memory>
#include <vector>

#include "model/model.h"
#include "op/add.h"
#include "op/embedding.h"
#include "op/matmul.h"
#include "op/mha.h"
#include "op/rmsnorm.h"
#include "op/rope.h"
#include "op/swiglu.h"

namespace model {

struct LLama2Layers {
  std::shared_ptr<op::Layer> add_layer_;
  std::shared_ptr<op::Layer> rope_layer_;
  std::shared_ptr<op::Layer> swiglu_layer_;
  std::shared_ptr<op::Layer> mha_layer_;

  std::vector<std::shared_ptr<op::Layer>> wq_layers_;
  std::vector<std::shared_ptr<op::Layer>> wk_layers_;
  std::vector<std::shared_ptr<op::Layer>> wv_layers_;
  std::vector<std::shared_ptr<op::Layer>> wo_layers_;

  std::vector<std::shared_ptr<op::Layer>> w1_layers_;
  std::vector<std::shared_ptr<op::Layer>> w2_layers_;
  std::vector<std::shared_ptr<op::Layer>> w3_layers_;
  std::vector<std::shared_ptr<op::Layer>> rmsnorm_layers_;

  std::shared_ptr<op::Layer> cls_layer_;
  std::shared_ptr<op::Layer> embedding_layer_;

  // Q/K normalization weights (Qwen3 specific)
  std::vector<const float*> q_norm_weights_;  // [head_size] per layer
  std::vector<const float*> k_norm_weights_;  // [head_size] per layer
};




/**
 * 模型子类
 */
class LLama2Model : public Model {
 private:
 //补充上具体的层算子实例
  std::shared_ptr<kernel::CudaStream> cuda_stream_;
  std::unique_ptr<LLama2Layers> llama_layers_;

  // NVTX context prefix for profiling labels
  std::string nvtx_context_;
  int32_t current_forward_pos_ = 0;
  bool is_prefill_phase_ = false;

 public:
  explicit LLama2Model(base::TokenizerType tokenizer_type, std::string token_path,
                       std::string model_path, bool is_quant_model);

  base::error::Status init(base::DeviceType_t device_type) override;

  base::error::Status predict(const tensor::Tensor& input,
                              const tensor::Tensor& pos_tensor,
                              bool is_prompt, int& next) override;

  base::error::Status forward(const tensor::Tensor& input,
                              const tensor::Tensor& pos_tensor,
                              int& next) override;

  op::EmbeddingOutput embedding(const std::vector<int>& tokens) override;

  // NVTX context label, set by benchmark/demo before each run.
  // e.g. "warmup1/prefill", "run3/decode"
  void set_nvtx_context(const std::string& label) { nvtx_context_ = label; }

 private:
  void init_mem() override;
  base::error::Status create_layers() override;
  void create_param_layers() override;
  void create_nonparam_layers() override;
  void create_param_quant_layers() override;

  void attention_rms(int32_t layer_idx, const tensor::Tensor& input);
  void attention_qkv(int32_t layer_idx, int32_t pos);
  void attention_mha(int32_t layer_idx, int32_t pos);
  void feed_forward(int32_t layer_idx, const tensor::Tensor& input);
  void cls_logits(const tensor::Tensor& input);

  int32_t post_processing(const tensor::Tensor& pos, bool is_prompt) override;

  void transfer_to_device();
  void set_cuda_stream_on_all_layers();
};

}  // namespace model
