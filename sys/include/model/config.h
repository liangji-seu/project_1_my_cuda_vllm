#pragma once

#include <cstdint>

namespace model {

// .bin file header (legacy format + flags extension)
struct ModelConfig {
  int32_t dim = 0;//模型维度
  int32_t hidden_dim = 0;//up层维度
  int32_t layer_num = 0;//层数
  int32_t head_num = 0;//q头数
  int32_t kv_head_num = 0;//kv头数
  int32_t vocab_size = 0;  // positive = tied embeddings（lm_head复用权重）, negative = separate
  int32_t seq_len = 0;//最大上下文长度
  int32_t head_dim = 0;//头维
};

// Flags bitfield (4 bytes, immediately after ModelConfig in .bin)
// These REPLACE all heuristics — every structural fact is explicit.
//.bin文件的头部布局中的标志位区域
enum ModelFlags : int32_t {
  FLAG_NONE          = 0,
  FLAG_HAS_QKV_BIAS  = 1 << 0,  // q_proj/k_proj/v_proj have bias
  FLAG_HAS_QK_NORM   = 1 << 1,  // q_norm/k_norm present (Qwen3)
  FLAG_HAS_O_BIAS    = 1 << 2,  // o_proj has bias
  FLAG_HAS_MLP_BIAS  = 1 << 3,  // mlp gate/up/down have bias
  FLAG_TIED_WEIGHTS  = 1 << 4,  // lm_head shares embedding weight
};


//综合计算后的我们需要的完整参数，模型的超参直接描述
struct TransformerConfig {
  int32_t kv_dim_ = 0;//kv维数
  int32_t kv_mul_ = 0;//gqa的倍率
  int32_t head_size_ = 0;//头维
  int32_t vocab_size_ = 0;//词表大小
  int32_t q_dim_ = 0;//q维数

  int32_t dim_ = 0;//模型维数
  int32_t hidden_dim_ = 0;//up向量的维数
  int32_t layer_num_ = 0;//层数
  int32_t head_num_ = 0;//q头数
  int32_t kv_head_num_ = 0;//kv头数
  int32_t seq_len_ = 0;//最大上下文长度
  bool is_shared_weight_ = false;//lm_head和embedding共享权重

  int32_t flags_ = FLAG_NONE;  // ModelFlags bitfield
};

}  // namespace model
