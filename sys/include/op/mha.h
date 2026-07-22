#pragma once

#include "layer.h"
#include "base/base.h"

namespace op {

class MultiHeadAttentionLayer : public Layer {
private:
  int32_t layer_index_ = 0;//层号，
  int32_t pos_ = 0;//当前是第几个q向量
  int32_t kv_mul_ = 0;//GQA分组倍数，一个kv头配多少倍的Q头
  int32_t kv_dim_ = 0;//kv向量的维度
  int32_t seq_len_ = 0;//最大上下文长度
  int32_t head_num_ = 0;
  int32_t head_size_ = 0;

public:
  explicit MultiHeadAttentionLayer(
      base::DeviceType_t device_type,
      int32_t layer_index,
      int32_t kv_mul,
      int32_t kv_dim,
      int32_t seq_len,
      int32_t head_num,
      int32_t head_size);

  void set_pos(int32_t pos);
  void set_layer_index(int32_t layer_index);

  base::error::Status check_layer() override;

  base::error::Status forward() override;
};

}  // namespace op
