#pragma once

#include "layer.h"
#include "base/base.h"

namespace op{

struct EmbeddingOutput {
  tensor::Tensor input_tokens;
  tensor::Tensor input_embeddings;
  tensor::Tensor input_token_num;

  explicit EmbeddingOutput(tensor::Tensor input_tokens,
                           tensor::Tensor input_embeddings,
                           tensor::Tensor input_token_num)
      : input_tokens(std::move(input_tokens)),
        input_embeddings(std::move(input_embeddings)),
        input_token_num(std::move(input_token_num)) {}
};


/**
 * 词嵌入算子层，输入tokenid = one-hot编码，对权重张量查表得到token向量（语义空间）
 * 
 * 权重张量（vocab_size x dim）
 * 
 */
class EmbeddingLayer : public LayerParam {
private:
  int32_t dim_ = 0;//模型维度
  int32_t vocab_size_ = 0;//词表大小

public:
  explicit EmbeddingLayer(
      base::DeviceType_t device_type,
      int32_t dim,
      int32_t vocab_size);

  base::error::Status check_layer() override;

  base::error::Status forward() override;
};

}  // namespace op
