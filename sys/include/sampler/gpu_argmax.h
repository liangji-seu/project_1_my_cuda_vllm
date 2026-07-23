#pragma once
#include <cstddef>

namespace sampler {

/**
 * GPU argmax: 在GPU上找到logits中最大值的索引
 * 只将结果(4字节)拷回CPU, 避免整个vocab_size的logits拷贝
 *
 * @param logits_gpu  GPU上的logits数组指针
 * @param size        vocab_size
 * @param stream      CUDA stream
 * @return            最大值的索引
 */
size_t gpu_argmax(const float* logits_gpu, size_t size, void* stream = nullptr);

}  // namespace sampler
