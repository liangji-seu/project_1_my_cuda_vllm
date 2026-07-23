#pragma once
#include <cstddef>
#include <cstdint>

namespace sampler {

/**
 * GPU argmax (异步, 不阻塞): 在GPU上找到logits中最大值的索引
 *
 * 将 token_id(int32) 直接写入 GPU buffer (供下一轮 embedding 使用),
 * 同时异步拷贝到 CPU (供显示解码, 调用方在需要时自行 cudaStreamSynchronize)
 *
 * @param logits_gpu  GPU上的logits数组
 * @param size        vocab_size
 * @param token_gpu   输出: GPU上写入 token_id 的位置
 * @param token_cpu   输出: CPU上接收 token_id (可为nullptr跳过CPU拷贝)
 * @param stream      CUDA stream
 */
void gpu_argmax_async(const float* logits_gpu, size_t size,
                      int32_t* token_gpu, int32_t* token_cpu, void* stream);

}  // namespace sampler
