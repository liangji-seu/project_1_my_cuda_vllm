#pragma once

#include <cstddef>
#include <cstdint>

namespace model {

//用来整体存储我们mmap后的model.safetensor的.bin的地址信息
struct RawModelData {
  ~RawModelData();
  int32_t fd = -1;
  size_t file_size = 0;
  void* data = nullptr;//.bin的起始地址
  void* weight_data = nullptr;//权重的起始地址，跳过了开头

  virtual const void* weight(size_t offset) const = 0;
};


//不同精度的方式来读取参数
struct RawModelDataFp32 : RawModelData {
  const void* weight(size_t offset) const override;
};

struct RawModelDataInt8 : RawModelData {
  const void* weight(size_t offset) const override;
  const float* scale_data = nullptr;  // points to scale array after INT8 weights
  const float* scale(size_t offset) const { return scale_data + offset; }
};

}  // namespace model
