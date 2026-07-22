#pragma once

// NVTX utilities for Nsight Systems profiling.
//
// Usage:
//   #include "profile/nvtx_utils.h"
//   {
//     NVTX_RANGE("prefill");
//     // ... GPU work ...
//   }  // range ends automatically (RAII)
//
// Controlled by CMake option ENABLE_NVTX. When disabled, all macros
// expand to nothing and no nvtx3 library is required.

#ifdef ENABLE_NVTX
#include <nvtx3/nvToolsExt.h>
#include <string>

namespace profile {

class NvtxRange {
 public:
  explicit NvtxRange(const std::string& name) {
    nvtxEventAttributes_t attr = {};
    attr.version = NVTX_VERSION;
    attr.size = NVTX_EVENT_ATTRIBUTES_STRUCT_SIZE;
    attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
    attr.message.ascii = name.c_str();
    nvtxRangePushEx(&attr);
  }

  ~NvtxRange() { nvtxRangePop(); }

  NvtxRange(const NvtxRange&) = delete;
  NvtxRange& operator=(const NvtxRange&) = delete;
};

}  // namespace profile

#define NVTX_RANGE(name) profile::NvtxRange _nvtx_ ## __LINE__(name)

#else  // !ENABLE_NVTX

// No-op when NVTX is disabled
namespace profile {
class NvtxRange {
 public:
  explicit NvtxRange(const char*) {}
};
}  // namespace profile

#define NVTX_RANGE(name) do { (void)(name); } while (0)

#endif  // ENABLE_NVTX
