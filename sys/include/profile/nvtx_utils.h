#pragma once

#ifdef ENABLE_NVTX
#include <nvtx3/nvToolsExt.h>
#include <string>

namespace profile {

// ARGB color constants for NVTX ranges
namespace nvtx_color {
  constexpr uint32_t kPrefill  = 0xFF0066CC;   // blue — prefill
  constexpr uint32_t kDecode   = 0xFF00AA44;   // green — decode
  constexpr uint32_t kWarmup   = 0xFF888888;   // gray — warmup
  constexpr uint32_t kRun      = 0xFFFF6600;   // orange — formal run
  constexpr uint32_t kDefault  = 0x00000000;   // transparent (use GUI default)
}  // namespace nvtx_color

class NvtxRange {
 public:
  explicit NvtxRange(const char* name, uint32_t color = 0) {
    nvtxEventAttributes_t attr = {};
    attr.version = NVTX_VERSION;
    attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
    attr.message.ascii = name;
    if (color != 0) {
      attr.colorType = NVTX_COLOR_ARGB;
      attr.color = color;
    }
    nvtxRangePushEx(&attr);
  }

  ~NvtxRange() { nvtxRangePop(); }

  NvtxRange(const NvtxRange&) = delete;
  NvtxRange& operator=(const NvtxRange&) = delete;
};

}  // namespace profile

#define NVTX_RANGE(name)       profile::NvtxRange _nvtx_ ## __LINE__(name, 0)
#define NVTX_RANGE_C(name, c)  profile::NvtxRange _nvtx_ ## __LINE__(name, c)

#else  // !ENABLE_NVTX

namespace profile {
class NvtxRange {
 public:
  explicit NvtxRange(const char*, uint32_t = 0) {}
};
}  // namespace profile

#define NVTX_RANGE(name)       do { (void)(name); } while (0)
#define NVTX_RANGE_C(name, c)  do { (void)(name); (void)(c); } while (0)

#endif  // ENABLE_NVTX
