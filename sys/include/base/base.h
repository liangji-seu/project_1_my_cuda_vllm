#pragma once

namespace base {

enum class DeviceType_t {
    Unknown = 0,
    CPU = 1,
    GPU = 2,
};

// 属性基类：禁止拷贝构造
class NoCopyable {
protected:
    NoCopyable()  = default;
    ~NoCopyable() = default;

public:
    NoCopyable(const NoCopyable&)            = delete;
    NoCopyable& operator=(const NoCopyable&) = delete;
};

} // namespace base
