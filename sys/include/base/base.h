#pragma once

#include <cstdint>
#include <cstdio>
#include <string>

#include <glog/logging.h>

namespace base {

enum class DeviceType_t {
    Unknown = 0,
    CPU = 1,
    GPU = 2,
};

enum class ModelType : uint8_t {
    kModelTypeUnknown = 0,
    kModelTypeLLama2 = 1,
};

enum class TokenizerType {
    kEncodeUnknown = -1,
    kEncodeSpe = 0,
    kEncodeBpe = 1,
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




/**
 * 
 * 错误检测模块
 */
namespace error {

enum StatusCode : uint8_t {
    kSuccess              = 0,
    kFunctionUnImplement   = 1,
    kPathNotValid          = 2,
    kModelParseError       = 3,
    kInternalError         = 5,
    kKeyValueHasExist      = 6,
    kInvalidArgument       = 7,
};

class Status {
public:
    Status(int code = StatusCode::kSuccess, std::string err_message = "");

    Status(const Status& other) = default;

    Status& operator=(const Status& other) = default;

    Status& operator=(int code);

    bool operator==(int code) const;

    bool operator!=(int code) const;

    operator int() const;

    operator bool() const;

    int32_t get_err_code() const;

    const std::string& get_err_msg() const;

    void set_err_msg(const std::string& err_msg);

private:
    int code_ = StatusCode::kSuccess;
    std::string message_;
};

} // namespace error





} // namespace base

#define STATUS_CHECK(call)                                                                     \
  do {                                                                                         \
    const base::error::Status& status = call;                                                  \
    if (!status) {                                                                             \
      const size_t buf_size = 512;                                                             \
      char buf[buf_size];                                                                      \
      snprintf(buf, buf_size - 1,                                                              \
               "Infer error\n File:%s Line:%d\n Error code:%d\n Error msg:%s\n", __FILE__,     \
               __LINE__, int(status), status.get_err_msg().c_str());                           \
      LOG(FATAL) << buf;                                                                       \
    }                                                                                          \
  } while (0)
