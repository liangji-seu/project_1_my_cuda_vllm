#include "base/base.h"

namespace base {
namespace error {

Status::Status(int code, std::string err_message)
    : code_(code), message_(std::move(err_message)) {}

Status& Status::operator=(int code) {
    code_ = code;
    return *this;
}

bool Status::operator==(int code) const {
    return code_ == code;
}

bool Status::operator!=(int code) const {
    return code_ != code;
}

Status::operator int() const {
    return code_;
}

Status::operator bool() const {
    return code_ == StatusCode::kSuccess;
}

int32_t Status::get_err_code() const {
    return code_;
}

const std::string& Status::get_err_msg() const {
    return message_;
}

void Status::set_err_msg(const std::string& err_msg) {
    message_ = err_msg;
}

} // namespace error
} // namespace base
