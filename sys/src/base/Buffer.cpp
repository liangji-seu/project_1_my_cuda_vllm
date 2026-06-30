#include "base/Buffer.h"

namespace base {

Buffer::~Buffer() {
    // TODO
}

Buffer::Buffer(void* ptr,
               size_t byte_size,
               DeviceType_t buffer_device_type,
               std::shared_ptr<DeviceController> controller,
               bool flag_is_external)
    : ptr(ptr),
      byte_size(byte_size),
      buffer_device_type(buffer_device_type),
      controller(controller),
      flag_is_external(flag_is_external) {}

void Buffer::buffer_copy_from(const Buffer* buffer) const {
    // TODO
}

void Buffer::buffer_self_allocate() const {
    // TODO
}

size_t Buffer::get_byte_size() const {
    // TODO
    return 0;
}

void* Buffer::get_ptr() const {
    // TODO
    return nullptr;
}

DeviceType_t Buffer::get_buffer_device_type() const {
    // TODO
    return DeviceType_t::Unknown;
}

std::shared_ptr<DeviceController> Buffer::get_controller() const {
    // TODO
    return nullptr;
}

bool Buffer::get_flag_is_external() const {
    // TODO
    return false;
}

} // namespace base
