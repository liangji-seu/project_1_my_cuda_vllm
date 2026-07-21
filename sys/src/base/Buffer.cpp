#include "base/Buffer.h"

namespace base {

//buffer创建对象，默认必须是内部自行申请内存。且立刻当场分配
Buffer::Buffer(size_t byte_size,
               void* ptr,
               
               DeviceType_t buffer_device_type,
               std::shared_ptr<DeviceController> controller,
               bool flag_is_external
               )
    : ptr(ptr),
      byte_size(byte_size),
      controller(controller),
      flag_is_external(flag_is_external) {
    if(!ptr && controller){
        device_type = controller->get_device_type();
        this->flag_is_external = false;
        this->ptr = controller->mem_alloc(byte_size);
    }
}


Buffer::~Buffer() {
    if(!flag_is_external){
        //自行分配的内存，当buffer结束，需要随着对象一起释放
        if(ptr && controller){
            //这个buffer里面有内存地址，且有设备控制器用于释放
            controller->mem_release(ptr);
            ptr = nullptr;
        }
    }
    //外部的，别的buffer的内存，不需要随着这个buffer结束而释放
}


//从另一个buffer里面拷贝内存过来
void Buffer::buffer_copy_from(const Buffer& buffer) {
    CHECK(controller != nullptr);
    CHECK(buffer.get_ptr() != nullptr);

    //只拷贝少数的那一个
    size_t copy_size = byte_size < (buffer.get_byte_size()) ? (byte_size):(buffer.get_byte_size());
    CHECK(device_type != DeviceType_t::Unknown);
    CHECK(buffer.get_device_type() != DeviceType_t::Unknown);
    byte_size = copy_size;
    return controller->mem_copy(buffer.get_ptr(), ptr, copy_size,
                                buffer.get_device_type(), device_type);

}

//重新分配新的byte_size个内存
void Buffer::buffer_self_allocate() {
    CHECK(controller != nullptr);
    CHECK(byte_size > 0);
    CHECK(flag_is_external == false);
    if(ptr){
        controller->mem_release(ptr);
    }
    ptr = controller->mem_alloc(byte_size);
    CHECK(ptr != nullptr) << "GPU/CPU memory allocation failed for " << byte_size << " bytes";
    return;
}

size_t Buffer::get_byte_size() const {
    return byte_size;
}

void* Buffer::get_ptr() const {
    return ptr;
}

DeviceType_t Buffer::get_device_type() const {
    return device_type;
}

void Buffer::set_device_type(DeviceType_t type) {
    device_type = type;
}

std::shared_ptr<DeviceController> Buffer::get_controller() const {
    return controller;
}

bool Buffer::get_flag_is_external() const {
    return flag_is_external;
}

} // namespace base
