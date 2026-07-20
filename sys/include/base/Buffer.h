#pragma once

#include <cstddef>
#include <memory>

#include "base.h"
#include "base/DeviceController.h"

namespace base {

class Buffer : public NoCopyable {
private:
    void* ptr                                          = nullptr;             // 内存地址
    size_t byte_size                                   = 0;                   // 内存大小
    DeviceType_t device_type                           = DeviceType_t::Unknown; // 内存的设备类型
    std::shared_ptr<DeviceController> controller;                             // 设备控制器
    bool flag_is_external                              = false;                // true:外部指定内存  false:控制器开辟内存

public:
    explicit Buffer() = default;
    ~Buffer();

    explicit Buffer( size_t byte_size,
                    void* ptr                                              = nullptr,
                    DeviceType_t buffer_device_type                        = DeviceType_t::Unknown,
                    std::shared_ptr<DeviceController> controller           = nullptr,
                    bool flag_is_external                                  = false
                                                          );

    /*操作方法*/
    void buffer_copy_from(const Buffer& buffer);                          // 从目标buffer内存，拷贝固定byte_size长度的内存到本buffer
    void buffer_self_allocate() ;                                          // 开始自行调用控制器开辟内存

    /*查询方法*/
    size_t get_byte_size() const;                                               // 获取内存大小
    void* get_ptr() const;                                                      // 获取内存地址
    DeviceType_t get_device_type() const;                                // 获取内存的设备类型
    void set_device_type(DeviceType_t type);                             // 设置内存的设备类型
    std::shared_ptr<DeviceController> get_controller() const;                   // 获取设备控制器
    bool get_flag_is_external() const;                                          // 获取外部内存标志
};

} // namespace base
