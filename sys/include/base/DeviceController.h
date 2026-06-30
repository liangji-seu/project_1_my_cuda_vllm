#pragma once

#include <cstddef>
#include <memory>

#include "base.h"

namespace base {



//基类
class DeviceController {
private:
    DeviceType_t device_type = DeviceType_t::Unknown;

public:
    explicit DeviceController(DeviceType_t device_type);
    virtual ~DeviceController() = default;

    virtual DeviceType_t show_device_type() const;

    // 分配具体大小内存
    virtual void* mem_alloc(size_t byte_size);

    // 释放内存（ptr为指向该块内存的指针）
    virtual void mem_release(void* ptr);

    // 拷贝内存
    virtual void mem_copy(const void* src,
                          void* dst,
                          size_t byte_size,
                          DeviceType_t src_device_type,
                          DeviceType_t dst_device_type,
                          void* stream = nullptr,
                          bool need_sync = false);

    // 置零内存
    virtual void mem_reset(void* ptr,
                           size_t byte_size,
                           DeviceType_t device_type,
                           void* stream = nullptr,
                           bool need_sync = false);
};



//子类：cpu内存控制器
class CPUDeviceController : public DeviceController {
public:
    explicit CPUDeviceController() = default;

    void* mem_alloc(size_t byte_size) override;
    void mem_release(void* ptr) override;
};


//子类：gpu显存控制器
class GPUDeviceController : public DeviceController {
public:
    explicit GPUDeviceController() = default;

    void* mem_alloc(size_t byte_size) override;
    void mem_release(void* ptr) override;
};

// CPU设备控制器工厂
class CPUDeviceControllerFactory {
private:
    static std::shared_ptr<CPUDeviceController> controller;

public:
    static std::shared_ptr<CPUDeviceController> get_instance();
};

// GPU设备控制器工厂
class GPUDeviceControllerFactory {
private:
    static std::shared_ptr<GPUDeviceController> controller;

public:
    static std::shared_ptr<GPUDeviceController> get_instance();
};

} // namespace base
