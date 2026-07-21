#pragma once

#include <cstddef>
#include <memory>
#include <map>
#include <vector>

#include "base.h"

namespace base {


    /**
     * 内存控制器层
     */

//基类
class DeviceController {
private:
    DeviceType_t device_type = DeviceType_t::Unknown; //设备类型

public:
    explicit DeviceController(DeviceType_t device_type);
    virtual ~DeviceController() = default;

    /**
     * 查询方法
     */
    virtual DeviceType_t get_device_type() const; //查询设备类型


    /**
     * 操作方法
     */
    // 分配具体大小内存
    virtual void* mem_alloc(size_t byte_size) = 0;

    // 释放内存（ptr为指向该块内存的指针）
    virtual void mem_release(void* ptr) = 0;

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
    explicit CPUDeviceController();

    //重写方法：分配，释放
    void* mem_alloc(size_t byte_size) override;
    void mem_release(void* ptr) override;
};






//子类：gpu显存控制器 (显存池)
#define THRESHOLD 1024*1024 
//大块内存，小块内存的区别， 1M左右
struct CudaMemoryBlock{
    //定义显存池里面的显存块 BCB
    void* ptr;       //地址
    size_t byte_size; //大小
    bool busy;

    CudaMemoryBlock()=default;
    CudaMemoryBlock(void* ptr,
                    size_t byte_size,
                    bool busy) :
                    ptr(ptr), byte_size(byte_size), busy(busy) {};
};

class GPUDeviceController : public DeviceController {

private:
//本质上这个内存池是一个初始为空，后面逐渐扩充的内存池
    std::map<int, size_t> idle_mini_block_size_cnt;
    std::map<int, std::vector<CudaMemoryBlock>> big_block_map; //大显存块表
    std::map<int, std::vector<CudaMemoryBlock>> mini_block_map;//小显存块表


public:
    explicit GPUDeviceController();

    //重写方法： 分配，释放
    void* mem_alloc(size_t byte_size) override;
    void mem_release(void* ptr) override;
};




/**
 * 工厂方法，返回一个静态的对象
 */

// CPU设备控制器工厂
class CPUDeviceControllerFactory {
private:
    static std::shared_ptr<CPUDeviceController> instance;

public:
    
    static std::shared_ptr<CPUDeviceController> get_instance();
};









// GPU设备控制器工厂
class GPUDeviceControllerFactory {
private:
    static std::shared_ptr<GPUDeviceController> instance;

public:
    static std::shared_ptr<GPUDeviceController> get_instance();
};

} // namespace base
