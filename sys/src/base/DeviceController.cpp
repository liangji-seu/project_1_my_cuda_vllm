#include "base/DeviceController.h"
#include <cuda_runtime_api.h>
#include <cstdlib>

namespace base {

/**
 * 基类：DeviceController
 */

DeviceController::DeviceController(DeviceType_t device_type)
    : device_type(device_type) {}

//查询
DeviceType_t DeviceController::get_device_type() const {
    return device_type;
}


void DeviceController::mem_copy(const void* src, void* dst, size_t byte_size,
                                DeviceType_t src_device_type, DeviceType_t dst_device_type,
                                void* stream, bool need_sync) {
    //检查输入参数
    CHECK_NE(src, nullptr);
    CHECK_NE(dst, nullptr);
    if(byte_size <=0){
        return;
    }

    //设置默认cuda流
    cudaStream_t stream_ = nullptr;
    if(stream){
        stream_ = static_cast<cudaStream_t>(stream);
    }

    //执行内存拷贝
    if(src_device_type == DeviceType_t::CPU && dst_device_type == DeviceType_t::CPU){
        //CPU to CPU
        std::memcpy(dst, src, byte_size);
    } else if(src_device_type == DeviceType_t::CPU && dst_device_type == DeviceType_t::GPU){
        //CPU to GPU
        if(!stream_){
            //未指定cuda流
            cudaMemcpy(dst, src, byte_size, cudaMemcpyHostToDevice);
        } else{
            //指定了cuda流
            cudaMemcpyAsync(dst, src, byte_size, cudaMemcpyHostToDevice, stream_);
        }
    } else if(src_device_type == DeviceType_t::GPU && dst_device_type == DeviceType_t::CPU){
        //GPU to CPU
        if(!stream_){
            //未指定cuda流
            cudaMemcpy(dst, src, byte_size, cudaMemcpyDeviceToHost);
        } else{
            //指定了cuda流
            cudaMemcpyAsync(dst, src, byte_size, cudaMemcpyDeviceToHost, stream_);
        }

    } else if(src_device_type == DeviceType_t::GPU && dst_device_type == DeviceType_t::GPU){
        //GPU to GPU
        if(!stream_){
            //未指定cuda流
            cudaMemcpy(dst, src, byte_size, cudaMemcpyDeviceToDevice);
        } else{
            //指定了cuda流
            cudaMemcpyAsync(dst, src, byte_size, cudaMemcpyDeviceToDevice, stream_);
        }
    } else{
        LOG(FATAL) << "Unknown memcpy kind: " << int(src_device_type) << " to " << int(dst_device_type);
    }

    if(need_sync){
        cudaDeviceSynchronize();//等待所有cuda工作流的任务都执行完成
    }


}

void DeviceController::mem_reset(void* ptr, size_t byte_size, DeviceType_t device_type,
                                 void* stream, bool need_sync) {
    CHECK(device_type != DeviceType_t::Unknown);
    if(ptr == nullptr){
        LOG(FATAL) << "ptr is nullptr";
        return;
    }
    
    
    if(device_type == DeviceType_t::CPU){
        std::memset(ptr, 0, byte_size);
    } else if (device_type == DeviceType_t::GPU){
        if(stream){
            cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
            cudaMemsetAsync(ptr, 0, byte_size, stream_);
        }else{
            cudaMemset(ptr, 0, byte_size);
        }
        
        if(need_sync){
            cudaDeviceSynchronize();
        }
    } else {
        LOG(FATAL) << "error device_type";
        return;
    }
}












/**
 * 子类：CPUDeviceController
 */

CPUDeviceController::CPUDeviceController() : DeviceController(DeviceType_t::CPU) {}

 //cpu分配多少内存，直接调用malloc
void* CPUDeviceController::mem_alloc(size_t byte_size) {
    CHECK(byte_size > 0);
    return (void*)malloc(byte_size);
}

void CPUDeviceController::mem_release(void* ptr) {
    if(ptr != nullptr){
        free(ptr);
    }
}








// ---- GPUDeviceController ----

GPUDeviceController::GPUDeviceController() : DeviceController(DeviceType_t::GPU){}

//朴素分配显存：直接调用 cudaMalloc
void* GPUDeviceController::mem_alloc(size_t byte_size) {
    //检查当前有无GPU
    int id = -1;
    CHECK(cudaGetDevice(&id) == cudaSuccess);

    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, byte_size);
    CHECK(err == cudaSuccess) << "CUDA malloc failed for " << byte_size
                              << " bytes: " << cudaGetErrorString(err);
    return ptr;
}

//朴素释放显存：直接调用 cudaFree
void GPUDeviceController::mem_release(void* ptr) {
    CHECK(ptr != nullptr);
    cudaError_t state = cudaFree(ptr);
    CHECK(state == cudaSuccess) << "CUDA free failed: " << cudaGetErrorString(state);
}

// ---- CPUDeviceControllerFactory ----

std::shared_ptr<CPUDeviceController> CPUDeviceControllerFactory::instance = nullptr;

std::shared_ptr<CPUDeviceController> CPUDeviceControllerFactory::get_instance() {
    if (instance == nullptr) {
        instance = std::make_shared<CPUDeviceController>();
    }
    return instance;
}

// ---- GPUDeviceControllerFactory ----

std::shared_ptr<GPUDeviceController> GPUDeviceControllerFactory::instance = nullptr;

std::shared_ptr<GPUDeviceController> GPUDeviceControllerFactory::get_instance() {
    if (instance == nullptr) {
        instance = std::make_shared<GPUDeviceController>();
    }
    return instance;
}

} // namespace base
