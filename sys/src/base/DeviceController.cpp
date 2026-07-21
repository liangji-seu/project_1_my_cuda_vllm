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

//分配显存+缓存池（避免重复申请）
void* GPUDeviceController::mem_alloc(size_t byte_size) {
    //检查当前有无GPU
    int id = -1;
    CHECK(cudaGetDevice(&id) == cudaSuccess);

    if(byte_size > THRESHOLD){
        /**
         * 
         * 需要分配大块内存
         * 
         * 
        */

        //获取该设备上的大块内存的空闲表
        auto& big_block_list = big_block_map[id];

        int sel_id = -1;
        //选择合适的大块缓冲块
        for(int i =0; i<big_block_list.size(); i++){
            //若第i个大块block，仅支持分配一次byte_size, 分完变成小块
            if(big_block_list[i].byte_size >= byte_size &&
                !big_block_list[i].busy &&
                big_block_list[i].byte_size - byte_size < THRESHOLD){
                    if(sel_id == -1 ||
                        big_block_list[sel_id].byte_size > big_block_list[i].byte_size){
                        //这个第i个block更小且更贴合
                        sel_id = i;
                    }
                }
        }
        //现在已经确认，选择sel_id 这个大block来分配
        if(sel_id != -1){
            big_block_list[sel_id].busy = true;
            return big_block_list[sel_id].ptr;
        }

        //需要大内存，且控制器的大内存块的表里面没有合适的，就只能调用cuda驱动分配
        void* ptr = nullptr;
        if(cudaMalloc(&ptr, byte_size) != cudaSuccess){
            LOG(ERROR) << "Error: CUDA error when allocate " << byte_size << "byte memory";
            return nullptr;
        }
        //记录下重新分配的大块占用块
        big_block_list.emplace_back(ptr, byte_size, true);
        return ptr;
    }

        /**
         * 
         * 需要分配小块内存
         * 
         * 
        */
    auto& mini_block_list = mini_block_map[id];
    for(int i =0; i< mini_block_list.size(); i++){
        if(mini_block_list[i].byte_size >= byte_size &&
           !mini_block_list[i].busy){
            mini_block_list[i].busy = true;
            idle_mini_block_size_cnt[id] -= mini_block_list[i].byte_size;
            return mini_block_list[i].ptr;
           }
    }
    void*ptr = nullptr;
    if(cudaMalloc(&ptr, byte_size) != cudaSuccess){
        LOG(ERROR)<< "Error: CUDA error when allocate mini block";
        return nullptr;
    }
    mini_block_list.emplace_back(ptr, byte_size, true);
    return ptr;
}



void GPUDeviceController::mem_release(void* ptr) {
    CHECK(ptr != nullptr);
    CHECK(!mini_block_map.empty());

    cudaError_t state = cudaSuccess;
    for(auto& it : mini_block_map){
        //这一台设备的小块缓存已经超过了1G了， 才真正触发实际的释放，否则释放就只是标记不忙
        if(idle_mini_block_size_cnt[it.first] > THRESHOLD*1024){
            auto& mini_buffer_list = it.second;//这台设备的所有小块缓存
            std::vector<CudaMemoryBlock> temp;
            for(int i = 0; i < mini_buffer_list.size(); i++){
                if(!mini_buffer_list[i].busy){
                    //释放掉不忙的
                    state = cudaSetDevice(it.first);
                    state = cudaFree(mini_buffer_list[i].ptr);
                    CHECK(state == cudaSuccess);
                } else{
                    //重新保存忙的
                    temp.push_back(mini_buffer_list[i]);
                }
            }
            mini_buffer_list.clear();//清除旧缓存
            it.second = temp; //替换新缓存
            idle_mini_block_size_cnt[it.first] = 0;//重置
        }
    }

    for(auto& it: mini_block_map){
        auto& mini_block_list = it.second;
        for(int i = 0; i < mini_block_list.size(); i++){
            if(mini_block_list[i].ptr == ptr){
                idle_mini_block_size_cnt[it.first] += mini_block_list[i].byte_size;
                mini_block_list[i].busy = false;
                return;
            }
        }

        auto& big_block_list = big_block_map[it.first];
        for(int i = 0; i < big_block_list.size(); i++){
            if(big_block_list[i].ptr == ptr){
                big_block_list[i].busy = false;
                return;
            }
        }

    }

    state = cudaFree(ptr);
    CHECK(state == cudaSuccess);
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
