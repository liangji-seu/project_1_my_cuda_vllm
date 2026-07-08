#include "kernel_interface.h"
#include "glog/logging.h"
#include "cpu/add_kernel.h"
#include "cpu/emb_kernel.h"
#include "cpu/rmsnorm_kernel.h"
#include "cpu/matmul_kernel.h"
#include "cpu/mha_kernel.h"
#include "gpu/add_kernel.cuh"
#include "gpu/emb_kernel.cuh"
#include "gpu/rmsnorm_kernel.cuh"
#include "gpu/matmul_kernel.cuh"
#include "gpu/mha_kernel.cuh"


namespace kernel{

//加法
Add_backend get_add_interface(base::DeviceType_t device_type){
    if(device_type == base::DeviceType_t::CPU){
        return add_kernel_cpu;
    } else if(device_type == base::DeviceType_t::GPU){
        return add_kernel_cuda;
    } else {
        LOG(ERROR)<<"error device type";
        return nullptr;
    }
}

//词嵌入层
Embedding_backend get_emb_interface(base::DeviceType_t device_type){
    if(device_type == base::DeviceType_t::CPU){
        return emb_kernel_cpu;
    } else if(device_type == base::DeviceType_t::GPU){
        return emb_kernel_cuda;
    } else {
        LOG(ERROR)<<"error device type";
        return nullptr;
    }
}

//RMSNorm层
RMSNorm_backend get_rmsnorm_interface(base::DeviceType_t device_type){
    if(device_type == base::DeviceType_t::CPU){
        return rmsnorm_kernel_cpu;
    } else if(device_type == base::DeviceType_t::GPU){
        return rmsnorm_kernel_cuda;
    } else {
        LOG(ERROR)<<"error device type";
        return nullptr;
    }
}

//矩阵乘层
Matmul_backend get_matmul_interface(base::DeviceType_t device_type){
    if(device_type == base::DeviceType_t::CPU){
        return matmul_kernel_cpu;
    } else if(device_type == base::DeviceType_t::GPU){
        return matmul_kernel_cuda;
    } else {
        LOG(ERROR)<<"error device type";
        return nullptr;
    }
}

//MHA层
MHA_backend get_mha_interface(base::DeviceType_t device_type){
    if(device_type == base::DeviceType_t::CPU){
        return mha_kernel_cpu;
    } else if(device_type == base::DeviceType_t::GPU){
        return mha_kernel_cuda;
    } else {
        LOG(ERROR)<<"error device type";
        return nullptr;
    }
}

} // namespace kernel