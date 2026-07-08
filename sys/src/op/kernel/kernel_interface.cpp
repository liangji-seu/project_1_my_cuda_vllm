#include "kernel_interface.h"
#include "glog/logging.h"
#include "cpu/add_kernel.h"
#include "cpu/emb_kernel.h"
#include "cpu/rmsnorm_kernel.h"
#include "cpu/matmul_kernel.h"
#include "cpu/softmax_kernel.h"
#include "cpu/rope_kernel.h"
#include "cpu/mha_kernel.h"
#include "gpu/add_kernel.cuh"
#include "gpu/emb_kernel.cuh"
#include "gpu/rmsnorm_kernel.cuh"
#include "gpu/matmul_kernel.cuh"
#include "gpu/softmax_kernel.cuh"
#include "gpu/rope_kernel.cuh"
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

//RoPE层
RoPE_backend get_rope_interface(base::DeviceType_t device_type){
    if(device_type == base::DeviceType_t::CPU){
        return rope_kernel_cpu;
    } else if(device_type == base::DeviceType_t::GPU){
        return rope_kernel_cuda;
    } else {
        LOG(ERROR)<<"error device type";
        return nullptr;
    }
}

//Softmax层
Softmax_backend get_softmax_interface(base::DeviceType_t device_type){
    if(device_type == base::DeviceType_t::CPU){
        return softmax_inplace_cpu;
    } else if(device_type == base::DeviceType_t::GPU){
        return softmax_inplace_cuda;
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