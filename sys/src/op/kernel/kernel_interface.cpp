#include "kernel_interface.h"
#include "glog/logging.h"
#include "cpu/add_kernel.h"
#include "gpu/add_kernel.cuh"


namespace kernel{

//全连接层
//分词器层
//词嵌入层
//RMSNorm， 层归一化层
//矩阵乘
//旋转位置编码， RoPE
//多头注意力层MHA
//softmax
//加法
Add_backend get_add_interface(base::DeviceType_t device_type){
    if(device_type == base::DeviceType_t::CPU){
        //返回cpu后端
        return add_kernel_cpu;
    } else if(device_type == base::DeviceType_t::GPU){
        //返回GPU后端
        return add_kernel_cuda;
    } else {
        LOG(ERROR)<<"error device type";
        return nullptr;
    }
}
//SwiGLU 激活函数

} // namespace kernel