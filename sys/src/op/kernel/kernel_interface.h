#pragma once

#include "base/cuda_stream.h"
#include "base/base.h"
#include "tensor/tensor.h"





/*

    //不同层的类型的注册
    enum class LayerType_t : uint8_t {
        Unknown = 0,
        Linear = 1,
        Encode = 2,
        Embedding = 3,
        RMSNorm = 4,
        Matmul = 5,
        RoPe = 6,
        MHA = 7,
        Softmax = 8,
        Add = 9,
        SwiGLU = 10,
    };
    

*/

/**
 * 这个kernel_interface, 作为算子层前端访问算子层后端内核的接口
 * 实现双后端的映射
 */


namespace kernel{


/**
 * 定义算子后端接口
 */



//全连接层
//输入张量1个，输出张量1个，权重张量2个（W + B）
typedef void (*Linear_backend)(
    const tensor::Tensor& x1,
    const tensor::Tensor& x2,
    const tensor::Tensor& w,
    const tensor::Tensor& b,
    const tensor::Tensor& y,
    void* stream
);



//分词器层
// typedef void (*Encode_backend){

// }

//词嵌入层
//输入一个张量，输出一个张量，权重张量1个
typedef void (*Embedding_backend)(
    const tensor::Tensor& x,
    const tensor::Tensor& w,
    const tensor::Tensor& y,
    size_t vocab_size, //词袋大小
    void* stream
);

Embedding_backend get_emb_interface(base::DeviceType_t device_type);


//RMSNorm， 层归一化层
//输入一个张量，输出一个张量，权重张量1个
typedef void (*RMSNorm_backend)(
    const tensor::Tensor& x,
    const tensor::Tensor& w,
    const tensor::Tensor& y,
    void* stream
);

RMSNorm_backend get_rmsnorm_interface(base::DeviceType_t device_type);


//矩阵乘
typedef void (*Matmul_backend)(
    const tensor::Tensor& x,
    const tensor::Tensor& w,
    float scale,
    const tensor::Tensor& y,
    void* stream
);

Matmul_backend get_matmul_interface(base::DeviceType_t device_type);


//旋转位置编码， RoPE
typedef void (*RoPE_backend)(
    int32_t dim,
    int32_t kv_dim,
    int32_t head_size,
    const tensor::Tensor& input_q,
    const tensor::Tensor& input_k,
    const tensor::Tensor& input_pos,
    const tensor::Tensor& sin_cache,
    const tensor::Tensor& cos_cache,
    void* stream
);

RoPE_backend get_rope_interface(base::DeviceType_t device_type);

//多头注意力层MHA
typedef void (*MHA_backend)(
    int32_t pos,
    int32_t head_num,
    int32_t layer_index,
    int32_t seq_len,
    int32_t kv_dim,
    int32_t kv_mul,
    int32_t head_size,
    const tensor::Tensor& mha_out,
    const tensor::Tensor& query_tensor,
    const tensor::Tensor& score_tensor,
    const tensor::Tensor& key_cache_tensor,
    const tensor::Tensor& value_cache_tensor,
    void* stream
);

MHA_backend get_mha_interface(base::DeviceType_t device_type);

//softmax
typedef void (Softmax_backend)(

);

//加法
typedef void (*Add_backend)(
    const tensor::Tensor& x1,
    const tensor::Tensor& x2,
    tensor::Tensor& y,
    void* stream
);

//SwiGLU 激活函数
typedef void (*SwiGLU_backend)(
    const tensor::Tensor& x,
    const tensor::Tensor& y,
    void* stream
);



/**
 * 
 * 返回真正合适的后端方法。
 */


//全连接层
//分词器层
//词嵌入层
//RMSNorm， 层归一化层
//矩阵乘
//旋转位置编码， RoPE
//多头注意力层MHA
//softmax
//加法
Add_backend get_add_interface(base::DeviceType_t device_type);
//SwiGLU 激活函数

}