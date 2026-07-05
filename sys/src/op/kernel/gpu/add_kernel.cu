#include "add_kernel.cuh"
#include <cuda_runtime.h>


namespace kernel{


    //一个thread，负责一个元素的计算，朴素实现
    __global__ void add_kernel_cuda_fp32(
        size_t size, 
        const float* in1,
        const float* in2,
        const float* out
    ){
        //每个thread的全局id
        int32_t tid = blockDim.x*blockIdx.x + threadIdx.x;

        if(tid >= size){
            return;
        }

        float in_val1 = in1[tid];
        float in_val2 = in2[tid];
        out[tid] = in_val1 + in_val2;
    }





    void add_kernel_cuda(
        const tensor::Tensor& x1,
        const tensor::Tensor& x2,
        const tensor::Tensor& y,
        void* stream
    ){
        //先校验
        CHECK(x1.is_empty() == false);
        CHECK(x2.is_empty() == false);
        CHECK(y.is_empty() == false);

        size_t size = static_cast<size_t>(x1.get_size());
        CHECK(size== x2.get_size());
        CHECK(size== y.get_size());


        //开始规划thread资源
        size_t block_size = 512; //512 threads / block
        size_t grid_size = (size + block_size - 1) / block_size; //能覆盖size的thread

        //所以是朴素实现

        if(stream){
            cudaStream_t _stream = static_cast<cudaStream_t>(stream);

            add_kernel_cuda_fp32<<<grid_size, block_size, 0,_stream>>>(
                size,
                static_cast<cosnt float*>(x1.get_ptr()),
                static_cast<cosnt float*>(x2.get_ptr()),
                static_cast<cosnt float*>(y.get_ptr())
            );
        } else{
            add_kernel_cuda_fp32<<<grid_size, block_size>>>(
                size,
                static_cast<cosnt float*>(x1.get_ptr()),
                static_cast<cosnt float*>(x2.get_ptr()),
                static_cast<cosnt float*>(y.get_ptr())
            );
            
        }
        






    }









}
