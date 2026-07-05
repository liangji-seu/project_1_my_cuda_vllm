#include "add_kernel.h"
#include <armadillo>//cpu的浮点线性代数运算


namespace kernel{

    //加法算子的CPU后端的内核
    void add_kernel_cpu(
        const tensor::Tensor& x1,
        const tensor::Tensor& x2,
        tensor::Tensor& y,
        void* stream
    ){
        (void)stream;//stream在cpu上没有用
        CHECK(x1.is_empty() == false);
        CHECK(x2.is_empty() == false);
        CHECK(y.is_empty() == false);

        CHECK(x1.get_size() == x2.get_size());
        CHECK(x1.get_size() == y.get_size());

        //开始加法运算， 用arma::fvec类来操作这块内存
        arma::fvec input_vec1((float*)(x1.get_ptr()), x1.get_size(), false, true);
        arma::fvec input_vec2((float*)(x2.get_ptr()), x2.get_size(), false, true);
        arma::fvec output_vec((float*)(y.get_ptr()), y.get_size(), false, true);
        output_vec = input_vec1 + input_vec2; 
    }
}