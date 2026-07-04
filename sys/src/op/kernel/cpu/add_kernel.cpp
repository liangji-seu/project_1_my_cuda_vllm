#include "add_kernel.h"
#include <armadillo>//cpu的浮点线性代数运算


namespace kernel{

    //加法算子的CPU后端的内核
    void add_kernel_cpu(
        const tensor::Tensor& x1,
        const tensor::Tensor& x2,
        const tensor::Tensor& y,
        void* stream
    ){
        UNUSED(stream);//stream在cpu上没有用
        CHECK(x1.is_empty() == false);
        CHECK(x2.is_empty() == false);
        CHECK(y.is_empty() == false);

        CHECK(x1.get_size() == x1.get_size());
        CHECK(x1.get_size() == y.get_size());

        //开始加法运算
        arma::fvec input_vec1(const_cast<float*>(input1.get_ptr<float>()), input1.size(), false, true);


    }
}