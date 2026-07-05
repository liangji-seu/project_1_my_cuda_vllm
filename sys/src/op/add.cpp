#include "op/add.h"
#include "kernel/kernel_interface.h"

namespace op{

    VecAddLayer::VecAddLayer(
        base::DeviceType_t device_type
    ):
    Layer(device_type, LayerType_t::Add, tensor::DataType_t::fp32, "add_layer")
    {}


    base::error::Status VecAddLayer::check_layer() {
        CHECK(check_tensor(get_input(0)) == base::error::Status());
        CHECK(check_tensor(get_input(1)) == base::error::Status());
        CHECK(check_tensor(get_output(0)) == base::error::Status());
        return base::error::Status();
    }

    //真实绑定返回后端内核接口
    base::error::Status VecAddLayer::forward() {
        CHECK(this->check_layer() == base::error::Status());
    
        auto input1 = this->get_input(0);
        auto input2 = this->get_input(1);

        auto output = this->get_output(0);

        //如果这个加法层是GPU上的运算，一定要指定cuda流
        void* stream_ptr = nullptr;
        if(device_type == base::DeviceType_t::GPU){
            CHECK(cuda_stream != nullptr);
            stream_ptr = cuda_stream->stream;
        }

        kernel::get_add_interface(device_type)(
            input1,
            input2,
            output,
            stream_ptr
        );
        return base::error::Status();

    }



}
