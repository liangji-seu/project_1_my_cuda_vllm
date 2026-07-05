#include "add.h"
#include "kernel_interface.h"

namespace op{

    VecAddLayer::VecAddLayer(
        base::DeviceType device_type
    ):
    Layer(device_type, LayerType_t::Add, tensor::DataType_t::fp32, "add_layer")
    {}


    base::error::Status VecAddLayer::check_layer() const {
        CHECK(check_tensor(get_input(0)) == base::error::Success());
        CHECK(check_tensor(get_input(1)) == base::error::Success());
        CHECK(check_tensor(get_output(0), false) == base::error::Success());
        return base::error::Success();
    }

    //真实绑定返回后端内核接口
    base::error::Status VecAddLayer::forward() {
        CHECK(this->check_layer() == base::error::Success());
    
        auto input1 = this->get_input(0);
        auto input2 = this->get_input(1);

        auto output = this->get_output(0);

        //如果这个加法层是GPU上的运算，一定要指定cuda流
        if(device_type == base::DeviceType_t::GPU){
            CHECK(cuda_stream != nullptr);
        }

        kernel::get_add_interface(device_type)(
            input1,
            input2,
            output,
            cuda_stream
        );
        return base::error::Success();

    }



}
