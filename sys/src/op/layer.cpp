#include "op/layer.h"
#include "glog/logging.h"
#include "base/cuda_stream.h"


namespace op{

    /**
     * BaseLayer
     */

    BaseLayer::BaseLayer(
                base::DeviceType_t device_type,
                LayerType_t layer_type,
                tensor::DataType_t data_type,
                std::string layer_name
            ):
            name(layer_name),
            layer_type(layer_type),
            data_type(data_type),
            device_type(device_type)
            {}


    /**
     * 查询方法实现
     */
    tensor::DataType_t BaseLayer::get_data_type() const {
        return this->data_type;
    }

    LayerType_t BaseLayer::get_layer_type() const {
        return this->layer_type;
    }

    const std::string& BaseLayer::get_layer_name() const {
        return this->name;
    }

    base::DeviceType_t BaseLayer::get_device_type() const {
        return this->device_type;
    }

    void BaseLayer::set_layer_name(const std::string& layer_name) {
        this->name = layer_name;
    }

    void BaseLayer::set_device_type(base::DeviceType_t device_type) {
        this->device_type = device_type;
    }





    /**
     * Layer
     */

    Layer::Layer(
            base::DeviceType_t device_type,
            LayerType_t layer_type,
            tensor::DataType_t data_type,
            std::string layer_name
        ):
        cuda_stream(nullptr),
        BaseLayer(
            device_type,
            layer_type,
            data_type,
            layer_name
        )
        {}

    //初始化无权重层，暂定
    base::error::Status Layer::init() {
        //TODO
        return base::error::Status();
    }

    base::error::Status Layer::check_layer() {
        //自检查
        //TODO 
        return base::error::Status();
    }

    void Layer::set_input_tensor(size_t index, const tensor::Tensor& input){
        CHECK(this->inputs.size() >= 0);
        CHECK(index >=0 && index < inputs.size());
        this->inputs[index] = input;
        return;
    }

    void Layer::set_output_tensor(int32_t index, const tensor::Tensor& output){
        CHECK(this->outputs.size() >= 0);
        CHECK(index >=0 && index < outputs.size());
        this->outputs[index] = output;
        return;
    }

    base::error::Status Layer::set_weight_tensor(size_t index, const tensor::Tensor& weight){
        //无权重的层，空实现
        return base::error::Status();
    }

    base::error::Status Layer::check_tensor(const tensor::Tensor& tensor, bool is_input) const {
        if(is_input)
            CHECK(!tensor.is_empty());
        CHECK(this->get_device_type() == tensor.get_device_type());
        (void)is_input;
        return base::error::Status();
    }

    tensor::Tensor& Layer::get_input(size_t index) {
        CHECK(index < inputs.size());
        return inputs[index];
    }

    tensor::Tensor& Layer::get_output(size_t index) {
        CHECK(index < outputs.size());
        return outputs[index];
    }

    size_t Layer::get_input_tensor_num() const {
        return inputs.size();
    }

    size_t Layer::get_output_tensor_num() const {
        return outputs.size();
    }

    void Layer::set_cuda_stream(std::shared_ptr<kernel::CudaStream> stream) {
        this->cuda_stream = stream;
    }

    std::shared_ptr<kernel::CudaStream> Layer::get_cuda_stream() const {
        return cuda_stream;
    }

    void Layer::reset_input_tensor_num(size_t size){
        this->inputs.resize(size);
    }
    void Layer::reset_output_tensor_num(size_t size){
        this->outputs.resize(size);
    }


    //前向传播
    base::error::Status Layer::forward(){
        //调用后端的接口（空），每个具体的算子子类，来实现
        return base::error::Status(base::error::kFunctionUnImplement, "");
    }

    base::error::Status Layer::forward(const tensor::Tensor& input1,
                        const tensor::Tensor& output1){
        //
        CHECK(this->check_tensor(input1) == base::error::Status());
        this->reset_input_tensor_num(1);
        this->set_input_tensor(0, input1);

        //CHECK(this->check_tensor(output1) == base::error::Status());
        this->reset_output_tensor_num(1);
        this->set_output_tensor(0, output1);
        return this->forward();
    }

    base::error::Status Layer::forward(const tensor::Tensor& input1,
                        const tensor::Tensor& input2,
                        const tensor::Tensor& output1){
        CHECK(this->check_tensor(input1) == base::error::Status());
        CHECK(this->check_tensor(input2) == base::error::Status());
        this->reset_input_tensor_num(2);
        this->set_input_tensor(0, input1);
        this->set_input_tensor(1, input2);

        this->reset_output_tensor_num(1);
        this->set_output_tensor(0, output1);
        return this->forward();
    }

    base::error::Status Layer::forward(const tensor::Tensor& input1,
                        const tensor::Tensor& input2,
                        const tensor::Tensor& input3,
                        const tensor::Tensor& output1){
        CHECK(this->check_tensor(input1) == base::error::Status());
        CHECK(this->check_tensor(input2) == base::error::Status());
        CHECK(this->check_tensor(input3) == base::error::Status());
        this->reset_input_tensor_num(3);
        this->set_input_tensor(0, input1);
        this->set_input_tensor(1, input2);
        this->set_input_tensor(2, input3);

        this->reset_output_tensor_num(1);
        this->set_output_tensor(0, output1);
        return this->forward();
    }

    base::error::Status Layer::forward(const tensor::Tensor& input1,
                        const tensor::Tensor& input2,
                        const tensor::Tensor& input3,
                        const tensor::Tensor& input4,
                        const tensor::Tensor& output1){
        CHECK(this->check_tensor(input1) == base::error::Status());
        CHECK(this->check_tensor(input2) == base::error::Status());
        CHECK(this->check_tensor(input3) == base::error::Status());
        CHECK(this->check_tensor(input4) == base::error::Status());
        this->reset_input_tensor_num(4);
        this->set_input_tensor(0, input1);
        this->set_input_tensor(1, input2);
        this->set_input_tensor(2, input3);
        this->set_input_tensor(3, input4);

        this->reset_output_tensor_num(1);
        this->set_output_tensor(0, output1);
        return this->forward();
    }

    base::error::Status Layer::forward(const tensor::Tensor& input1,
                        const tensor::Tensor& input2,
                        const tensor::Tensor& input3,
                        const tensor::Tensor& input4,
                        const tensor::Tensor& input5,
                        const tensor::Tensor& output1){
        CHECK(this->check_tensor(input1) == base::error::Status());
        CHECK(this->check_tensor(input2) == base::error::Status());
        CHECK(this->check_tensor(input3) == base::error::Status());
        CHECK(this->check_tensor(input4) == base::error::Status());
        CHECK(this->check_tensor(input5) == base::error::Status());
        this->reset_input_tensor_num(5);
        this->set_input_tensor(0, input1);
        this->set_input_tensor(1, input2);
        this->set_input_tensor(2, input3);
        this->set_input_tensor(3, input4);
        this->set_input_tensor(4, input5);

        this->reset_output_tensor_num(1);
        this->set_output_tensor(0, output1);
        return this->forward();
    }






    LayerParam::LayerParam(
        base::DeviceType_t device_type,
        LayerType_t layer_type,
        bool is_quant_layer,
        std::string layer_name
    ):
    is_quant_layer(is_quant_layer),//写死fp32的数据类型
    Layer(device_type, layer_type, tensor::DataType_t::fp32,layer_name)
    {
        //层的数据类型未定，等权重张量给定后再确认
    }

    size_t LayerParam::get_weight_tensor_num() const {
        return weights.size();
    }

    size_t LayerParam::get_scale_num() const {
        return scales.is_empty() ? 0 : 1;
    }

    void LayerParam::reset_weight_tensor_num(size_t size) {
        weights.resize(size);
    }

    tensor::Tensor& LayerParam::get_weight(size_t index) {
        CHECK(index < weights.size());
        return weights[index];
    }

    base::error::Status LayerParam::set_weight_tensor(size_t index, const tensor::Tensor& weight) {
        CHECK(index < weights.size());
        weights[index] = weight;
        return base::error::Status();
    }

    base::error::Status LayerParam::set_weight(size_t idx, const std::vector<size_t>& dims,
                                                const void* ptr, base::DeviceType_t device_type) {
        CHECK(idx < weights.size());
        CHECK(ptr != nullptr);

        tensor::Tensor weight(data_type, dims, false, nullptr,
                              const_cast<void*>(ptr));
        weight.set_device_type(device_type);
        weights[idx] = weight;
        return base::error::Status();
    }

    void LayerParam::set_scales(const tensor::Tensor& scales) {
        this->scales = scales;
    }

    void LayerParam::set_group_size(size_t group_size) {
        this->group_size = group_size;
    }

    void LayerParam::to(std::string dst_device_type){
        cudaStream_t stream = cuda_stream ? cuda_stream->stream : nullptr;
        for(auto& input : inputs){
            if(!input.is_empty()){
                input.to(dst_device_type, stream);
            }
        }

        for(auto& output : outputs){
            if(!output.is_empty()){
                output.to(dst_device_type, stream);
            }
        }

        for(auto& weight : weights){
            if(!weight.is_empty()){
                weight.to(dst_device_type, stream);
            }
        }
    }

}