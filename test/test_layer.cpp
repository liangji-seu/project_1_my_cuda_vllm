#include <glog/logging.h>
#include <gtest/gtest.h>

#include "base/DeviceController.h"
#include "op/layer.h"

// Layer 是抽象类(set_weight_tensor 为纯虚函数)，需要一个具体子类来实例化测试
class TestLayer : public op::Layer {
public:
    using op::Layer::Layer;

    base::error::Status set_weight_tensor(size_t index, const tensor::Tensor& weight) override {
        return base::error::Status();
    }
};

// ============================================================
// Layer 构造函数测试
// ============================================================

TEST(Layer, constructor_default) {
    TestLayer layer(base::DeviceType_t::CPU, op::LayerType_t::Linear);

    EXPECT_EQ(layer.get_device_type(), base::DeviceType_t::CPU);
    EXPECT_EQ(layer.get_layer_type(), op::LayerType_t::Linear);
    EXPECT_EQ(layer.get_data_type(), tensor::DataType_t::fp32);
    EXPECT_EQ(layer.get_layer_name(), "default-Layer");
    EXPECT_EQ(layer.get_input_tensor_num(), 0);
    EXPECT_EQ(layer.get_output_tensor_num(), 0);
    EXPECT_EQ(layer.get_cuda_stream(), nullptr);
}

TEST(Layer, constructor_with_name) {
    TestLayer layer(base::DeviceType_t::GPU, op::LayerType_t::RMSNorm, tensor::DataType_t::int8, "my-layer");

    EXPECT_EQ(layer.get_device_type(), base::DeviceType_t::GPU);
    EXPECT_EQ(layer.get_layer_type(), op::LayerType_t::RMSNorm);
    EXPECT_EQ(layer.get_data_type(), tensor::DataType_t::int8);
    EXPECT_EQ(layer.get_layer_name(), "my-layer");
}

// ============================================================
// Layer init / check_layer 测试
// ============================================================

TEST(Layer, init) {
    TestLayer layer(base::DeviceType_t::CPU, op::LayerType_t::Embedding);
    auto status = layer.init();
    EXPECT_EQ(status, base::error::kSuccess);
}

TEST(Layer, check_layer) {
    TestLayer layer(base::DeviceType_t::CPU, op::LayerType_t::Softmax);
    auto status = layer.check_layer();
    EXPECT_EQ(status, base::error::kSuccess);
}

// ============================================================
// Layer 张量管理测试
// ============================================================

TEST(Layer, reset_input_tensor_num) {
    TestLayer layer(base::DeviceType_t::CPU, op::LayerType_t::Add);

    layer.reset_input_tensor_num(3);
    EXPECT_EQ(layer.get_input_tensor_num(), 3);

    layer.reset_input_tensor_num(0);
    EXPECT_EQ(layer.get_input_tensor_num(), 0);
}

TEST(Layer, reset_output_tensor_num) {
    TestLayer layer(base::DeviceType_t::CPU, op::LayerType_t::Add);

    layer.reset_output_tensor_num(2);
    EXPECT_EQ(layer.get_output_tensor_num(), 2);

    layer.reset_output_tensor_num(0);
    EXPECT_EQ(layer.get_output_tensor_num(), 0);
}

TEST(Layer, set_input_tensor) {
    auto cpu_controller = base::CPUDeviceControllerFactory::get_instance();
    TestLayer layer(base::DeviceType_t::CPU, op::LayerType_t::Matmul);

    layer.reset_input_tensor_num(2);

    tensor::Tensor t1(tensor::DataType_t::fp32, {2, 3}, true, cpu_controller, nullptr);
    tensor::Tensor t2(tensor::DataType_t::fp32, {3, 4}, true, cpu_controller, nullptr);

    layer.set_input_tensor(0, t1);
    layer.set_input_tensor(1, t2);

    EXPECT_FALSE(layer.get_input(0).is_empty());
    EXPECT_FALSE(layer.get_input(1).is_empty());
    EXPECT_EQ(layer.get_input(0).get_size(), 6);
    EXPECT_EQ(layer.get_input(1).get_size(), 12);
}

TEST(Layer, set_output_tensor) {
    auto cpu_controller = base::CPUDeviceControllerFactory::get_instance();
    TestLayer layer(base::DeviceType_t::CPU, op::LayerType_t::Matmul);

    layer.reset_output_tensor_num(1);

    tensor::Tensor out(tensor::DataType_t::fp32, {2, 4}, true, cpu_controller, nullptr);
    layer.set_output_tensor(0, out);

    EXPECT_FALSE(layer.get_output(0).is_empty());
    EXPECT_EQ(layer.get_output(0).get_size(), 8);
}

TEST(Layer, check_tensor_valid) {
    auto cpu_controller = base::CPUDeviceControllerFactory::get_instance();
    TestLayer layer(base::DeviceType_t::CPU, op::LayerType_t::Linear);

    tensor::Tensor t(tensor::DataType_t::fp32, {3, 3}, true, cpu_controller, nullptr);
    auto status = layer.check_tensor(t);
    EXPECT_EQ(status, base::error::kSuccess);
}

TEST(Layer, get_input_tensor_num_initial) {
    TestLayer layer(base::DeviceType_t::CPU, op::LayerType_t::RoPe);
    EXPECT_EQ(layer.get_input_tensor_num(), 0);
}

TEST(Layer, get_output_tensor_num_initial) {
    TestLayer layer(base::DeviceType_t::CPU, op::LayerType_t::RoPe);
    EXPECT_EQ(layer.get_output_tensor_num(), 0);
}

// ============================================================
// Layer CUDA 流测试
// ============================================================

TEST(Layer, set_cuda_stream) {
    TestLayer layer(base::DeviceType_t::GPU, op::LayerType_t::MHA);

    auto stream = std::make_shared<kernel::CudaStream>();
    layer.set_cuda_stream(stream);

    EXPECT_NE(layer.get_cuda_stream(), nullptr);
    EXPECT_EQ(layer.get_cuda_stream(), stream);
}

TEST(Layer, get_cuda_stream_default_null) {
    TestLayer layer(base::DeviceType_t::CPU, op::LayerType_t::MHA);
    EXPECT_EQ(layer.get_cuda_stream(), nullptr);
}

// ============================================================
// Layer set_layer_name / set_device_type 测试
// ============================================================

TEST(Layer, set_layer_name) {
    TestLayer layer(base::DeviceType_t::CPU, op::LayerType_t::SwiGLU);
    layer.set_layer_name("new-name");
    EXPECT_EQ(layer.get_layer_name(), "new-name");
}

TEST(Layer, set_device_type) {
    TestLayer layer(base::DeviceType_t::CPU, op::LayerType_t::SwiGLU);
    layer.set_device_type(base::DeviceType_t::GPU);
    EXPECT_EQ(layer.get_device_type(), base::DeviceType_t::GPU);
}

// ============================================================
// Layer forward 测试 (基类返回 kFunctionUnImplement)
// ============================================================

TEST(Layer, forward) {
    TestLayer layer(base::DeviceType_t::CPU, op::LayerType_t::Encode);
    auto status = layer.forward();
    EXPECT_EQ(status, base::error::kFunctionUnImplement);
}

TEST(Layer, forward_1_input) {
    auto cpu_controller = base::CPUDeviceControllerFactory::get_instance();
    TestLayer layer(base::DeviceType_t::CPU, op::LayerType_t::Encode);

    tensor::Tensor input(tensor::DataType_t::fp32, {2, 3}, true, cpu_controller, nullptr);
    tensor::Tensor output(tensor::DataType_t::fp32, {2, 3}, true, cpu_controller, nullptr);

    auto status = layer.forward(input, output);

    EXPECT_EQ(status, base::error::kFunctionUnImplement);
    EXPECT_EQ(layer.get_input_tensor_num(), 1);
    EXPECT_EQ(layer.get_output_tensor_num(), 1);
    EXPECT_FALSE(layer.get_input(0).is_empty());
    EXPECT_FALSE(layer.get_output(0).is_empty());
}

TEST(Layer, forward_2_inputs) {
    auto cpu_controller = base::CPUDeviceControllerFactory::get_instance();
    TestLayer layer(base::DeviceType_t::CPU, op::LayerType_t::Add);

    tensor::Tensor input1(tensor::DataType_t::fp32, {2, 3}, true, cpu_controller, nullptr);
    tensor::Tensor input2(tensor::DataType_t::fp32, {2, 3}, true, cpu_controller, nullptr);
    tensor::Tensor output(tensor::DataType_t::fp32, {2, 3}, true, cpu_controller, nullptr);

    auto status = layer.forward(input1, input2, output);

    EXPECT_EQ(status, base::error::kFunctionUnImplement);
    EXPECT_EQ(layer.get_input_tensor_num(), 2);
    EXPECT_EQ(layer.get_output_tensor_num(), 1);
}

TEST(Layer, forward_3_inputs) {
    auto cpu_controller = base::CPUDeviceControllerFactory::get_instance();
    TestLayer layer(base::DeviceType_t::CPU, op::LayerType_t::Add);

    tensor::Tensor input1(tensor::DataType_t::fp32, {2, 3}, true, cpu_controller, nullptr);
    tensor::Tensor input2(tensor::DataType_t::fp32, {2, 3}, true, cpu_controller, nullptr);
    tensor::Tensor input3(tensor::DataType_t::fp32, {2, 3}, true, cpu_controller, nullptr);
    tensor::Tensor output(tensor::DataType_t::fp32, {2, 3}, true, cpu_controller, nullptr);

    auto status = layer.forward(input1, input2, input3, output);

    EXPECT_EQ(status, base::error::kFunctionUnImplement);
    EXPECT_EQ(layer.get_input_tensor_num(), 3);
    EXPECT_EQ(layer.get_output_tensor_num(), 1);
}

TEST(Layer, forward_4_inputs) {
    auto cpu_controller = base::CPUDeviceControllerFactory::get_instance();
    TestLayer layer(base::DeviceType_t::CPU, op::LayerType_t::Add);

    tensor::Tensor input1(tensor::DataType_t::fp32, {2, 3}, true, cpu_controller, nullptr);
    tensor::Tensor input2(tensor::DataType_t::fp32, {2, 3}, true, cpu_controller, nullptr);
    tensor::Tensor input3(tensor::DataType_t::fp32, {2, 3}, true, cpu_controller, nullptr);
    tensor::Tensor input4(tensor::DataType_t::fp32, {2, 3}, true, cpu_controller, nullptr);
    tensor::Tensor output(tensor::DataType_t::fp32, {2, 3}, true, cpu_controller, nullptr);

    auto status = layer.forward(input1, input2, input3, input4, output);

    EXPECT_EQ(status, base::error::kFunctionUnImplement);
    EXPECT_EQ(layer.get_input_tensor_num(), 4);
    EXPECT_EQ(layer.get_output_tensor_num(), 1);
}

TEST(Layer, forward_5_inputs) {
    auto cpu_controller = base::CPUDeviceControllerFactory::get_instance();
    TestLayer layer(base::DeviceType_t::CPU, op::LayerType_t::Add);

    tensor::Tensor input1(tensor::DataType_t::fp32, {2, 3}, true, cpu_controller, nullptr);
    tensor::Tensor input2(tensor::DataType_t::fp32, {2, 3}, true, cpu_controller, nullptr);
    tensor::Tensor input3(tensor::DataType_t::fp32, {2, 3}, true, cpu_controller, nullptr);
    tensor::Tensor input4(tensor::DataType_t::fp32, {2, 3}, true, cpu_controller, nullptr);
    tensor::Tensor input5(tensor::DataType_t::fp32, {2, 3}, true, cpu_controller, nullptr);
    tensor::Tensor output(tensor::DataType_t::fp32, {2, 3}, true, cpu_controller, nullptr);

    auto status = layer.forward(input1, input2, input3, input4, input5, output);

    EXPECT_EQ(status, base::error::kFunctionUnImplement);
    EXPECT_EQ(layer.get_input_tensor_num(), 5);
    EXPECT_EQ(layer.get_output_tensor_num(), 1);
}

// ============================================================
// LayerParam 构造函数测试
// ============================================================

TEST(LayerParam, constructor_default) {
    op::LayerParam layer(base::DeviceType_t::CPU, op::LayerType_t::Linear);

    EXPECT_EQ(layer.get_device_type(), base::DeviceType_t::CPU);
    EXPECT_EQ(layer.get_layer_type(), op::LayerType_t::Linear);
    EXPECT_EQ(layer.get_data_type(), tensor::DataType_t::fp32);
    EXPECT_EQ(layer.get_layer_name(), "default-LayerParam");
    EXPECT_EQ(layer.get_weight_tensor_num(), 0);
    EXPECT_EQ(layer.get_scale_num(), 0);
}

TEST(LayerParam, constructor_quant) {
    op::LayerParam layer(base::DeviceType_t::GPU, op::LayerType_t::Linear, true, "quant-layer");

    EXPECT_EQ(layer.get_layer_name(), "quant-layer");
    EXPECT_EQ(layer.get_device_type(), base::DeviceType_t::GPU);
}

TEST(LayerParam, constructor_not_quant) {
    op::LayerParam layer(base::DeviceType_t::CPU, op::LayerType_t::Embedding, false);

    EXPECT_EQ(layer.get_device_type(), base::DeviceType_t::CPU);
    EXPECT_EQ(layer.get_weight_tensor_num(), 0);
}

// ============================================================
// LayerParam 权重管理测试
// ============================================================

TEST(LayerParam, reset_weight_tensor_num) {
    op::LayerParam layer(base::DeviceType_t::CPU, op::LayerType_t::Linear);

    layer.reset_weight_tensor_num(2);
    EXPECT_EQ(layer.get_weight_tensor_num(), 2);

    layer.reset_weight_tensor_num(0);
    EXPECT_EQ(layer.get_weight_tensor_num(), 0);
}

TEST(LayerParam, set_weight_tensor) {
    auto cpu_controller = base::CPUDeviceControllerFactory::get_instance();
    op::LayerParam layer(base::DeviceType_t::CPU, op::LayerType_t::Linear);

    layer.reset_weight_tensor_num(1);

    tensor::Tensor weight(tensor::DataType_t::fp32, {4, 4}, true, cpu_controller, nullptr);
    auto status = layer.set_weight_tensor(0, weight);

    EXPECT_EQ(status, base::error::kSuccess);
    EXPECT_FALSE(layer.get_weight(0).is_empty());
    EXPECT_EQ(layer.get_weight(0).get_size(), 16);
}

TEST(LayerParam, get_weight_tensor_num_initial) {
    op::LayerParam layer(base::DeviceType_t::CPU, op::LayerType_t::Linear);
    EXPECT_EQ(layer.get_weight_tensor_num(), 0);
}

// ============================================================
// LayerParam scales / group_size 测试
// ============================================================

TEST(LayerParam, set_scales) {
    auto cpu_controller = base::CPUDeviceControllerFactory::get_instance();
    op::LayerParam layer(base::DeviceType_t::CPU, op::LayerType_t::Linear);

    tensor::Tensor scales(tensor::DataType_t::fp32, {4}, true, cpu_controller, nullptr);
    layer.set_scales(scales);

    EXPECT_EQ(layer.get_scale_num(), 1);
}

TEST(LayerParam, get_scale_num_empty) {
    op::LayerParam layer(base::DeviceType_t::CPU, op::LayerType_t::Linear);
    EXPECT_EQ(layer.get_scale_num(), 0);
}

TEST(LayerParam, set_group_size) {
    op::LayerParam layer(base::DeviceType_t::CPU, op::LayerType_t::Linear);

    layer.set_group_size(128);
    // group_size 没有公开 getter，验证不崩溃即可
    SUCCEED();
}

// ============================================================
// LayerParam to 设备转移测试
// ============================================================

TEST(LayerParam, to_cpu_already_cpu) {
    auto cpu_controller = base::CPUDeviceControllerFactory::get_instance();
    op::LayerParam layer(base::DeviceType_t::CPU, op::LayerType_t::Linear);

    layer.reset_input_tensor_num(1);
    layer.reset_output_tensor_num(1);
    layer.reset_weight_tensor_num(1);

    tensor::Tensor input(tensor::DataType_t::fp32, {2, 3}, true, cpu_controller, nullptr);
    tensor::Tensor output(tensor::DataType_t::fp32, {2, 3}, true, cpu_controller, nullptr);
    tensor::Tensor weight(tensor::DataType_t::fp32, {3, 3}, true, cpu_controller, nullptr);

    layer.set_input_tensor(0, input);
    layer.set_output_tensor(0, output);
    layer.set_weight_tensor(0, weight);

    void* input_ptr_before = layer.get_input(0).get_ptr();
    void* output_ptr_before = layer.get_output(0).get_ptr();
    void* weight_ptr_before = layer.get_weight(0).get_ptr();

    layer.to("cpu");

    EXPECT_EQ(layer.get_input(0).get_ptr(), input_ptr_before);
    EXPECT_EQ(layer.get_output(0).get_ptr(), output_ptr_before);
    EXPECT_EQ(layer.get_weight(0).get_ptr(), weight_ptr_before);
    EXPECT_EQ(layer.get_input(0).get_device_type(), base::DeviceType_t::CPU);
    EXPECT_EQ(layer.get_output(0).get_device_type(), base::DeviceType_t::CPU);
    EXPECT_EQ(layer.get_weight(0).get_device_type(), base::DeviceType_t::CPU);
}

TEST(LayerParam, to_cpu_with_empty_tensors) {
    op::LayerParam layer(base::DeviceType_t::CPU, op::LayerType_t::Linear);

    layer.reset_input_tensor_num(2);
    layer.reset_output_tensor_num(1);
    layer.reset_weight_tensor_num(1);

    layer.to("cpu");

    EXPECT_EQ(layer.get_input_tensor_num(), 2);
}

// ============================================================
// LayerParam 继承自 Layer 的方法测试
// ============================================================

TEST(LayerParam, inherit_layer_methods) {
    auto cpu_controller = base::CPUDeviceControllerFactory::get_instance();
    op::LayerParam layer(base::DeviceType_t::CPU, op::LayerType_t::Linear);

    layer.reset_input_tensor_num(1);
    layer.reset_output_tensor_num(1);

    tensor::Tensor input(tensor::DataType_t::fp32, {2, 3}, true, cpu_controller, nullptr);
    layer.set_input_tensor(0, input);

    EXPECT_EQ(layer.get_input_tensor_num(), 1);
    EXPECT_FALSE(layer.get_input(0).is_empty());
}

TEST(LayerParam, forward_inherited) {
    auto cpu_controller = base::CPUDeviceControllerFactory::get_instance();
    op::LayerParam layer(base::DeviceType_t::CPU, op::LayerType_t::Linear);

    tensor::Tensor input(tensor::DataType_t::fp32, {2, 3}, true, cpu_controller, nullptr);
    tensor::Tensor output(tensor::DataType_t::fp32, {2, 3}, true, cpu_controller, nullptr);

    auto status = layer.forward(input, output);
    EXPECT_EQ(status, base::error::kFunctionUnImplement);
}
