#include <chrono>
#include <iostream>

#include <glog/logging.h>
#include <gtest/gtest.h>

#include "base/DeviceController.h"
#include "op/add.h"

// ============================================================
// VecAddLayer 构造函数测试
// ============================================================

TEST(VecAddLayer, constructor_default) {
    op::VecAddLayer layer;

    EXPECT_EQ(layer.get_device_type(), base::DeviceType_t::Unknown);
    EXPECT_EQ(layer.get_layer_type(), op::LayerType_t::Add);
    EXPECT_EQ(layer.get_data_type(), tensor::DataType_t::fp32);
    EXPECT_EQ(layer.get_layer_name(), "add_layer");
}

TEST(VecAddLayer, constructor_cpu) {
    op::VecAddLayer layer(base::DeviceType_t::CPU);

    EXPECT_EQ(layer.get_device_type(), base::DeviceType_t::CPU);
    EXPECT_EQ(layer.get_layer_type(), op::LayerType_t::Add);
}

TEST(VecAddLayer, constructor_gpu) {
    op::VecAddLayer layer(base::DeviceType_t::GPU);

    EXPECT_EQ(layer.get_device_type(), base::DeviceType_t::GPU);
}

// ============================================================
// VecAddLayer check_layer 测试
// ============================================================

TEST(VecAddLayer, check_layer_empty_inputs) {
    // check_layer 会对 inputs[0], inputs[1], outputs[0] 做 check_tensor
    // 如果张量是空的，CHECK 会 abort
    // 这个测试验证正常设置后 check_layer 通过
    auto cpu = base::CPUDeviceControllerFactory::get_instance();
    op::VecAddLayer layer(base::DeviceType_t::CPU);

    layer.reset_input_tensor_num(2);
    layer.reset_output_tensor_num(1);

    tensor::Tensor in1(tensor::DataType_t::fp32, {4}, true, cpu, nullptr);
    tensor::Tensor in2(tensor::DataType_t::fp32, {4}, true, cpu, nullptr);
    tensor::Tensor out(tensor::DataType_t::fp32, {4}, true, cpu, nullptr);

    layer.set_input_tensor(0, in1);
    layer.set_input_tensor(1, in2);
    layer.set_output_tensor(0, out);

    auto status = layer.check_layer();
    EXPECT_EQ(status, base::error::kSuccess);
}

// ============================================================
// VecAddLayer forward CPU 测试
// ============================================================

TEST(VecAddLayer, forward_cpu_small) {
    auto cpu = base::CPUDeviceControllerFactory::get_instance();
    op::VecAddLayer layer(base::DeviceType_t::CPU);

    layer.reset_input_tensor_num(2);
    layer.reset_output_tensor_num(1);

    tensor::Tensor in1(tensor::DataType_t::fp32, {4}, true, cpu, nullptr);
    tensor::Tensor in2(tensor::DataType_t::fp32, {4}, true, cpu, nullptr);
    tensor::Tensor out(tensor::DataType_t::fp32, {4}, true, cpu, nullptr);

    float* d1 = static_cast<float*>(in1.get_ptr());
    float* d2 = static_cast<float*>(in2.get_ptr());
    for (size_t i = 0; i < 4; i++) {
        d1[i] = static_cast<float>(i + 1);
        d2[i] = static_cast<float>(i * 2);
    }

    layer.set_input_tensor(0, in1);
    layer.set_input_tensor(1, in2);
    layer.set_output_tensor(0, out);

    auto start = std::chrono::high_resolution_clock::now();
    auto status = layer.forward();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    EXPECT_EQ(status, base::error::kSuccess);
    std::cout << "  forward_cpu_small (4 elem): " << duration.count() << " us" << std::endl;

    float* out_data = static_cast<float*>(out.get_ptr());
    EXPECT_FLOAT_EQ(out_data[0], 1.0f);   // 1 + 0
    EXPECT_FLOAT_EQ(out_data[1], 4.0f);   // 2 + 2
    EXPECT_FLOAT_EQ(out_data[2], 7.0f);   // 3 + 4
    EXPECT_FLOAT_EQ(out_data[3], 10.0f);  // 4 + 6
}

TEST(VecAddLayer, forward_cpu_large) {
    auto cpu = base::CPUDeviceControllerFactory::get_instance();
    op::VecAddLayer layer(base::DeviceType_t::CPU);

    size_t N = 1024;
    layer.reset_input_tensor_num(2);
    layer.reset_output_tensor_num(1);

    tensor::Tensor in1(tensor::DataType_t::fp32, {N}, true, cpu, nullptr);
    tensor::Tensor in2(tensor::DataType_t::fp32, {N}, true, cpu, nullptr);
    tensor::Tensor out(tensor::DataType_t::fp32, {N}, true, cpu, nullptr);

    float* d1 = static_cast<float*>(in1.get_ptr());
    float* d2 = static_cast<float*>(in2.get_ptr());
    for (size_t i = 0; i < N; i++) {
        d1[i] = static_cast<float>(i);
        d2[i] = static_cast<float>(i * 2);
    }

    layer.set_input_tensor(0, in1);
    layer.set_input_tensor(1, in2);
    layer.set_output_tensor(0, out);

    auto start = std::chrono::high_resolution_clock::now();
    auto status = layer.forward();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    EXPECT_EQ(status, base::error::kSuccess);
    std::cout << "  forward_cpu_large (1024 elem): " << duration.count() << " us" << std::endl;

    float* out_data = static_cast<float*>(out.get_ptr());
    for (size_t i = 0; i < N; i++) {
        EXPECT_FLOAT_EQ(out_data[i], static_cast<float>(i * 3));
    }
}

TEST(VecAddLayer, forward_cpu_multi_dim) {
    auto cpu = base::CPUDeviceControllerFactory::get_instance();
    op::VecAddLayer layer(base::DeviceType_t::CPU);

    layer.reset_input_tensor_num(2);
    layer.reset_output_tensor_num(1);

    tensor::Tensor in1(tensor::DataType_t::fp32, {2, 3}, true, cpu, nullptr);
    tensor::Tensor in2(tensor::DataType_t::fp32, {2, 3}, true, cpu, nullptr);
    tensor::Tensor out(tensor::DataType_t::fp32, {2, 3}, true, cpu, nullptr);

    float* d1 = static_cast<float*>(in1.get_ptr());
    float* d2 = static_cast<float*>(in2.get_ptr());
    for (size_t i = 0; i < 6; i++) {
        d1[i] = 1.0f;
        d2[i] = 2.0f;
    }

    layer.set_input_tensor(0, in1);
    layer.set_input_tensor(1, in2);
    layer.set_output_tensor(0, out);

    auto start = std::chrono::high_resolution_clock::now();
    auto status = layer.forward();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    EXPECT_EQ(status, base::error::kSuccess);
    std::cout << "  forward_cpu_multi_dim (6 elem): " << duration.count() << " us" << std::endl;

    float* out_data = static_cast<float*>(out.get_ptr());
    for (size_t i = 0; i < 6; i++) {
        EXPECT_FLOAT_EQ(out_data[i], 3.0f);
    }
}

TEST(VecAddLayer, forward_cpu_negative_values) {
    auto cpu = base::CPUDeviceControllerFactory::get_instance();
    op::VecAddLayer layer(base::DeviceType_t::CPU);

    layer.reset_input_tensor_num(2);
    layer.reset_output_tensor_num(1);

    tensor::Tensor in1(tensor::DataType_t::fp32, {3}, true, cpu, nullptr);
    tensor::Tensor in2(tensor::DataType_t::fp32, {3}, true, cpu, nullptr);
    tensor::Tensor out(tensor::DataType_t::fp32, {3}, true, cpu, nullptr);

    float* d1 = static_cast<float*>(in1.get_ptr());
    float* d2 = static_cast<float*>(in2.get_ptr());
    d1[0] = -1.0f; d1[1] = 0.0f; d1[2] = 5.5f;
    d2[0] = 3.0f;  d2[1] = -2.0f; d2[2] = -3.5f;

    layer.set_input_tensor(0, in1);
    layer.set_input_tensor(1, in2);
    layer.set_output_tensor(0, out);

    auto start = std::chrono::high_resolution_clock::now();
    auto status = layer.forward();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    EXPECT_EQ(status, base::error::kSuccess);
    std::cout << "  forward_cpu_negative (3 elem): " << duration.count() << " us" << std::endl;

    float* out_data = static_cast<float*>(out.get_ptr());
    EXPECT_FLOAT_EQ(out_data[0], 2.0f);
    EXPECT_FLOAT_EQ(out_data[1], -2.0f);
    EXPECT_FLOAT_EQ(out_data[2], 2.0f);
}

// ============================================================
// VecAddLayer 属性测试
// ============================================================

TEST(VecAddLayer, set_layer_name_works) {
    op::VecAddLayer layer(base::DeviceType_t::CPU);
    layer.set_layer_name("my_add");
    EXPECT_EQ(layer.get_layer_name(), "my_add");
}

TEST(VecAddLayer, set_cuda_stream_works) {
    op::VecAddLayer layer(base::DeviceType_t::GPU);

    auto stream = std::make_shared<kernel::CudaStream>();
    layer.set_cuda_stream(stream);

    EXPECT_NE(layer.get_cuda_stream(), nullptr);
    EXPECT_EQ(layer.get_cuda_stream(), stream);
}

TEST(VecAddLayer, get_input_output_num_initial) {
    op::VecAddLayer layer;
    EXPECT_EQ(layer.get_input_tensor_num(), 0);
    EXPECT_EQ(layer.get_output_tensor_num(), 0);
}
