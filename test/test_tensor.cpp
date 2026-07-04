#include <cstring>
#include <numeric>

#include <glog/logging.h>
#include <gtest/gtest.h>

#include "base/DeviceController.h"
#include "tensor/tensor.h"

// ============================================================
// 构造函数测试
// ============================================================

TEST(Tensor, default_constructor) {
    tensor::Tensor t;

    EXPECT_TRUE(t.is_empty());
    EXPECT_EQ(t.get_dims_size(), 0);
    EXPECT_EQ(t.get_data_type(), tensor::DataType_t::Unknown);
    EXPECT_EQ(t.get_buffer(), nullptr);
}

TEST(Tensor, construct_with_dims_no_alloc) {
    auto cpu_controller = base::CPUDeviceControllerFactory::get_instance();
    tensor::Tensor t(tensor::DataType_t::fp32, {2, 3, 4}, false, cpu_controller, nullptr);

    EXPECT_EQ(t.get_data_type(), tensor::DataType_t::fp32);
    EXPECT_EQ(t.get_dims_size(), 3);
    EXPECT_EQ(t.get_dim(0), 2);
    EXPECT_EQ(t.get_dim(1), 3);
    EXPECT_EQ(t.get_dim(2), 4);
}

TEST(Tensor, construct_with_dims_and_alloc) {
    auto cpu_controller = base::CPUDeviceControllerFactory::get_instance();
    tensor::Tensor t(tensor::DataType_t::fp32, {3, 4}, true, cpu_controller, nullptr);

    EXPECT_FALSE(t.is_empty());
    EXPECT_EQ(t.get_data_type(), tensor::DataType_t::fp32);
    EXPECT_EQ(t.get_dims_size(), 2);
    EXPECT_EQ(t.get_size(), 12);
    EXPECT_NE(t.get_ptr(), nullptr);
    EXPECT_EQ(t.get_device_type(), base::DeviceType_t::CPU);
}

// ============================================================
// 查询方法测试
// ============================================================

TEST(Tensor, is_empty_default) {
    tensor::Tensor t;
    EXPECT_TRUE(t.is_empty());
}

TEST(Tensor, is_empty_after_alloc) {
    auto cpu_controller = base::CPUDeviceControllerFactory::get_instance();
    tensor::Tensor t(tensor::DataType_t::fp32, {2, 2}, true, cpu_controller, nullptr);
    EXPECT_FALSE(t.is_empty());
}

TEST(Tensor, get_size) {
    auto cpu_controller = base::CPUDeviceControllerFactory::get_instance();
    tensor::Tensor t(tensor::DataType_t::int32, {3, 5}, true, cpu_controller, nullptr);

    EXPECT_EQ(t.get_size(), 15);
}

TEST(Tensor, get_byte_size_fp32) {
    auto cpu_controller = base::CPUDeviceControllerFactory::get_instance();
    tensor::Tensor t(tensor::DataType_t::fp32, {2, 3}, true, cpu_controller, nullptr);

    // 2 * 3 * sizeof(float) = 6 * 4 = 24
    EXPECT_EQ(t.get_byte_size(), 24);
}

TEST(Tensor, get_byte_size_int8) {
    auto cpu_controller = base::CPUDeviceControllerFactory::get_instance();
    tensor::Tensor t(tensor::DataType_t::int8, {10}, true, cpu_controller, nullptr);

    // 10 * sizeof(int8_t) = 10
    EXPECT_EQ(t.get_byte_size(), 10);
}

TEST(Tensor, get_byte_size_int32) {
    auto cpu_controller = base::CPUDeviceControllerFactory::get_instance();
    tensor::Tensor t(tensor::DataType_t::int32, {5}, true, cpu_controller, nullptr);

    // 5 * sizeof(int32_t) = 20
    EXPECT_EQ(t.get_byte_size(), 20);
}

TEST(Tensor, get_dims_size) {
    auto cpu_controller = base::CPUDeviceControllerFactory::get_instance();
    tensor::Tensor t(tensor::DataType_t::fp32, {2, 3, 4, 5}, true, cpu_controller, nullptr);

    EXPECT_EQ(t.get_dims_size(), 4);
}

TEST(Tensor, get_data_type) {
    auto cpu_controller = base::CPUDeviceControllerFactory::get_instance();
    tensor::Tensor t(tensor::DataType_t::int8, {3}, true, cpu_controller, nullptr);

    EXPECT_EQ(t.get_data_type(), tensor::DataType_t::int8);
}

TEST(Tensor, get_dim) {
    auto cpu_controller = base::CPUDeviceControllerFactory::get_instance();
    tensor::Tensor t(tensor::DataType_t::fp32, {2, 3, 4}, true, cpu_controller, nullptr);

    EXPECT_EQ(t.get_dim(0), 2);
    EXPECT_EQ(t.get_dim(1), 3);
    EXPECT_EQ(t.get_dim(2), 4);
}

TEST(Tensor, get_dims) {
    auto cpu_controller = base::CPUDeviceControllerFactory::get_instance();
    tensor::Tensor t(tensor::DataType_t::fp32, {4, 5}, true, cpu_controller, nullptr);

    const auto& dims = t.get_dims();
    ASSERT_EQ(dims.size(), 2);
    EXPECT_EQ(dims[0], 4);
    EXPECT_EQ(dims[1], 5);
}

TEST(Tensor, get_strides) {
    auto cpu_controller = base::CPUDeviceControllerFactory::get_instance();
    tensor::Tensor t(tensor::DataType_t::fp32, {2, 3, 4}, true, cpu_controller, nullptr);

    // strides: [3*4, 4, 1] = [12, 4, 1]
    const auto& strides = t.get_strides();
    ASSERT_EQ(strides.size(), 3);
    EXPECT_EQ(strides[0], 12);
    EXPECT_EQ(strides[1], 4);
    EXPECT_EQ(strides[2], 1);
}

TEST(Tensor, get_strides_scalar) {
    auto cpu_controller = base::CPUDeviceControllerFactory::get_instance();
    tensor::Tensor t(tensor::DataType_t::fp32, {1}, true, cpu_controller, nullptr);

    // 标量 tensor: strides = [1]
    const auto& strides = t.get_strides();
    ASSERT_EQ(strides.size(), 1);
    EXPECT_EQ(strides[0], 1);
}

TEST(Tensor, get_device_type_cpu) {
    auto cpu_controller = base::CPUDeviceControllerFactory::get_instance();
    tensor::Tensor t(tensor::DataType_t::fp32, {2}, true, cpu_controller, nullptr);

    EXPECT_EQ(t.get_device_type(), base::DeviceType_t::CPU);
}

TEST(Tensor, get_ptr) {
    auto cpu_controller = base::CPUDeviceControllerFactory::get_instance();
    tensor::Tensor t(tensor::DataType_t::fp32, {4}, true, cpu_controller, nullptr);

    void* ptr = t.get_ptr();
    EXPECT_NE(ptr, nullptr);
}

TEST(Tensor, get_ptr_offset) {
    auto cpu_controller = base::CPUDeviceControllerFactory::get_instance();
    tensor::Tensor t(tensor::DataType_t::int32, {5}, true, cpu_controller, nullptr);

    // 写入数据
    int32_t* base = static_cast<int32_t*>(t.get_ptr());
    for (int32_t i = 0; i < 5; ++i) {
        base[i] = i * 10;
    }

    // 验证通过 offset 指针读取
    int32_t* p2 = static_cast<int32_t*>(t.get_ptr_offset(2));
    EXPECT_EQ(*p2, 20);

    int32_t* p4 = static_cast<int32_t*>(t.get_ptr_offset(4));
    EXPECT_EQ(*p4, 40);
}

TEST(Tensor, get_buffer) {
    auto cpu_controller = base::CPUDeviceControllerFactory::get_instance();
    tensor::Tensor t(tensor::DataType_t::fp32, {2}, true, cpu_controller, nullptr);

    auto buf = t.get_buffer();
    EXPECT_NE(buf, nullptr);
    EXPECT_EQ(buf->get_byte_size(), 8); // 2 * sizeof(float)
}

// ============================================================
// 操作方法测试
// ============================================================

TEST(Tensor, reshape_same_size) {
    auto cpu_controller = base::CPUDeviceControllerFactory::get_instance();
    tensor::Tensor t(tensor::DataType_t::fp32, {2, 6}, true, cpu_controller, nullptr);

    // 写入数据
    float* data = static_cast<float*>(t.get_ptr());
    for (size_t i = 0; i < 12; ++i) {
        data[i] = static_cast<float>(i);
    }

    // reshape 到相同元素个数，不需要重新分配
    t.reshape({3, 4});

    EXPECT_EQ(t.get_dims_size(), 2);
    EXPECT_EQ(t.get_dim(0), 3);
    EXPECT_EQ(t.get_dim(1), 4);
    EXPECT_EQ(t.get_size(), 12);
    // 原有数据应该还在
    EXPECT_FLOAT_EQ(static_cast<float*>(t.get_ptr())[0], 0.0f);
    EXPECT_FLOAT_EQ(static_cast<float*>(t.get_ptr())[11], 11.0f);
}

TEST(Tensor, reshape_larger) {
    auto cpu_controller = base::CPUDeviceControllerFactory::get_instance();
    tensor::Tensor t(tensor::DataType_t::fp32, {2, 3}, true, cpu_controller, nullptr);

    // 写入数据
    float* data = static_cast<float*>(t.get_ptr());
    for (size_t i = 0; i < 6; ++i) {
        data[i] = static_cast<float>(i);
    }

    // reshape 到更大元素个数，需要重新分配
    t.reshape({4, 4});

    EXPECT_EQ(t.get_dims_size(), 2);
    EXPECT_EQ(t.get_dim(0), 4);
    EXPECT_EQ(t.get_dim(1), 4);
    EXPECT_EQ(t.get_size(), 16);
    EXPECT_NE(t.get_ptr(), nullptr);
    // 原有数据应该被拷贝过来
    EXPECT_FLOAT_EQ(static_cast<float*>(t.get_ptr())[0], 0.0f);
    EXPECT_FLOAT_EQ(static_cast<float*>(t.get_ptr())[5], 5.0f);
}

TEST(Tensor, reshape_smaller) {
    auto cpu_controller = base::CPUDeviceControllerFactory::get_instance();
    tensor::Tensor t(tensor::DataType_t::fp32, {2, 3}, true, cpu_controller, nullptr);

    // 写入数据
    float* data = static_cast<float*>(t.get_ptr());
    for (size_t i = 0; i < 6; ++i) {
        data[i] = static_cast<float>(i);
    }

    // reshape 到更小元素个数，不重新分配
    t.reshape({3, 1});

    EXPECT_EQ(t.get_dims_size(), 2);
    EXPECT_EQ(t.get_dim(0), 3);
    EXPECT_EQ(t.get_dim(1), 1);
    EXPECT_EQ(t.get_size(), 3);
    // 原有数据应该还在
    EXPECT_FLOAT_EQ(static_cast<float*>(t.get_ptr())[0], 0.0f);
    EXPECT_FLOAT_EQ(static_cast<float*>(t.get_ptr())[2], 2.0f);
}

TEST(Tensor, reshape_empty_buffer) {
    auto cpu_controller = base::CPUDeviceControllerFactory::get_instance();
    // 不分配内存
    tensor::Tensor t(tensor::DataType_t::int32, {2, 3}, false, cpu_controller, nullptr);

    t.reshape({3, 3});

    EXPECT_EQ(t.get_dim(0), 3);
    EXPECT_EQ(t.get_dim(1), 3);
    EXPECT_EQ(t.get_size(), 9);
}

TEST(Tensor, reset) {
    auto cpu_controller = base::CPUDeviceControllerFactory::get_instance();
    tensor::Tensor t(tensor::DataType_t::fp32, {2, 3}, true, cpu_controller, nullptr);

    // reset 成新形状和类型
    t.reset({4, 5}, tensor::DataType_t::int8);

    EXPECT_EQ(t.get_data_type(), tensor::DataType_t::int8);
    EXPECT_EQ(t.get_dim(0), 4);
    EXPECT_EQ(t.get_dim(1), 5);
    EXPECT_EQ(t.get_size(), 20);
    EXPECT_EQ(t.get_buffer(), nullptr);
}

TEST(Tensor, reset_default_type) {
    auto cpu_controller = base::CPUDeviceControllerFactory::get_instance();
    tensor::Tensor t(tensor::DataType_t::fp32, {2}, true, cpu_controller, nullptr);

    // 使用默认 data_type 参数
    t.reset({3});

    EXPECT_EQ(t.get_data_type(), tensor::DataType_t::int32);
    EXPECT_EQ(t.get_dim(0), 3);
    EXPECT_EQ(t.get_size(), 3);
}

TEST(Tensor, assign) {
    auto cpu_controller = base::CPUDeviceControllerFactory::get_instance();
    tensor::Tensor t(tensor::DataType_t::fp32, {2, 3}, true, cpu_controller, nullptr);

    // 分配一个更大的外部 buffer
    auto external_buffer = std::make_shared<base::Buffer>(
        128, nullptr, base::DeviceType_t::Unknown, cpu_controller, false);

    bool result = t.assign(external_buffer);
    EXPECT_TRUE(result);
    EXPECT_EQ(t.get_buffer(), external_buffer);
}

TEST(Tensor, assign_buffer_too_small) {
    auto cpu_controller = base::CPUDeviceControllerFactory::get_instance();
    tensor::Tensor t(tensor::DataType_t::fp32, {4, 4}, true, cpu_controller, nullptr);
    // 4 * 4 * sizeof(float) = 64 bytes

    // 分配一个更小的外部 buffer
    auto small_buffer = std::make_shared<base::Buffer>(
        8, nullptr, base::DeviceType_t::Unknown, cpu_controller, false);

    bool result = t.assign(small_buffer);
    EXPECT_FALSE(result);
}

TEST(Tensor, assign_to_empty_tensor) {
    auto cpu_controller = base::CPUDeviceControllerFactory::get_instance();
    tensor::Tensor t(tensor::DataType_t::fp32, {2, 2}, false, cpu_controller, nullptr);
    // 没有分配内存，byte_size 基于 dims 计算

    auto external_buffer = std::make_shared<base::Buffer>(
        64, nullptr, base::DeviceType_t::Unknown, cpu_controller, false);

    bool result = t.assign(external_buffer);
    EXPECT_TRUE(result);
}

TEST(Tensor, clone) {
    auto cpu_controller = base::CPUDeviceControllerFactory::get_instance();
    tensor::Tensor t(tensor::DataType_t::int32, {2, 3}, true, cpu_controller, nullptr);

    // 写入数据
    int32_t* data = static_cast<int32_t*>(t.get_ptr());
    for (int32_t i = 0; i < 6; ++i) {
        data[i] = i * 100;
    }

    // 深拷贝
    tensor::Tensor t2 = t.clone();

    // 验证形状和类型一致
    EXPECT_EQ(t2.get_data_type(), tensor::DataType_t::int32);
    EXPECT_EQ(t2.get_dims_size(), 2);
    EXPECT_EQ(t2.get_dim(0), 2);
    EXPECT_EQ(t2.get_dim(1), 3);
    EXPECT_EQ(t2.get_size(), 6);

    // 验证数据拷贝一致
    EXPECT_EQ(memcmp(t.get_ptr(), t2.get_ptr(), t.get_byte_size()), 0);

    // 验证是独立拷贝（修改 t2 不影响 t）
    int32_t* t2_data = static_cast<int32_t*>(t2.get_ptr());
    t2_data[0] = 999;
    EXPECT_EQ(static_cast<int32_t*>(t.get_ptr())[0], 0);
}

TEST(Tensor, clone_empty_tensor) {
    auto cpu_controller = base::CPUDeviceControllerFactory::get_instance();
    tensor::Tensor t(tensor::DataType_t::fp32, {2, 2}, true, cpu_controller, nullptr);

    tensor::Tensor t2 = t.clone();

    EXPECT_EQ(t2.get_data_type(), tensor::DataType_t::fp32);
    EXPECT_EQ(t2.get_size(), t.get_size());
    EXPECT_NE(t2.get_ptr(), t.get_ptr());
}

// ============================================================
// peek_index / peek_position 访问元素测试
// ============================================================

TEST(Tensor, peek_index_read) {
    auto cpu_controller = base::CPUDeviceControllerFactory::get_instance();
    tensor::Tensor t(tensor::DataType_t::int32, {5}, true, cpu_controller, nullptr);

    int32_t* data = static_cast<int32_t*>(t.get_ptr());
    for (int32_t i = 0; i < 5; ++i) {
        data[i] = i * 10;
    }

    EXPECT_EQ(t.peek_index<int32_t>(0), 0);
    EXPECT_EQ(t.peek_index<int32_t>(2), 20);
    EXPECT_EQ(t.peek_index<int32_t>(4), 40);
}

TEST(Tensor, peek_index_write) {
    auto cpu_controller = base::CPUDeviceControllerFactory::get_instance();
    tensor::Tensor t(tensor::DataType_t::fp32, {3}, true, cpu_controller, nullptr);

    t.peek_index<float>(0) = 1.5f;
    t.peek_index<float>(1) = 2.5f;
    t.peek_index<float>(2) = 3.5f;

    float* data = static_cast<float*>(t.get_ptr());
    EXPECT_FLOAT_EQ(data[0], 1.5f);
    EXPECT_FLOAT_EQ(data[1], 2.5f);
    EXPECT_FLOAT_EQ(data[2], 3.5f);
}

TEST(Tensor, peek_position_read) {
    auto cpu_controller = base::CPUDeviceControllerFactory::get_instance();
    tensor::Tensor t(tensor::DataType_t::fp32, {2, 3, 4}, true, cpu_controller, nullptr);

    // strides: [12, 4, 1]
    // offset = row*12 + col*4 + ch*1
    float* data = static_cast<float*>(t.get_ptr());
    for (size_t i = 0; i < 24; ++i) {
        data[i] = static_cast<float>(i);
    }

    // pos {0, 0, 0} -> offset 0
    EXPECT_FLOAT_EQ(t.peek_position<float>({0, 0, 0}), 0.0f);
    // pos {0, 0, 3} -> offset 3
    EXPECT_FLOAT_EQ(t.peek_position<float>({0, 0, 3}), 3.0f);
    // pos {0, 1, 0} -> offset 4
    EXPECT_FLOAT_EQ(t.peek_position<float>({0, 1, 0}), 4.0f);
    // pos {1, 0, 0} -> offset 12
    EXPECT_FLOAT_EQ(t.peek_position<float>({1, 0, 0}), 12.0f);
    // pos {1, 2, 3} -> offset 12 + 8 + 3 = 23
    EXPECT_FLOAT_EQ(t.peek_position<float>({1, 2, 3}), 23.0f);
}

TEST(Tensor, peek_position_write) {
    auto cpu_controller = base::CPUDeviceControllerFactory::get_instance();
    tensor::Tensor t(tensor::DataType_t::int32, {2, 3}, true, cpu_controller, nullptr);

    // strides: [3, 1]
    // pos {0, 2} -> offset 2, pos {1, 1} -> offset 4
    t.peek_position<int32_t>({0, 2}) = 100;
    t.peek_position<int32_t>({1, 1}) = 200;

    int32_t* data = static_cast<int32_t*>(t.get_ptr());
    EXPECT_EQ(data[2], 100);   // offset 2
    EXPECT_EQ(data[4], 200);   // offset 4
}

// ============================================================
// DataTypeSize 辅助函数测试
// ============================================================

TEST(DataTypeSize, fp32) {
    EXPECT_EQ(tensor::DataTypeSize(tensor::DataType_t::fp32), sizeof(float));
}

TEST(DataTypeSize, int8) {
    EXPECT_EQ(tensor::DataTypeSize(tensor::DataType_t::int8), sizeof(int8_t));
}

TEST(DataTypeSize, int32) {
    EXPECT_EQ(tensor::DataTypeSize(tensor::DataType_t::int32), sizeof(int32_t));
}

TEST(DataTypeSize, unknown) {
    EXPECT_EQ(tensor::DataTypeSize(tensor::DataType_t::Unknown), 0);
}

// ============================================================
// to 设备转移测试 (需要 CUDA GPU)
// ============================================================

TEST(Tensor, to_cpu_already_cpu) {
    auto cpu_controller = base::CPUDeviceControllerFactory::get_instance();
    tensor::Tensor t(tensor::DataType_t::fp32, {2, 2}, true, cpu_controller, nullptr);

    // 已经在 CPU 上，to("cpu") 应该无变化
    void* ptr_before = t.get_ptr();
    t.to("cpu", nullptr);
    EXPECT_EQ(t.get_ptr(), ptr_before);
    EXPECT_EQ(t.get_device_type(), base::DeviceType_t::CPU);
}
