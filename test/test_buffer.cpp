#include <glog/logging.h>
#include <gtest/gtest.h>

#include "base/Buffer.h"
#include "base/DeviceController.h"

TEST(Buffer, cpu_alloc) {
    auto cpu_controller = base::CPUDeviceControllerFactory::get_instance();
    base::Buffer buffer(32, nullptr, base::DeviceType_t::Unknown, cpu_controller, false);

    EXPECT_NE(buffer.get_ptr(), nullptr);
    EXPECT_EQ(buffer.get_byte_size(), 32);
    EXPECT_EQ(buffer.get_device_type(), base::DeviceType_t::CPU);
    EXPECT_EQ(buffer.get_flag_is_external(), false);
}

TEST(Buffer, get_ptr) {
    auto cpu_controller = base::CPUDeviceControllerFactory::get_instance();
    base::Buffer buffer(64, nullptr, base::DeviceType_t::Unknown, cpu_controller, false);

    void* ptr = buffer.get_ptr();
    EXPECT_NE(ptr, nullptr);
}

TEST(Buffer, get_byte_size) {
    auto cpu_controller = base::CPUDeviceControllerFactory::get_instance();
    base::Buffer buffer(128, nullptr, base::DeviceType_t::Unknown, cpu_controller, false);

    EXPECT_EQ(buffer.get_byte_size(), 128);
}

TEST(Buffer, get_device_type) {
    auto cpu_controller = base::CPUDeviceControllerFactory::get_instance();
    base::Buffer buffer(32, nullptr, base::DeviceType_t::Unknown, cpu_controller, false);

    EXPECT_EQ(buffer.get_device_type(), base::DeviceType_t::CPU);
}

TEST(Buffer, get_controller) {
    auto cpu_controller = base::CPUDeviceControllerFactory::get_instance();
    base::Buffer buffer(32, nullptr, base::DeviceType_t::Unknown, cpu_controller, false);

    EXPECT_EQ(buffer.get_controller(), cpu_controller);
}

TEST(Buffer, get_flag_is_external) {
    auto cpu_controller = base::CPUDeviceControllerFactory::get_instance();
    base::Buffer buffer(32, nullptr, base::DeviceType_t::Unknown, cpu_controller, false);

    EXPECT_FALSE(buffer.get_flag_is_external());
}
