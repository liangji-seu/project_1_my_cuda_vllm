#include <glog/logging.h>
#include <gtest/gtest.h>

#include "base/DeviceController.h"

TEST(DeviceController, cpu_alloc) {
    auto cpu_controller = base::CPUDeviceControllerFactory::get_instance();
    void* ptr = cpu_controller->mem_alloc(32);
    EXPECT_NE(ptr, nullptr);
}
