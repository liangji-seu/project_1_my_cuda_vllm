#include "base/DeviceController.h"

#include <cstdlib>

namespace base {

// ---- DeviceController ----

DeviceController::DeviceController(DeviceType_t device_type)
    : device_type(device_type) {}

DeviceType_t DeviceController::show_device_type() const {
    return device_type;
}

void* DeviceController::mem_alloc(size_t byte_size) {
    // TODO
    return nullptr;
}

void DeviceController::mem_release(void* ptr) {
    // TODO
}

void DeviceController::mem_copy(const void* src, void* dst, size_t byte_size,
                                DeviceType_t src_device_type, DeviceType_t dst_device_type,
                                void* stream, bool need_sync) {
    // TODO
}

void DeviceController::mem_reset(void* ptr, size_t byte_size, DeviceType_t device_type,
                                 void* stream, bool need_sync) {
    // TODO
}

// ---- CPUDeviceController ----

void* CPUDeviceController::mem_alloc(size_t byte_size) {
    // TODO
    return nullptr;
}

void CPUDeviceController::mem_release(void* ptr) {
    // TODO
}

// ---- GPUDeviceController ----

void* GPUDeviceController::mem_alloc(size_t byte_size) {
    // TODO
    return nullptr;
}

void GPUDeviceController::mem_release(void* ptr) {
    // TODO
}

// ---- CPUDeviceControllerFactory ----

std::shared_ptr<CPUDeviceController> CPUDeviceControllerFactory::controller = nullptr;

std::shared_ptr<CPUDeviceController> CPUDeviceControllerFactory::get_instance() {
    if (controller == nullptr) {
        controller = std::make_shared<CPUDeviceController>();
    }
    return controller;
}

// ---- GPUDeviceControllerFactory ----

std::shared_ptr<GPUDeviceController> GPUDeviceControllerFactory::controller = nullptr;

std::shared_ptr<GPUDeviceController> GPUDeviceControllerFactory::get_instance() {
    if (controller == nullptr) {
        controller = std::make_shared<GPUDeviceController>();
    }
    return controller;
}

} // namespace base
