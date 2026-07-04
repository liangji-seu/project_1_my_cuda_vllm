add_executable(test_main test_main.cpp test_device_controller.cpp test_buffer.cpp test_tensor.cpp)

target_link_libraries(test_main
    my_cuda_vllm
    glog::glog
    GTest::gtest
    GTest::gtest_main
    pthread
    cudart
)

target_include_directories(test_main PUBLIC
    ${PROJECT_SOURCE_DIR}/sys/include
)
