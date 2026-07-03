add_executable(test_main test_main.cpp)

target_link_libraries(test_main
    my_cuda_vllm
    glog::glog
    gtest
    gtest_main
    pthread
    cudart
)

target_include_directories(test_main PUBLIC
    ${PROJECT_SOURCE_DIR}/sys/include
)
