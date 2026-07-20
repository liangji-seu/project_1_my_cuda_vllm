set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

add_executable(demo main.cpp)
target_link_libraries(demo my_cuda_vllm glog::glog)

add_executable(chat chat.cpp)
target_link_libraries(chat my_cuda_vllm glog::glog)
