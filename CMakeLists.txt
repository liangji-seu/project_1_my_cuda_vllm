cmake_minimum_required(VERSION 3.16)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set the CUDA compiler path
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")

project(my_cuda_vllm CXX CUDA)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)

# ---- Dependencies ----
find_package(GTest REQUIRED)
find_package(glog REQUIRED)
find_package(Armadillo REQUIRED)

# ---- Library target ----
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

aux_source_directory(sys/src/base DIR_BASE)
aux_source_directory(sys/src/tensor DIR_TENSOR)
aux_source_directory(sys/src/op DIR_OP)
aux_source_directory(sys/src/model DIR_MODEL)
aux_source_directory(sys/src/sampler DIR_SAMPLER)

add_library(my_cuda_vllm SHARED
  ${DIR_BASE} ${DIR_TENSOR} ${DIR_OP} ${DIR_MODEL} ${DIR_SAMPLER}
)

target_link_libraries(my_cuda_vllm
  glog::glog gtest gtest_main pthread cudart armadillo
)

target_link_directories(my_cuda_vllm PUBLIC ${CMAKE_CUDA_COMPILER_LIBRARY_ROOT}/lib64)

target_include_directories(my_cuda_vllm PUBLIC
  ${glog_INCLUDE_DIR}
  ${Armadillo_INCLUDE_DIR}
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  ${PROJECT_SOURCE_DIR}/sys/include
)
