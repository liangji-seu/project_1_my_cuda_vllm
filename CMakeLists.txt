cmake_minimum_required(VERSION 3.16)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set the CUDA compiler path
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")

project(my_cuda_vllm CXX CUDA)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)

option(LLAMA3_SUPPORT "Enable Llama3 BPE tokenizer support" ON)
if (LLAMA3_SUPPORT)
  add_definitions(-DLLAMA3_SUPPORT)
endif()

# ---- Dependencies ----
find_package(GTest REQUIRED)
find_package(glog REQUIRED)
find_package(Armadillo REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(FetchContent)

FetchContent_Declare(
  absl
  GIT_REPOSITORY https://github.com/abseil/abseil-cpp.git
  GIT_TAG 20240722.0
)
set(ABSL_PROPAGATE_CXX_STD ON CACHE BOOL "" FORCE)
set(ABSL_ENABLE_INSTALL ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(absl)

FetchContent_Declare(
  re2
  GIT_REPOSITORY https://github.com/google/re2.git
  GIT_TAG 2024-07-02
)
FetchContent_MakeAvailable(re2)

FetchContent_Declare(
  nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(nlohmann_json)

# ---- Library target ----
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

aux_source_directory(sys/src/base DIR_BASE)
aux_source_directory(sys/src/tensor DIR_TENSOR)
aux_source_directory(sys/src/op DIR_OP)
aux_source_directory(sys/src/op/kernel DIR_OP_KERNEL)
aux_source_directory(sys/src/op/kernel/cpu DIR_OP_KERNEL_CPU)
aux_source_directory(sys/src/op/kernel/gpu DIR_OP_KERNEL_GPU)
aux_source_directory(sys/src/model DIR_MODEL)
aux_source_directory(sys/src/sampler DIR_SAMPLER)

add_library(my_cuda_vllm SHARED
  ${DIR_BASE} ${DIR_TENSOR} ${DIR_OP} ${DIR_OP_KERNEL} ${DIR_OP_KERNEL_CPU} ${DIR_OP_KERNEL_GPU} ${DIR_MODEL} ${DIR_SAMPLER}
)

target_link_libraries(my_cuda_vllm
  glog::glog GTest::gtest GTest::gtest_main pthread cudart armadillo
  absl::strings re2::re2 nlohmann_json::nlohmann_json
)

target_link_directories(my_cuda_vllm PUBLIC ${CMAKE_CUDA_COMPILER_LIBRARY_ROOT}/lib64)

target_include_directories(my_cuda_vllm PUBLIC
  ${glog_INCLUDE_DIR}
  ${Armadillo_INCLUDE_DIR}
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  ${PROJECT_SOURCE_DIR}/sys/include
)

# ---- Test target ----
add_subdirectory(test)

# ---- Demo target ----
add_subdirectory(demo)
