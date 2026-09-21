cmake_minimum_required(VERSION 3.16)
project(walking_g1)

find_package(unitree_sdk2 REQUIRED)
find_package(onnxruntime REQUIRED)
find_library(ONNX_RUNTIME_LIB onnxruntime)
find_package(Eigen3 REQUIRED CONFIG)
find_package(yaml-cpp REQUIRED)

message(STATUS "ONNX Runtime includes: ${ONNX_RUNTIME_LIB}")
add_executable(walking_g1 src/walking_g1.cpp)
add_executable(walking_g1_mjlab src/walking_g1_mjlab.cpp)
add_executable(walking_g1_mjlab_dolly src/walking_g1_mjlab.cpp)
target_compile_definitions(walking_g1_mjlab_dolly PRIVATE MJLAB_DOLLY_TASK)
add_executable(walking_g1_falcon_dolly src/walking_g1_mjlab.cpp)
target_compile_definitions(walking_g1_falcon_dolly PRIVATE MJLAB_DOLLY_TASK FALCON_LOCOMOTION)
add_executable(walking_g1_homie_dolly src/walking_g1_mjlab.cpp)
target_compile_definitions(walking_g1_homie_dolly PRIVATE MJLAB_DOLLY_TASK HOMIE_LOCOMOTION)
add_executable(arm_sdk_upper_body src/arm_sdk_upper_body.cpp)
option(onnxruntime_USE_CUDA "Build with CUDA support" OFF) # Lets start with cpu


target_link_libraries(walking_g1 PRIVATE 
  unitree_sdk2 
  onnxruntime::onnxruntime
  Eigen3::Eigen)

target_link_libraries(walking_g1_mjlab PRIVATE 
  unitree_sdk2 
  onnxruntime::onnxruntime
  Eigen3::Eigen)

target_link_libraries(walking_g1_mjlab_dolly PRIVATE
  unitree_sdk2
  onnxruntime::onnxruntime
  Eigen3::Eigen
  yaml-cpp)

target_link_libraries(walking_g1_falcon_dolly PRIVATE
  unitree_sdk2
  onnxruntime::onnxruntime
  Eigen3::Eigen
  yaml-cpp)

target_link_libraries(walking_g1_homie_dolly PRIVATE
  unitree_sdk2
  onnxruntime::onnxruntime
  Eigen3::Eigen
  yaml-cpp)

target_link_libraries(arm_sdk_upper_body PRIVATE unitree_sdk2)


SET(CMAKE_BUILD_TYPE Release)
