cmake_minimum_required(VERSION 3.16)
project(walking_g1)

find_package(unitree_sdk2 REQUIRED)
find_package(onnxruntime REQUIRED)
find_library(ONNX_RUNTIME_LIB onnxruntime)
find_package(Eigen3 REQUIRED CONFIG)

message(STATUS "ONNX Runtime includes: ${ONNX_RUNTIME_LIB}")
add_executable(walking_g1 src/walking_g1.cpp)
add_executable(walking_g1_mjlab src/walking_g1_mjlab.cpp)
option(onnxruntime_USE_CUDA "Build with CUDA support" OFF) # Lets start with cpu


target_link_libraries(walking_g1 PRIVATE 
  unitree_sdk2 
  onnxruntime::onnxruntime
  Eigen3::Eigen)

target_link_libraries(walking_g1_mjlab PRIVATE 
  unitree_sdk2 
  onnxruntime::onnxruntime
  Eigen3::Eigen)


SET(CMAKE_BUILD_TYPE Release)
