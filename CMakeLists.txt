cmake_minimum_required(VERSION 3.28.0)

project(OpenInference CUDA CXX)

set(CUTLASS_PATH ${CMAKE_SOURCE_DIR}/third_party/cutlass)

add_subdirectory(${CUTLASS_PATH})