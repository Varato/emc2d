cmake_minimum_required(VERSION 3.9)
# since VERSION 3.10, CUDA becomes an intrinsic language to cmake
# since VERSION 3.12, can use FindPython
# since VERSION 3.17, can use FindCUDAToolkit


# To build cpp extensions for emc2d
project(emc2d C CXX)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# set(CMAKE_CXX_FLAGS_RELEASE "-O3")

add_subdirectory(pybind11)
add_subdirectory(emc2d/extensions)
