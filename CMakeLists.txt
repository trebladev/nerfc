cmake_minimum_required(VERSION 3.0)

project(cuda_achieve LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
# set(CUDA_TOOLKIT_ROOT_DIR "/home/xuan/cuda/cuda-11.7")
# set(CUDA_CUDART_LIBRARY "/home/xuan/cuda/cuda-11.7/lib64/libcudart.so")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")
include(${PROJECT_SOURCE_DIR}/cmake/UseCUDA.cmake)

find_package(OpenCV REQUIRED)
#set(CUDA_HEADERS "${PROJECT_SOURCE_DIR}/3rdparty/cuda-samples/Common")
include_directories(
        ${CUDA_INCLUDE_DIRS}
        #${CUDA_HEADERS}
        ./include
        ${OpenCV_INCLUDE_DIRS}
)

list(APPEND NeRF_SOURCES
    src/nerf/nerf_loader.cu)


# add_subdirectory(src/demo)

cuda_add_executable(nerfc main.cpp ${NeRF_SOURCES})
target_link_libraries(nerfc ${OpenCV_LIBS})
