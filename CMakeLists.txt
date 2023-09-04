cmake_minimum_required(VERSION 3.0)

project(cuda_achieve LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
# set(CUDA_TOOLKIT_ROOT_DIR "/home/xuan/cuda/cuda-11.7")
# set(CUDA_CUDART_LIBRARY "/home/xuan/cuda/cuda-11.7/lib64/libcudart.so")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")
include(${PROJECT_SOURCE_DIR}/cmake/UseCUDA.cmake)

set(PCL_DIR "/usr/local/pcl-1.12")
set(OpenCV_DIR "/home/xuan/Lib/opencv-4.5.3/build")
find_package(VTK REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PCL 1.12.0 REQUIRED)

set(Eigen3_DIR CMAKE_INSTALL_PREFIX /usr/local/share/eigen3/cmake)
find_package(Eigen3 REQUIRED)
#set(CUDA_HEADERS "${PROJECT_SOURCE_DIR}/3rdparty/cuda-samples/Common")
include_directories(
        ${CUDA_INCLUDE_DIRS}
        #${CUDA_HEADERS}
        ./include
        ./include/tqdm
        ${OpenCV_INCLUDE_DIRS}
        ${PCL_INCLUDE_DIRS}
)

list(APPEND NeRF_SOURCES
    src/nerf/nerf_loader.cu
    src/nerf/utils.cu)

link_directories(${PCL_LIBRARY_DIRS})

# add_subdirectory(src/demo)

cuda_add_executable(nerfc main.cpp ${NeRF_SOURCES})
target_link_libraries(nerfc ${OpenCV_LIBS} Eigen3::Eigen)

cuda_add_executable(volume_rendeding test/volume_rendeding.cpp ${NeRF_SOURCES})
target_link_libraries(volume_rendeding ${OpenCV_LIBS} Eigen3::Eigen ${PCL_LIBRARIES})
