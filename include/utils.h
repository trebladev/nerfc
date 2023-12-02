//
// Created by xuan on 6/20/23.
//

#ifndef CUDA_ACHIEVE_UTILS_H
#define CUDA_ACHIEVE_UTILS_H

#include <cuda_runtime.h>
#include <string.h>
#include <sys/time.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>

#define CHECK(call)                                          \
  {                                                          \
    const cudaError_t error = call;                          \
    if (error != cudaSuccess) {                              \
      fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
      fprintf(stderr, "code: %d, reason: %s\n", error,       \
              cudaGetErrorString(error));                    \
      exit(1);                                               \
    }                                                        \
  }

#define CHECK_CUBLAS(call)                                             \
  {                                                                    \
    cublasStatus_t err;                                                \
    if ((err = (call)) != CUBLAS_STATUS_SUCCESS) {                     \
      fprintf(stderr, "Got CUBLAS error %d at %s:%d\n", err, __FILE__, \
              __LINE__);                                               \
      exit(1);                                                         \
    }                                                                  \
  }

#define CHECK_CURAND(call)                                             \
  {                                                                    \
    curandStatus_t err;                                                \
    if ((err = (call)) != CURAND_STATUS_SUCCESS) {                     \
      fprintf(stderr, "Got CURAND error %d at %s:%d\n", err, __FILE__, \
              __LINE__);                                               \
      exit(1);                                                         \
    }                                                                  \
  }

#define CHECK_CUFFT(call)                                             \
  {                                                                   \
    cufftResult err;                                                  \
    if ((err = (call)) != CUFFT_SUCCESS) {                            \
      fprintf(stderr, "Got CUFFT error %d at %s:%d\n", err, __FILE__, \
              __LINE__);                                              \
      exit(1);                                                        \
    }                                                                 \
  }

#define CHECK_CUSPARSE(call)                                               \
  {                                                                        \
    cusparseStatus_t err;                                                  \
    if ((err = (call)) != CUSPARSE_STATUS_SUCCESS) {                       \
      fprintf(stderr, "Got error %d at %s:%d\n", err, __FILE__, __LINE__); \
      cudaError_t cuda_err = cudaGetLastError();                           \
      if (cuda_err != cudaSuccess) {                                       \
        fprintf(stderr, "  CUDA error \"%s\" also detected\n",             \
                cudaGetErrorString(cuda_err));                             \
      }                                                                    \
      exit(1);                                                             \
    }                                                                      \
  }

inline double seconds() {
  struct timeval tp;
  struct timezone tzp;
  int i = gettimeofday(&tp, &tzp);
  return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

struct VariableInfo {
  std::string name;
  void* data;
  size_t size;

  VariableInfo(const std::string& n, void* d, size_t s)
      : name(n), data(d), size(s) {}
};

// std::vector<VariableInfo> uploadedVariables;

// template <typename T>
// void tocuda(T& data, T& data_cuda, int size) {
//   CHECK(cudaMalloc((void**)&data_cuda, size * sizeof(T)));
//   CHECK(cudaMemcpy(data_cuda, data, size * sizeof(T),
//   cudaMemcpyHostToDevice));
//   // store the variable info which on GPU
//   uploadedVariables.push_back(
//       VariableInfo(std::string(data), data_cuda, size * sizeof(T)));
// }

#define TIC(NAME)                                               \
  std::chrono::steady_clock::time_point __TIMER##NAME##_START = \
      std::chrono::steady_clock::now();

#define TOC(NAME)                                                       \
  std::chrono::steady_clock::time_point __TIMER##NAME##_END =           \
      std::chrono::steady_clock::now();                                 \
  {                                                                     \
    auto d = std::chrono::duration_cast<std::chrono::duration<double>>( \
                 __TIMER##NAME##_END - __TIMER##NAME##_START)           \
                 .count();                                              \
    std::cout << "Timer: " << #NAME << " " << (d) * 1000 << "ms"        \
              << "\n";                                                  \
  }

//  void plotRays(const Eigen::MatrixXf& rays_o, const Eigen::MatrixXf& rays_d,
//  const Eigen::VectorXf& t);

#endif  // CUDA_ACHIEVE_UTILS_H
