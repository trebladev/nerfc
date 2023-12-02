//
// Created by xuan on 8/14/23.
//

#ifndef CUDA_ACHIEVE_NERF_LOADER_H
#define CUDA_ACHIEVE_NERF_LOADER_H

#include <json/json.h>
#include <math.h>
#include <stdio.h>
#include <utils.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <argparse.hpp>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

#include "tqdm/tqdm.h"

namespace nerf {
class Loader {
 public:
  int W;
  int H;
  int N;
  int C;
  int num_image;
  std::string mode;
  std::string root_dir;
  argparse::ArgumentParser arg;

  std::vector<cv::Mat> images;
  std::vector<Eigen::Matrix4f> poses;
  void load_images();
  void count_images();
  void load_configs(std::string config_path);
  void generate_rays(int image_idx, int batch_size,
                     std::vector<Eigen::Vector3f> rays_d,
                     std::vector<Eigen::Vector3f> rays_o,
                     std::vector<Eigen::Vector4i> colors);
  void print();

  Loader(std::string root_dir, std::string mode)
      : root_dir(root_dir), mode(mode) {}
  ~Loader() = default;

 private:
  struct intrinsics {
    float f_x;
    float f_y;
    float c_x;
    float c_y;
  };

  intrinsics Intrinsics;

  float near;
  float far;

  Eigen::Matrix4f projection;
};
};  // namespace nerf

#endif  // CUDA_ACHIEVE_NERF_LOADER_H
