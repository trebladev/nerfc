//
// Created by xuan on 8/14/23.
//

#ifndef CUDA_ACHIEVE_NERF_LOADER_H
#define CUDA_ACHIEVE_NERF_LOADER_H

#include <opencv2/opencv.hpp>
#include <iostream>
#include <argparse.hpp>
#include "tqdm/tqdm.h"
#include <json/json.h>
#include <fstream>

namespace nerf{
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
        void load_images();
        void count_images();
        void load_configs(std::string config_path);

        Loader(std::string root_dir, std::string mode) : root_dir(root_dir), mode(mode) {}
        ~Loader() = default;
    };
};


#endif //CUDA_ACHIEVE_NERF_LOADER_H
