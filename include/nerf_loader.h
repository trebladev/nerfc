//
// Created by xuan on 8/14/23.
//

#ifndef CUDA_ACHIEVE_NERF_LOADER_H
#define CUDA_ACHIEVE_NERF_LOADER_H

#include <opencv2/opencv.hpp>
#include <iostream>
#include <argparse.hpp>

namespace nerf{
    class Loader {
    public:
        int W;
        int H;
        int N;
        std::string root_dir;
        argparse::ArgumentParser arg;

        std::vector<cv::Mat> images;
        void load_images(std::string root_dir);

    };
};


#endif //CUDA_ACHIEVE_NERF_LOADER_H
