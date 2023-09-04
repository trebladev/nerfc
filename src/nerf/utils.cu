//
// Created by xuan on 8/25/23.
//

#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

//void plotRays(const Eigen::MatrixXf& rays_o, const Eigen::MatrixXf& rays_d, const Eigen::VectorXf& t){
//    int imageSize = 800;
//    cv::Mat canvas(imageSize, imageSize, CV_8UC3, cv::Scalar(255, 255, 255));
//
//    Eigen::MatrixXf pt1 = rays_o;
//    Eigen::MatrixXf pt2 = rays_o + rays_d.cwiseProduct(t.replicate(1, 3));
//
//    for (int i = 0; i < pt1.cols(); ++i) {
//        cv::Point2d p1(pt1(0, i), pt1(1, i));
//        cv::Point2d p2(pt2(0, i), pt2(1, i));
//        cv::line(canvas, p1, p2, cv::Scalar(0, 0, 0));
//    }
//
//    cv::imshow("rays", canvas);
//    cv::waitKey();
//
//}
