#pragma once

#include <opencv2/opencv.hpp>

struct BallROI
{
    cv::Mat image;
    cv::Mat mask;
};

BallROI extractBall(
    const cv::Mat& image,
    const cv::Vec3f& detectedCircle
);
