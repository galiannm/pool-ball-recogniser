#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

std::vector<cv::Vec3f> filterPocketDetections(
    const std::vector<cv::Vec3f>& circles
);