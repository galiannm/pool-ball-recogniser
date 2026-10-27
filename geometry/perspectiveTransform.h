#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

// Perspective transformation matrix
cv::Mat getTableTransformation();

// Transform Hough circle centres
std::vector<cv::Point2f> transformCentres(
    const std::vector<cv::Vec3f>& circles
);

// Transform the entire image
cv::Mat transformImage(const cv::Mat& image);

std::vector<cv::Point> getTableCorners();