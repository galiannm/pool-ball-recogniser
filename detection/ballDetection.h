#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

#include "config.h"

struct DetectionResult
{
    cv::Mat maskedImage;
    std::vector<cv::Vec3f> rawCircles;
    std::vector<cv::Vec3f> filteredCircles;
};

std::vector<cv::Vec3f> detectBalls(
    const cv::Mat& image,
    const HoughParams& params
);

std::vector<cv::Vec3f> detectAndFilterBalls(
    const cv::Mat& image,
    const HoughParams& params
);

DetectionResult runDetection(
    const cv::Mat& image,
    const HoughParams& params
);

cv::Mat maskTable(const cv::Mat& image);