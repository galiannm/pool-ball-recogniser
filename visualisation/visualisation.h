#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include "evaluation/evaluation.h"

cv::Mat drawDetectedBalls(
    const cv::Mat& image,
    const std::vector<cv::Vec3f>& circles
);

cv::Mat drawValidationMatches(
    const cv::Mat& transformedImage,
    const std::vector<cv::Point2f>& predictions,
    const std::vector<cv::Point2f>& groundTruth,
    const EvaluationResult& result
);

cv::Mat drawGroundTruthOverlay(
    const cv::Mat& transformedImage,
    const std::vector<cv::Point2f>& groundTruth,
    const std::vector<int>& ballIds
);

cv::Mat drawClassifiedBalls(
    const cv::Mat& image,
    const std::vector<cv::Vec3f>& circles,
    const std::vector<std::string>& predictedTypes,
    const std::vector<int>& predictedNumbers = {}
);
