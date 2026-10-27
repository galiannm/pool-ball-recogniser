#pragma once

#include "extractBall.h"
#include <opencv2/opencv.hpp>
#include <vector>

struct KMeansRegion {
    cv::Mat mask;

    double percentage = 0;

    // Average colour of cluster (OpenCV BGR)
    cv::Vec3b colour = cv::Vec3b(0, 0, 0);
    cv::Vec3b hsv = cv::Vec3b(0, 0, 0);
};

struct KMeansResult {
    int k = 0;
    cv::Mat segmentedImage;
    std::vector<KMeansRegion> regions;

    // Hue comparaisons between the two largest clusters
    double hueDifference = -1;
    double saturationDifference = 0;
    double brightnessDifference = 0;
    double bgrDistance = 0;
};

KMeansResult segmentBallKMeans(
    const BallROI& ball,
    int k
);
