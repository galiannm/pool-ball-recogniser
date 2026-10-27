
#pragma once

#include "extractBall.h"
#include "kMeansSegmentation.h"
#include <vector>

struct CCAFeatures {
    // Light region measurements
    int lightComponents = 0;
    double largestLightComponentPercent = 0;
    double lightBoundaryCoverage = 0;

    // Coloured/dark region measurements
    int nonLightComponents = 0;
    double largestNonLightComponentPercent = 0;

    cv::Mat lightMask;
    cv::Mat nonLightMask;
    cv::Mat boundaryBand;

    std::vector<cv::Mat> lightRegions;
    std::vector<cv::Mat> nonLightRegions;
};

CCAFeatures analyseConnectedComponents(
    const BallROI& ball,
    const KMeansResult& segmentation
);
