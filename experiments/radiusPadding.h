#pragma once

#include "recognition/extractBall.h"
#include <filesystem>

namespace experiments {

struct RadiusPaddingOptions {
    int padding = 0;
    bool filterCloth = false;
    int ringWidth = 6;
    // OpenCV HSV: hue 0-179, saturation/value 0-255.
    int hueTolerance = 10;
    int saturationTolerance = 70;
    int valueTolerance = 70;
};

BallROI extractPaddedBall(const cv::Mat& image,
                          const cv::Vec3f& circle,
                          const RadiusPaddingOptions& options);

void runRadiusPaddingExperiment(const std::filesystem::path& imagePath,
                                const std::filesystem::path& outputDirectory);

} // namespace experiments
