
#pragma once

#include "extractBall.h"
#include <opencv2/opencv.hpp>

struct ColourStats {
    double whitePercent = 0.0;
    double colourPercent = 0.0;
    double darkPercent = 0.0;
    double otherPercent = 0.0;

    int dominantHue = -1;
    double dominantHuePercent = 0.0;

    // Largest connected regions as percentages
    // of the entire circular ball area.
    double largestWhiteRegionPercent = 0.0;
    double largestColourRegionPercent = 0.0;

    int whiteComponentCount = 0;
    int colourComponentCount = 0;

    // Diagnostic output masks
    cv::Mat whiteMask;
    cv::Mat colourMask;
    cv::Mat darkMask;
    cv::Mat dominantColourMask;
    cv::Mat componentVisualisation;
};

ColourStats analyseBallColours(const BallROI& ball);