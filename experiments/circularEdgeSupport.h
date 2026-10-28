#pragma once

#include <opencv2/opencv.hpp>
#include <filesystem>
#include <vector>

namespace experiments {

struct CircularEdgeOptions {
    int angularSamples = 72; // Five-degree sectors.
    double tolerance = 2.5; // Source-image pixels, around the predicted boundary.
    double minimumRadialAlignment = 0.6;
    double cannyHigh = 75;
};

struct CircularEdgeMaps {
    cv::Mat edges, gradientX, gradientY, tableMask;
};

struct CircularEdgeSample {
    cv::Point2f boundary, edge = {-1, -1};
    bool valid = false, nearbyEdge = false, supported = false;
    double alignment = 0;
};

struct CircularEdgeScore {
    int validSamples = 0;
    double coverage = 0, nearbyEdgeFraction = 0, supportFraction = 0;
    double longestArcDegrees = 0, bridgedArcDegrees = 0;
    std::vector<CircularEdgeSample> samples;
};

// No annotations or candidate filtering: these functions measure evidence only.
CircularEdgeMaps prepareCircularEdgeMaps(const cv::Mat& image, const cv::Mat& tableMask,
                                        const CircularEdgeOptions& options = {});
CircularEdgeScore measureCircularEdgeSupport(const CircularEdgeMaps& maps, const cv::Vec3f& circle,
                                             const CircularEdgeOptions& options = {});
void runCircularEdgeExperiment(const std::filesystem::path& directory);

} // namespace experiments
