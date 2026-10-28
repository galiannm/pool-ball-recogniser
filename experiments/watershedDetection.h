#pragma once

#include <opencv2/opencv.hpp>
#include <filesystem>
#include <string>
#include <vector>

namespace experiments {

struct WatershedOptions {
    double groupDistanceFactor = 1.4; // Distance relative to the sum of Hough radii.
    double cropPaddingRadii = 2.0;
    int minClothSamples = 40;
    double clothMinDistance = 16;
    double clothMaxDistance = 45;
    double clothMADMultiplier = 2.5;
    double houghCoreRadius = 0.65;
    int closingKernel = 3;
    double peakNeighbourhoodRadii = 0.9;
    double minimumPeakRadii = 0.45;
    double minimumMarkerSeparationRadii = 1.15;
    double cropEdgeMarginRadii = 0.6;
    int seedRadius = 3;
    double minBasinAreaRatio = 0.25;
    double maxBasinAreaRatio = 1.8;
    double minCircularity = 0.35;
    double maxAspectRatio = 2.2;
    double minForegroundSupport = 0.35;
};

struct WatershedProposal {
    cv::Point localCentre;
    double peakDistance = 0;
    int label = 0;
    double areaRatio = 0, circularity = 0, aspectRatio = 0, foregroundSupport = 0;
    bool accepted = false;
    std::string reason;
};

struct WatershedRegion {
    cv::Rect crop;
    double expectedRadius = 0;
    int clothSamples = 0;
    double clothDistanceThreshold = 0;
    std::string status;
    std::vector<int> houghIndices;
    std::vector<WatershedProposal> proposals;
    cv::Mat foreground, distance, seededMarkers, hybridMarkers, seededLabels, hybridLabels;
};

struct WatershedDetectionResult {
    // The Hough-seeded control deliberately keeps the original circles.
    std::vector<cv::Vec3f> hybridCircles;
    std::vector<WatershedRegion> regions;
};

// No ground truth enters grouping, masks, markers, segmentation or acceptance.
WatershedDetectionResult segmentCrowdedBalls(const cv::Mat& image,
    const std::vector<cv::Vec3f>& baseline, const WatershedOptions& options = {});

void runWatershedDetectionExperiment(const std::filesystem::path& outputDirectory);

} // namespace experiments
