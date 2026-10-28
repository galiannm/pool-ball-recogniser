#pragma once

#include <opencv2/opencv.hpp>
#include <filesystem>
#include <string>
#include <vector>

namespace experiments {

struct SegmentationHoughOptions {
    int clothTileSize = 128;
    int minimumClothSamples = 30;
    double clothMinimumDistance = 16;
    double clothMaximumDistance = 45;
    double clothMADMultiplier = 2.5;
    int tableInsetKernel = 41; // Suppress rail/shadow connections before component labelling.
    int closingKernel = 3;
    int minimumComponentArea = 80;
    double clusterAreaRatio = 1.6;
    double clusterAspectRatio = 1.45;
    double elongatedMinimumAreaRatio = 0.8;
    double cropPaddingRadii = 1.25;
    double routingOverlapRadii = 0.5;
    double localAccumulatorThreshold = 22;
    double localMinCentreDistance = 12;
    double localMinRadiusRatio = 0.7;
    double localMaxRadiusRatio = 1.6;
    int absoluteMinLocalRadius = 12;
    int absoluteMaxLocalRadius = 32;
    int angularSamples = 72;
    double edgeTolerance = 2.5;
    double minimumRadialAlignment = 0.6;
    double minimumEdgeSupport = 0.30;
    double minimumArcDegrees = 75;
    double minimumForegroundSupport = 0.35;
    double duplicateDistanceRadii = 0.8;
    double watershedMinimumAreaRatio = 0.15;
    double watershedMaximumAreaRatio = 2.0;
};

struct GuidedProposal {
    cv::Vec3f circle;
    int region = 0;
    double edgeSupport = 0, arcDegrees = 0, foregroundSupport = 0, quality = 0;
    bool accepted = false, watershedAccepted = false;
    int guidedIndex = -1;
    double watershedAreaRatio = 0;
    std::string reason, watershedReason;
};

struct GuidedCluster {
    cv::Rect core, crop;
    double expectedRadius = 0;
    int minRadius = 0, maxRadius = 0;
    bool usedEmptySearchFallback = false;
    cv::Mat markers, labels;
    std::vector<int> seedIndices;
};

struct SegmentationGuidedResult {
    cv::Mat foreground, routingMask;
    std::vector<GuidedCluster> clusters;
    std::vector<GuidedProposal> proposals;
    std::vector<cv::Vec3f> baseline, standardOutside, guided, watershed;
};

// Foreground and cluster windows are constructed before Hough is run.
// Ground truth is not an input to the detection algorithm.
SegmentationGuidedResult detectSegmentationGuided(const cv::Mat& image,
    const SegmentationHoughOptions& options = {});

void runSegmentationGuidedExperiment(const std::filesystem::path& directory);

} // namespace experiments
