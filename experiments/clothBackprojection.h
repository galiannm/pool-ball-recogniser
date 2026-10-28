#pragma once

#include <opencv2/opencv.hpp>
#include <filesystem>
#include <string>
#include <vector>

namespace experiments {

struct ClothBackprojectionOptions {
    int hBins = 18;
    int sBins = 16;
    double similarityThreshold = 0.8;
    double maximumClothFraction = 0.9;
    int samplingInsetKernel = 11;
    int exclusionMargin = 4;
    int minimumSamples = 1000;
    double minimumCircleCoverage = 0.8;
    bool enabled = true;
};

struct ClothBackprojectionModel {
    cv::Mat samplingMask;
    cv::Mat histogram;       // CV_32F HxS, min/max-normalised to [0,255].
    cv::Mat similarity8U;    // Direct calcBackProject output, no map renormalisation.
    cv::Mat similarity;      // CV_32F, similarity8U / 255.
    int samples = 0;
    bool valid = false;
    std::string status;
};

struct ClothCandidateScore {
    int validPixels = 0;
    double coverage = 0;
    double meanSimilarity = -1;
    double clothFraction = -1;
    bool retained = true;
    std::string reason;
};

struct ClothVerificationResult {
    std::vector<cv::Vec3f> retained;
    std::vector<int> retainedIndices;
    std::vector<ClothCandidateScore> scores; // Original candidate order.
};

// Fixed geometry only: no annotated ball centres are used to exclude pockets.
cv::Mat makeClothPocketMask(cv::Size sourceSize);

ClothBackprojectionModel estimateClothBackprojection(const cv::Mat& image,
    const cv::Mat& tableMask, const cv::Mat& pocketMask,
    const std::vector<cv::Vec3f>& rawHoughCandidates,
    const ClothBackprojectionOptions& options = {});

ClothVerificationResult verifyClothCandidates(const ClothBackprojectionModel& model,
    const cv::Mat& tableMask, const std::vector<cv::Vec3f>& candidates,
    const ClothBackprojectionOptions& options = {});

void runBackprojectionExperiment(const std::filesystem::path& directory);

} // namespace experiments
