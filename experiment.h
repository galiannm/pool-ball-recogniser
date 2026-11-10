#include <opencv2/opencv.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

struct CCAExp {
    int lightCount = 0;
    int nonLightCount = 0;
    double largestLightPct = 0;
    double largestNonLightPct = 0;
    double boundaryLightPct = 0;
    cv::Mat lightMask;
    cv::Mat nonLightMask;
};

struct GuidedExperimentResult {
    cv::Mat foreground;
    std::vector<cv::Vec3f> baseline;
    std::vector<cv::Vec3f> guided;
    std::vector<cv::Vec3f> watershed;
    std::vector<ImagePanel> diagnostics;
};

CCAExp experimentCCA(const BallROI& ball, const KMeansResult& seg);

std::string experimentCCAType(const BallTypeResult& base, const CCAExp& spatial);

void runK2CCAExperiment();

cv::Mat makeExperimentForeground(const cv::Mat& image);

GuidedExperimentResult detectGuidedHoughWatershed(
    const cv::Mat& image,
    const std::vector<cv::Vec3f>& baseline);

void runGuidedHoughWatershedExperiment();

void runGuidedRecognitionExperiment();