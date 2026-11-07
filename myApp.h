#pragma once

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

// ========Structs========
struct HoughParams {
    double dp = 1.5;
    double minDist = 18.0;
    double param1 = 75.0;
    double param2 = 35.0;
    int minRadius = 15;
    int maxRadius = 30;
};

struct DetectionResult {
    cv::Mat maskedImage;
    std::vector<cv::Vec3f> rawCircles;
    std::vector<cv::Vec3f> filteredCircles;
};

struct BallROI {
    cv::Mat image;
    cv::Mat mask;
};

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

struct BallTypeResult {
    std::string type = "Unknown";

    double lightPercent = 0;
    double colouredPercent = 0;
    double darkPercent = 0;
};

struct LabelledBall {
    int number; // 1-15, 0 = cue ball
    cv::Point2f centre;
};

struct EvaluationResult {
    int TP = 0;
    int FP = 0;
    int FN = 0;

    double precision = 0;
    double recall = 0;
    double f1 = 0;
    double meanError = 0;

    // Index of matched ground-truth ball for each prediction.
    // -1 means the prediction is a false positive.
    std::vector<int> predictionToGroundTruth;

    // Index of matched prediction for each ground-truth ball.
    // -1 means the ground-truth ball was missed.
    std::vector<int> groundTruthToPrediction;
};

// Function declarations let the definitions appear in any order in myApp.cpp.

std::vector<std::string> getAllImages();

std::vector<cv::Point> getTableCorners();

cv::Mat getTableTransformation();

std::vector<cv::Point2f> transformCentres(const std::vector<cv::Vec3f>& circles);

std::vector<cv::Vec3f> filterPocketDetections(const std::vector<cv::Vec3f>& circles);

cv::Mat maskTable(const cv::Mat& image);

BallROI extractBall(const cv::Mat& image, const cv::Vec3f& detectedCircle);

cv::Mat transformImage(const cv::Mat& image);

void saveImage(const std::filesystem::path& path, const cv::Mat& image);

void finishOutputFile(std::ofstream& file);

std::string finalBallLabel(int number);

cv::Scalar diagramBallColour(int number);

std::vector<LabelledBall> loadLabelledGroundTruth(const std::string& csvPath, const std::string& imageName);

std::vector<cv::Point2f> loadGroundTruth(
    const std::string& csvPath,
    const std::string& imageName,
    std::vector<int>* ballIds = nullptr);

cv::Mat drawDetectedBalls(const cv::Mat& image, const std::vector<cv::Vec3f>& circles);

void drawFinalLabel(cv::Mat& image, cv::Point centre, int number, cv::Point offset = {});

cv::Mat drawTableDiagram(const std::vector<cv::Point2f>& centres, const std::vector<int>& numbers);

std::vector<cv::Vec3f> detectBalls(const cv::Mat& image, const HoughParams& params);

DetectionResult runDetection(const cv::Mat& image, 
                            const HoughParams& params, 
                            const std::filesystem::path directory, 
                            const std::string& filename);

std::vector<BallROI> runExtraction(
    const cv::Mat& image,
    const std::vector<cv::Vec3f>& circles,
    const std::filesystem::path& directory);

EvaluationResult evaluateDetections(const std::vector<cv::Point2f>& predictions, const std::vector<cv::Point2f>& groundTruth, double tolerance);

void saveFinalResults(const cv::Mat& image, const std::filesystem::path& directory, const std::vector<cv::Vec3f>& circles, const std::vector<cv::Point2f>& tableCentres, const std::vector<std::string>& types, const std::vector<int>& numbers);

void runGridSearch();

void runMainPipeline();

void runExperiments();

void MyApplication();
