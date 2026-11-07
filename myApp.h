#pragma once

#include "visualisation/combineImages.h"
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

struct TypeRecognitionResult {
    std::vector<BallTypeResult> types;
    int matched = 0;
    int correct = 0;
    int groundTruthCount = 0;
    double accuracy = 0;          // Correct types / matched detections.
    double endToEndAccuracy = 0;  // Correct types / all annotated balls.
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

struct DetectionResult {
    cv::Mat maskedImage;
    std::vector<cv::Vec3f> rawCircles;
    std::vector<cv::Vec3f> filteredCircles;
    EvaluationResult evaluation;
};

struct BallHistogramOptions {
    int hueBins = 18, saturationBins = 8, toneBins = 8;
    int lightMinV = 170, lightMaxS = 165;
    int erosionKernel = 3, minimumBodyPixels = 12, neutralChroma = 15;
};

struct LabelledColourSample {
    int number; // Training labels only: 1–15; cue (0) is excluded.
    BallROI ball;
};

struct BallColourModel {
    BallHistogramOptions options;
    std::array<cv::Mat, 8> histograms; // Colours 1–8; 9–15 share colours 1–7.
    std::array<int, 8> examples{};
};

struct BallColourScores {
    cv::Mat bodyMask;
    std::array<double, 8> similarity{};
    int pixels = 0, bestColour = -1;
    double margin = 0;
};

struct ColourRecognitionResult {
    std::vector<BallColourScores> colours;
    int matched = 0;          // Matched numbered balls; excludes cue.
    int correct = 0;
    int groundTruthCount = 0; // Annotated numbered balls; excludes cue.
    double accuracy = 0;
    double endToEndAccuracy = 0;
};

// HSV histogram colour recognition.
const char* bodyColourName(int colour);

void validate(const BallHistogramOptions& options);

cv::Mat makeColourFeatures(const BallROI& ball, const BallHistogramOptions& options);

cv::Mat smoothHistogram(const cv::Mat& histogram, const BallHistogramOptions& options);

cv::Mat histogramBodyMask(const BallROI& ball, const BallHistogramOptions& options = {});

cv::Mat buildColourHistogram(const cv::Mat& features, const cv::Mat& mask,
                            const BallHistogramOptions& options);

BallColourModel buildReferenceHistograms(
    const std::vector<LabelledColourSample>& samples,
    const BallHistogramOptions& options = {});

BallColourModel buildColourReferenceModel(const cv::Mat& referenceImage);

BallColourScores scoreBallColours(const BallROI& ball, const BallColourModel& model);

ColourRecognitionResult evaluateBallColours(
    const std::vector<BallColourScores>& colours,
    const std::vector<cv::Vec3f>& circles,
    const std::string& filename);

cv::Mat drawBallColours(
    const cv::Mat& image,
    const std::vector<cv::Vec3f>& circles,
    const std::vector<BallColourScores>& colours);

cv::Mat drawBodyPixelsGrid(
    const std::vector<BallROI>& balls,
    const std::vector<BallColourScores>& colours);

ColourRecognitionResult runColourRecognition(
    const cv::Mat& image,
    const std::vector<BallROI>& balls,
    const std::vector<cv::Vec3f>& circles,
    const BallColourModel& model,
    const std::filesystem::path& directory,
    const std::string& filename);

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

void saveDetectionEvaluation(const std::filesystem::path& directory, const EvaluationResult& evaluation);

DetectionResult runDetection(const cv::Mat& image, 
                            const HoughParams& params, 
                            const std::filesystem::path directory, 
                            const std::string& filename);

std::vector<BallROI> runExtraction(
    const cv::Mat& image,
    const std::vector<cv::Vec3f>& circles,
    const std::filesystem::path& directory);

EvaluationResult evaluateDetections(const std::vector<cv::Point2f>& predictions, const std::vector<cv::Point2f>& groundTruth, double tolerance);

KMeansResult segmentBallKMeans(const BallROI& ball, int k);

std::vector<KMeansResult> runKMeans(
    const std::vector<BallROI>& balls,
    int k,
    const std::filesystem::path& directory);

cv::Mat drawBallTypes(
    const cv::Mat& image,
    const std::vector<cv::Vec3f>& circles,
    const std::vector<BallTypeResult>& types);

BallTypeResult classifyBallType(const KMeansResult& segmentation);

TypeRecognitionResult evaluateBallTypes(
    const std::vector<BallTypeResult>& types,
    const std::vector<cv::Vec3f>& circles,
    const std::string& filename);

TypeRecognitionResult runTypeRecognition(
    const cv::Mat& image,
    const std::vector<BallROI>& balls,
    const std::vector<cv::Vec3f>& circles,
    const std::filesystem::path& directory,
    const std::string& filename);

void saveFinalResults(const cv::Mat& image, const std::filesystem::path& directory, const std::vector<cv::Vec3f>& circles, const std::vector<cv::Point2f>& tableCentres, const std::vector<std::string>& types, const std::vector<int>& numbers);

void runGridSearch();

void runMainPipeline();

void runExperiments();

void MyApplication();
