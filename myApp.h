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

struct BallGroupCounts {
    int annotated = 0;
    int matched = 0;
    int correctColour = 0;
    int correctType = 0;
    int correctIdentity = 0;
};

struct OverallEvaluation {
    int TP = 0;
    int FP = 0;
    int FN = 0;

    int correctTypes = 0;

    int correctColours = 0;
    int matchedColours = 0;
    int colourGroundTruth = 0;

    int correctIdentities = 0;
    int unknownIdentities = 0;

    std::map<std::string, BallGroupCounts> groups;
};

struct EvaluationContext {
    std::vector<LabelledBall> truth;
    EvaluationResult matching;
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

struct Candidate {
    cv::Vec3f circle;
    cv::Point2f tableCentre;
    BallROI roi;
    BallTypeResult baselineType;
    BallColourScores colours;
    std::array<double, 3> typeSupport{};
    std::array<double, 16> identityScore{};
    std::array<bool, 16> eligible{};
    int localBest = -1;
    double bestScore = 0;
};

struct IdentityRecognitionResult {
    std::vector<int> numbers;
    EvaluationResult detection;
    int correct = 0;
    int groundTruthCount = 0;
    int unknown = 0;
    double accuracy = 0;           
    double endToEndAccuracy = 0;  
};

struct CCAExp {
    int lightCount = 0;
    int nonLightCount = 0;
    double largestLightPct = 0;
    double largestNonLightPct = 0;
    double boundaryLightPct = 0;
    cv::Mat lightMask;
    cv::Mat nonLightMask;
};

struct ImagePanel {
    std::string label;
    cv::Mat image;
};

struct GuidedExperimentResult {
    cv::Mat foreground;
    std::vector<cv::Vec3f> baseline;
    std::vector<cv::Vec3f> guided;
    std::vector<cv::Vec3f> watershed;
    std::vector<ImagePanel> diagnostics;
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
    const EvaluationContext& context);

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
    const std::string& filename,
    const EvaluationContext& context);

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

EvaluationContext prepareEvaluation(
    const std::vector<cv::Vec3f>& circles,
    const std::string& filename);

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
    const EvaluationContext& context);

TypeRecognitionResult runTypeRecognition(
    const cv::Mat& image,
    const std::vector<BallROI>& balls,
    const std::vector<cv::Vec3f>& circles,
    const std::filesystem::path& directory,
    const std::string& filename,
    const EvaluationContext& context);

std::string identityType(int number);

void identityEvidence(Candidate& candidate);

Candidate rawWhitePadding(const cv::Mat& image, const Candidate& original,
                         const BallColourModel& model, int padding);

int selectAmbiguousCue(const std::vector<Candidate>& candidates);

void reserveCueIdentity(std::vector<Candidate>& candidates, int selected);

std::vector<int> maximumAssignment(const std::vector<std::vector<double>>& weights);

std::vector<int> assignUnique(const std::vector<Candidate>& candidates);

IdentityRecognitionResult evaluateBallIdentities(
    const std::vector<int>& numbers,
    const EvaluationContext& context);

IdentityRecognitionResult runIdentityRecognition(
    const cv::Mat& image,
    const std::vector<BallROI>& balls,
    const std::vector<cv::Vec3f>& circles,
    const std::vector<BallTypeResult>& types,
    const std::vector<BallColourScores>& colours,
    const BallColourModel& model,
    const std::filesystem::path& directory,
    const std::string& filename,
    const EvaluationContext& context);

void accumulateBallGroups(const std::vector<int>& numbers, const EvaluationContext& context, std::map<std::string, BallGroupCounts>& groups);

void printBallGroupEvaluation(const std::map<std::string, BallGroupCounts>& groups);

void printOverallEvaluation(const OverallEvaluation& totals);

void saveFinalResults(const cv::Mat& image, const std::filesystem::path& directory, const std::vector<cv::Vec3f>& circles, const std::vector<cv::Point2f>& tableCentres, const std::vector<std::string>& types, const std::vector<int>& numbers);

void runGridSearch();

CCAExp experimentCCA(const BallROI& ball, const KMeansResult& seg);

std::string experimentCCAType(const BallTypeResult& base, const CCAExp& spatial);

void runK2CCAExperiment();

cv::Mat makeExperimentForeground(const cv::Mat& image);

GuidedExperimentResult detectGuidedHoughWatershed(
    const cv::Mat& image,
    const std::vector<cv::Vec3f>& baseline);

void runGuidedHoughWatershedExperiment();

void runGuidedRecognitionExperiment();

void runMainPipeline();

void runExperiments();

void MyApplication();
