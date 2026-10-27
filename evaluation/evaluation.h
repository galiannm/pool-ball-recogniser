
#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

struct EvaluationResult
{
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

std::vector<cv::Point2f> loadGroundTruth(
    const std::string& csvPath,
    const std::string& imageName,
    std::vector<int>* ballIds = nullptr
);

EvaluationResult evaluateDetections(
    const std::vector<cv::Point2f>& predictions,
    const std::vector<cv::Point2f>& groundTruth,
    double tolerance
);

struct LabelledBall {
    int number; // 1-15, 0 = cue ball
    cv::Point2f centre;
};

std::vector<LabelledBall> loadLabelledGroundTruth(
    const std::string& csvPath,
    const std::string& imageName
);