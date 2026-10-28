#pragma once

#include "recognition/extractBall.h"
#include <array>
#include <filesystem>
#include <string>
#include <vector>

namespace experiments {
enum class BallHistogramSpace { HS, HSV, HLS };

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
    BallHistogramSpace space;
    BallHistogramOptions options;
    std::array<cv::Mat, 8> histograms; // Colours 1–8; 9–15 share colours 1–7.
    std::array<int, 8> examples{};
};

struct BallColourScores {
    cv::Mat bodyMask;
    std::array<double, 8> similarity{};
    std::array<cv::Mat, 8> backprojections; // Float histogram mass per pixel.
    int pixels = 0, bestColour = -1;
    double margin = 0;
};

const char* histogramSpaceName(BallHistogramSpace space);
const char* bodyColourName(int colour);
cv::Mat histogramBodyMask(const BallROI& ball, const BallHistogramOptions& options = {});
BallColourModel trainBallColourModel(const std::vector<LabelledColourSample>& samples,
    BallHistogramSpace space, const BallHistogramOptions& options = {});
// Inference has no ground-truth input and cannot change the trained model.
BallColourScores scoreBallColours(const BallROI& ball, const BallColourModel& model);
int histogramBallNumber(const std::string& predictedType, int bestColour);
void runBallColourHistogramExperiment(const std::filesystem::path& directory);
} // namespace experiments
