#pragma once

#include "recognitionSearchConfig.h"
#include <vector>

namespace experiments {

// Frozen winner of the preceding recognition search (configuration 2153).
inline RecognitionParameters ccaRecognitionBase() {
    RecognitionParameters p;
    p.k = 6;
    p.lightMaxS = 125;
    p.solidMinPercent = 80;
    p.bluePurple = 125;
    return p;
}

// Defaults reproduce the existing CCA rules and feature extraction.
struct CCAParameters {
    double minComponentPercent = 2;
    int boundaryKernel = 5;
    bool cue = true;
    bool stripe = true;
    bool solid = true;
    int cueMinLight = 85;
    int cueLargestLight = 80;
    int cueMinBoundary = 75;
    int stripeLargestLight = 20;
    int stripeMinBoundary = 20;
    int stripeMinNonLight = 10;
    int solidMinNonLight = 70;
    int solidMaxBoundary = 12;
    int solidMaxComponents = 1;
    int solidLargestNonLight = 0; // Optional dominance constraint; 0 disables it.
};

struct CCASearchGrid {
    std::vector<double> componentPercent = {0.5, 1, 2, 3};
    std::vector<int> boundaryKernel = {3, 5, 7};
    std::vector<int> cueMinLight = {80, 85, 90, 95};
    std::vector<int> cueLargestLight = {70, 80, 90};
    std::vector<int> cueMinBoundary = {60, 75, 90};
    std::vector<int> stripeLargestLight = {5, 10, 15, 20, 25, 30};
    std::vector<int> stripeMinBoundary = {5, 10, 20, 30, 40, 50};
    std::vector<int> stripeMinNonLight = {2, 5, 10, 15, 20};
    std::vector<int> solidMinNonLight = {60, 70, 80, 90};
    std::vector<int> solidMaxBoundary = {5, 12, 20, 30};
    std::vector<int> solidMaxComponents = {1, 2};
    std::vector<int> solidLargestNonLight = {0, 50, 70, 85};
    int passes = 2;
};

} // namespace experiments
