#pragma once

#include "kMeansSegmentation.h"
#include "connectedComponentAnalysis.h"

#include <string>

struct BallTypeResult {
    std::string type = "Unknown";

    double lightPercent = 0;
    double colouredPercent = 0;
    double darkPercent = 0;
};

// Existing K-means-only baseline
BallTypeResult classifyBallType(
    const KMeansResult& segmentation
);

// New K-means + CCA classifier
BallTypeResult classifyBallTypeWithCCA(
    const KMeansResult& segmentation,
    const CCAFeatures& cca
);

// Experimental HSV rules applied after type classification.
// Returns 0 for the cue ball, 1-15 for numbered balls, or -1 if unknown.
int classifyBallNumber(
    const KMeansResult& segmentation,
    const BallTypeResult& predictedType
);
