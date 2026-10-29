#pragma once
#include "ballColourHistograms.h"
#include "recognition/ballTypeClassifier.h"
#include <filesystem>
namespace experiments {
void runIdentityHeuristics(const std::filesystem::path& directory);
void runBestEffortIdentityHeuristics(const std::filesystem::path& directory);
void runWhiteExpansionIdentityHeuristics(const std::filesystem::path& directory);
void runColourScoreDiagnostics(const std::filesystem::path& directory);
void runRowCueSizeExperiment(const std::filesystem::path& directory);
// Shared inference backend for independent crop-refinement experiments. No labels.
BallColourModel loadFrozenIdentityColourModel(const std::filesystem::path& path);
std::vector<int> recogniseUniqueBallROIs(const std::vector<BallROI>& rois, const BallColourModel& model);
// Precomputed type/colour evidence permits a K-means-free experimental frontend.
std::vector<int> recogniseUniqueBallEvidence(const std::vector<BallTypeResult>& types,
    const std::vector<BallColourScores>& colours, int reservedCue = -1);
}
