#pragma once

#include "ccaSearchConfig.h"
#include "recognition/ballTypeClassifier.h"
#include "recognition/connectedComponentAnalysis.h"
#include <filesystem>

namespace experiments {

CCAFeatures measureExperimentalCCA(const BallROI& ball, const KMeansResult& segmentation,
                                   const RecognitionParameters& recognition,
                                   const CCAParameters& parameters);

BallTypeResult classifyExperimentalCCA(const BallTypeResult& baseline, const CCAFeatures& cca,
                                      const CCAParameters& parameters,
                                      std::string* decision = nullptr);

void runCCAGridSearch(const std::filesystem::path& outputDirectory);

} // namespace experiments
