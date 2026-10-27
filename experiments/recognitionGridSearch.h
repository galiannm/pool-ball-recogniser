#pragma once

#include "recognitionSearchConfig.h"
#include "recognition/ballTypeClassifier.h"
#include <filesystem>

namespace experiments {

BallTypeResult classifyExperimentalType(const KMeansResult& segmentation,
                                       const RecognitionParameters& parameters);
int classifyExperimentalNumber(const KMeansResult& segmentation,
                               const std::string& type,
                               const RecognitionParameters& parameters);

void runRecognitionGridSearch(const std::filesystem::path& outputDirectory);

} // namespace experiments
