#pragma once

#include <filesystem>

namespace experiments {
void runHistogramCCAExperiment(const std::filesystem::path& directory);
void runOriginalK2HistogramCCAExperiment(const std::filesystem::path& directory);
}
