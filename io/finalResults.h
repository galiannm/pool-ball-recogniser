#pragma once

#include <opencv2/core.hpp>
#include <filesystem>
#include <string>
#include <vector>

// Shared application/experiment presentation. Uses predictions only; no annotations.
void saveFinalResults(const cv::Mat& image, const std::filesystem::path& directory,
                      const std::vector<cv::Vec3f>& circles,
                      const std::vector<cv::Point2f>& tableCentres,
                      const std::vector<std::string>& types,
                      const std::vector<int>& numbers);
