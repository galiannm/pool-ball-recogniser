#pragma once

#include <opencv2/core.hpp>
#include <filesystem>
#include <fstream>

void saveImage(const std::filesystem::path& path, const cv::Mat& image);
void finishOutputFile(std::ofstream& file);
