#pragma once

#include <opencv2/core.hpp>
#include <string>
#include <vector>

struct ImagePanel {
    std::string label;
    cv::Mat image;
};

// Combine 8-bit grayscale, BGR, or BGRA images into a labelled BGR grid.
// Empty panels are skipped; images retain their original size and are centred.
// Returns an empty Mat when there are no non-empty panels.
cv::Mat combineImages(const std::vector<ImagePanel>& panels, int columns = 4);
