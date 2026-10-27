#include "combineImages.h"

#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <stdexcept>

cv::Mat combineImages(const std::vector<ImagePanel>& panels, int columns)
{
    if (columns <= 0)
        throw std::invalid_argument("Image grid columns must be positive");

    constexpr int padding = 8;
    constexpr int labelHeight = 24;
    constexpr double fontScale = 0.4;
    int width = 0;
    int height = 0;
    int count = 0;

    for (const auto& panel : panels) {
        if (panel.image.empty())
            continue;
        if (panel.image.depth() != CV_8U ||
            (panel.image.channels() != 1 && panel.image.channels() != 3 && panel.image.channels() != 4))
            throw std::invalid_argument("Image panels must be 8-bit grayscale, BGR, or BGRA");

        int baseline = 0;
        const auto labelSize = cv::getTextSize(panel.label, cv::FONT_HERSHEY_SIMPLEX,
                                              fontScale, 1, &baseline);
        width = std::max({width, panel.image.cols, labelSize.width});
        height = std::max(height, panel.image.rows);
        ++count;
    }

    if (count == 0)
        return {};

    columns = std::min(columns, count);
    const int rows = (count + columns - 1) / columns;
    const int cellWidth = width + 2 * padding;
    const int cellHeight = height + labelHeight + 2 * padding;
    cv::Mat combined(rows * cellHeight, columns * cellWidth, CV_8UC3, cv::Scalar(32, 32, 32));

    int index = 0;
    for (const auto& panel : panels) {
        if (panel.image.empty())
            continue;

        cv::Mat colour;
        if (panel.image.channels() == 1)
            cv::cvtColor(panel.image, colour, cv::COLOR_GRAY2BGR);
        else if (panel.image.channels() == 4)
            cv::cvtColor(panel.image, colour, cv::COLOR_BGRA2BGR);
        else
            colour = panel.image;

        const int x = (index % columns) * cellWidth + padding;
        const int y = (index / columns) * cellHeight + padding;
        cv::putText(combined, panel.label, cv::Point(x, y + 15),
                    cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        colour.copyTo(combined(cv::Rect(x + (width - colour.cols) / 2,
                                       y + labelHeight + (height - colour.rows) / 2,
                                       colour.cols, colour.rows)));
        ++index;
    }
    return combined;
}
