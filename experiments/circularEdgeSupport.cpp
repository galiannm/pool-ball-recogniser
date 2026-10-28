#include "circularEdgeSupport.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace experiments {
namespace {
void validate(const CircularEdgeOptions& p) {
    if (p.angularSamples < 4 || !std::isfinite(p.tolerance) || p.tolerance <= 0 ||
        !std::isfinite(p.minimumRadialAlignment) || p.minimumRadialAlignment < 0 ||
        p.minimumRadialAlignment > 1 || !std::isfinite(p.cannyHigh) || p.cannyHigh <= 0)
        throw std::invalid_argument("Invalid circular edge options");
}

int longestRun(const std::vector<bool>& supported) {
    int longest = 0, run = 0, n = static_cast<int>(supported.size());
    for (int i = 0; i < 2 * n; ++i) {
        run = supported[i % n] ? std::min(n, run + 1) : 0;
        longest = std::max(longest, run);
    }
    return longest;
}
} // namespace

CircularEdgeMaps prepareCircularEdgeMaps(const cv::Mat& image, const cv::Mat& tableMask,
                                        const CircularEdgeOptions& p) {
    validate(p);
    if (image.empty() || image.type() != CV_8UC3 || tableMask.type() != CV_8UC1 ||
        image.size() != tableMask.size())
        throw std::invalid_argument("Expected BGR image and same-size table mask");
    CircularEdgeMaps maps;
    maps.tableMask = tableMask.clone();
    cv::Mat grey, smooth;
    // Process the original image, avoiding artificial edges at the table-mask boundary.
    cv::cvtColor(image, grey, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(grey, smooth, {5, 5}, 1.5);
    cv::Canny(smooth, maps.edges, p.cannyHigh / 2, p.cannyHigh);
    cv::Sobel(smooth, maps.gradientX, CV_32F, 1, 0, 3);
    cv::Sobel(smooth, maps.gradientY, CV_32F, 0, 1, 3);
    return maps;
}

CircularEdgeScore measureCircularEdgeSupport(const CircularEdgeMaps& maps, const cv::Vec3f& circle,
                                             const CircularEdgeOptions& p) {
    validate(p);
    if (maps.edges.empty() || maps.edges.type() != CV_8UC1 || maps.tableMask.type() != CV_8UC1 ||
        maps.gradientX.type() != CV_32FC1 || maps.gradientY.type() != CV_32FC1 ||
        maps.edges.size() != maps.tableMask.size() || maps.edges.size() != maps.gradientX.size() ||
        maps.edges.size() != maps.gradientY.size() || !std::isfinite(circle[0]) ||
        !std::isfinite(circle[1]) || !std::isfinite(circle[2]) || circle[2] <= 0)
        throw std::invalid_argument("Invalid edge maps or circle");
    CircularEdgeScore score;
    std::vector<bool> supported(p.angularSamples, false);
    const int window = static_cast<int>(std::ceil(p.tolerance));
    int nearbyCount = 0, supportedCount = 0;
    for (int i = 0; i < p.angularSamples; ++i) {
        const double angle = 2 * CV_PI * i / p.angularSamples;
        const double dx = std::cos(angle), dy = std::sin(angle);
        CircularEdgeSample sample;
        sample.boundary = {float(circle[0] + circle[2] * dx), float(circle[1] + circle[2] * dy)};
        const int x = cvRound(sample.boundary.x), y = cvRound(sample.boundary.y);
        sample.valid = x >= 0 && y >= 0 && x < maps.edges.cols && y < maps.edges.rows &&
                       maps.tableMask.at<uchar>(y, x) != 0;
        if (sample.valid) {
            ++score.validSamples;
            double closest = p.tolerance + 1;
            for (int oy = -window; oy <= window; ++oy)
                for (int ox = -window; ox <= window; ++ox) {
                    const int ex = x + ox, ey = y + oy;
                    const double distance = std::hypot(ex - sample.boundary.x, ey - sample.boundary.y);
                    if (ex < 0 || ey < 0 || ex >= maps.edges.cols || ey >= maps.edges.rows ||
                        distance > p.tolerance || !maps.tableMask.at<uchar>(ey, ex) ||
                        !maps.edges.at<uchar>(ey, ex))
                        continue;
                    sample.nearbyEdge = true;
                    const double gx = maps.gradientX.at<float>(ey, ex), gy = maps.gradientY.at<float>(ey, ex);
                    const double magnitude = std::hypot(gx, gy);
                    const double alignment = magnitude > 1 ? std::abs(gx * dx + gy * dy) / magnitude : 0;
                    if (magnitude > 1 && alignment >= p.minimumRadialAlignment && distance < closest) {
                        sample.supported = true;
                        sample.edge = {float(ex), float(ey)};
                        sample.alignment = alignment;
                        closest = distance;
                    }
                }
            nearbyCount += sample.nearbyEdge;
            supportedCount += sample.supported;
        }
        supported[i] = sample.supported;
        score.samples.push_back(sample);
    }
    score.coverage = double(score.validSamples) / p.angularSamples;
    if (score.validSamples) {
        score.nearbyEdgeFraction = double(nearbyCount) / score.validSamples;
        score.supportFraction = double(supportedCount) / score.validSamples;
    }
    score.longestArcDegrees = longestRun(supported) * 360.0 / p.angularSamples;
    auto bridged = supported;
    for (int i = 0; i < p.angularSamples; ++i)
        if (score.samples[i].valid && supported[(i + p.angularSamples - 1) % p.angularSamples] &&
            supported[(i + 1) % p.angularSamples])
            bridged[i] = true;
    score.bridgedArcDegrees = longestRun(bridged) * 360.0 / p.angularSamples;
    return score;
}
} // namespace experiments
