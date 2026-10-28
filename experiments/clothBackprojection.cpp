#include "clothBackprojection.h"
#include "config.h"
#include "geometry/perspectiveTransform.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace experiments {
namespace {

void validateOptions(const ClothBackprojectionOptions& p) {
    if (p.hBins < 1 || p.sBins < 1 || p.samplingInsetKernel < 1 || p.samplingInsetKernel % 2 != 1 ||
        p.exclusionMargin < 0 || p.minimumSamples < 1 ||
        !std::isfinite(p.similarityThreshold) || p.similarityThreshold < 0 || p.similarityThreshold > 1 ||
        !std::isfinite(p.maximumClothFraction) || p.maximumClothFraction < 0 || p.maximumClothFraction > 1 ||
        !std::isfinite(p.minimumCircleCoverage) || p.minimumCircleCoverage < 0 || p.minimumCircleCoverage > 1)
        throw std::invalid_argument("Invalid cloth backprojection settings");
}

void validateCircle(const cv::Vec3f& c) {
    if (!std::isfinite(c[0]) || !std::isfinite(c[1]) || !std::isfinite(c[2]) || c[2] <= 0)
        throw std::invalid_argument("Invalid circle for cloth verification");
}

} // namespace

cv::Mat makeClothPocketMask(cv::Size sourceSize) {
    cv::Mat topDown = cv::Mat::zeros(config::tableHeight, config::tableWidth, CV_8UC1);
    for (const auto& point : std::vector<cv::Point>{
        {0, 0}, {config::tableWidth / 2, 0}, {config::tableWidth - 1, 0},
        {0, config::tableHeight - 1}, {config::tableWidth / 2, config::tableHeight - 1},
        {config::tableWidth - 1, config::tableHeight - 1}})
        cv::circle(topDown, point, cvRound(config::pocketRadius), cv::Scalar(255), cv::FILLED);
    cv::Mat source;
    cv::warpPerspective(topDown, source, getTableTransformation().inv(), sourceSize,
                        cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0));
    return source;
}

ClothBackprojectionModel estimateClothBackprojection(const cv::Mat& image,
    const cv::Mat& tableMask, const cv::Mat& pocketMask,
    const std::vector<cv::Vec3f>& rawHoughCandidates, const ClothBackprojectionOptions& p) {
    validateOptions(p);
    if (image.empty() || image.type() != CV_8UC3 || tableMask.type() != CV_8UC1 ||
        tableMask.size() != image.size() ||
        (!pocketMask.empty() && (pocketMask.type() != CV_8UC1 || pocketMask.size() != image.size())))
        throw std::invalid_argument("Cloth estimation needs BGR image and matching single-channel masks");
    ClothBackprojectionModel result;
    cv::erode(tableMask, result.samplingMask,
               cv::getStructuringElement(cv::MORPH_ELLIPSE, {p.samplingInsetKernel, p.samplingInsetKernel}),
               {-1, -1}, 1, cv::BORDER_CONSTANT, cv::Scalar(0));
    if (!pocketMask.empty())
        result.samplingMask.setTo(0, pocketMask);
    for (const auto& c : rawHoughCandidates) {
        validateCircle(c);
        cv::circle(result.samplingMask, {cvRound(c[0]), cvRound(c[1])},
                   std::max(1, cvRound(c[2])) + p.exclusionMargin, cv::Scalar(0), cv::FILLED);
    }
    result.samples = cv::countNonZero(result.samplingMask);
    if (result.samples < p.minimumSamples) {
        result.status = "insufficient_cloth_samples";
        return result;
    }
    cv::Mat hsv;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
    const int channels[] = {0, 1}, sizes[] = {p.hBins, p.sBins};
    const float hRange[] = {0, 180}, sRange[] = {0, 256};
    const float* ranges[] = {hRange, sRange};
    cv::calcHist(&hsv, 1, channels, result.samplingMask, result.histogram, 2, sizes, ranges, true, false);
    double minimum = 0, maximum = 0;
    cv::minMaxLoc(result.histogram, &minimum, &maximum);
    if (maximum <= minimum) {
        result.status = "flat_histogram";
        return result;
    }
    // calcBackProject inherits the input image depth. Scaling an 8-bit image's
    // histogram to [0,1] first would quantise away most of the useful scores.
    cv::normalize(result.histogram, result.histogram, 0, 255, cv::NORM_MINMAX);
    cv::calcBackProject(&hsv, 1, channels, result.histogram, result.similarity8U, ranges, 1, true);
    result.similarity8U.convertTo(result.similarity, CV_32F, 1.0 / 255.0);
    result.valid = true;
    result.status = "ok";
    return result;
}

ClothVerificationResult verifyClothCandidates(const ClothBackprojectionModel& model,
    const cv::Mat& tableMask, const std::vector<cv::Vec3f>& candidates, const ClothBackprojectionOptions& p) {
    validateOptions(p);
    if (tableMask.empty() || tableMask.type() != CV_8UC1 ||
        (model.valid && (model.similarity.type() != CV_32FC1 || model.similarity.size() != tableMask.size())))
        throw std::invalid_argument("Invalid cloth model/scoring mask");
    ClothVerificationResult result;
    for (size_t i = 0; i < candidates.size(); ++i) {
        const auto& c = candidates[i];
        validateCircle(c);
        ClothCandidateScore score;
        const int radius = std::max(1, cvRound(c[2]));
        const cv::Point centre(cvRound(c[0]), cvRound(c[1]));
        const cv::Rect full(centre.x - radius, centre.y - radius, 2 * radius + 1, 2 * radius + 1);
        const cv::Rect crop = full & cv::Rect(0, 0, tableMask.cols, tableMask.rows);
        cv::Mat fullCircle = cv::Mat::zeros(full.size(), CV_8UC1);
        cv::circle(fullCircle, {radius, radius}, radius, cv::Scalar(255), cv::FILLED);
        cv::Mat mask;
        if (!crop.empty()) {
            mask = fullCircle(cv::Rect(crop.tl() - full.tl(), crop.size())).clone();
            cv::bitwise_and(mask, tableMask(crop), mask);
            score.validPixels = cv::countNonZero(mask);
        }
        score.coverage = double(score.validPixels) / cv::countNonZero(fullCircle);
        if (!model.valid)
            score.reason = "skipped_" + model.status;
        else if (!score.validPixels || score.coverage < p.minimumCircleCoverage)
            score.reason = "insufficient_circle_coverage";
        else {
            score.meanSimilarity = cv::mean(model.similarity(crop), mask)[0];
            cv::Mat high;
            cv::compare(model.similarity(crop), p.similarityThreshold, high, cv::CMP_GE);
            cv::bitwise_and(high, mask, high);
            score.clothFraction = double(cv::countNonZero(high)) / score.validPixels;
            if (!p.enabled)
                score.reason = "verification_disabled";
            else if (score.clothFraction > p.maximumClothFraction) {
                score.retained = false;
                score.reason = "cloth_fraction";
            } else
                score.reason = "retained";
        }
        if (score.retained) {
            result.retainedIndices.push_back(static_cast<int>(i));
            result.retained.push_back(c);
        }
        result.scores.push_back(score);
    }
    return result;
}

} // namespace experiments
