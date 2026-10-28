#include "ballColourHistograms.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace experiments {
namespace {
void validate(const BallHistogramOptions& p) {
    if (p.hueBins < 2 || p.saturationBins < 2 || p.toneBins < 2 || p.erosionKernel < 1 ||
        p.erosionKernel % 2 == 0 || p.minimumBodyPixels < 1 || p.lightMinV < 0 || p.lightMinV > 255 ||
        p.lightMaxS < 0 || p.lightMaxS > 255 || p.neutralChroma < 0 || p.neutralChroma > 255)
        throw std::invalid_argument("Invalid ball histogram settings");
}

cv::Mat featureImage(const BallROI& ball, BallHistogramSpace space, const BallHistogramOptions& p) {
    cv::Mat converted, features;
    if (space == BallHistogramSpace::HLS) {
        cv::cvtColor(ball.image, converted, cv::COLOR_BGR2HLS);
        features.create(converted.size(), converted.type());
        const int mapping[] = {0, 0, 2, 1, 1, 2}; // Store H,S,L, so tone is always channel 2.
        cv::mixChannels(&converted, 1, &features, 1, mapping, 3);
    } else cv::cvtColor(ball.image, features, cv::COLOR_BGR2HSV);
    for (int y = 0; y < features.rows; ++y)
        for (int x = 0; x < features.cols; ++x) {
            const auto bgr = ball.image.at<cv::Vec3b>(y, x);
            const int chroma = int(std::max({bgr[0], bgr[1], bgr[2]})) - std::min({bgr[0], bgr[1], bgr[2]});
            if (chroma < p.neutralChroma) features.at<cv::Vec3b>(y, x)[0] = 0;
        }
    return features;
}

cv::Mat smoothHistogram(const cv::Mat& histogram, const BallHistogramOptions& p, bool tone) {
    cv::Mat current = histogram.clone();
    const int hCount = p.hueBins, sCount = p.saturationBins, tCount = tone ? p.toneBins : 1;
    auto at = [=](int h, int s, int t) {return (h * sCount + s) * tCount + t;};
    // Separable [1,2,1]/4 smoothing. Hue wraps; saturation and tone clamp at endpoints.
    for (int axis = 0; axis < (tone ? 3 : 2); ++axis) {
        cv::Mat next = cv::Mat::zeros(current.dims, current.size, CV_32F);
        const float* source = current.ptr<float>();
        float* target = next.ptr<float>();
        for (int h = 0; h < hCount; ++h)
            for (int s = 0; s < sCount; ++s)
                for (int t = 0; t < tCount; ++t)
                    for (int offset = -1; offset <= 1; ++offset) {
                        const int hh = axis == 0 ? (h + offset + hCount) % hCount : h;
                        const int ss = axis == 1 ? std::clamp(s + offset, 0, sCount - 1) : s;
                        const int tt = axis == 2 ? std::clamp(t + offset, 0, tCount - 1) : t;
                        target[at(h, s, t)] += source[at(hh, ss, tt)] * (offset == 0 ? 0.5f : 0.25f);
                    }
        current = next;
    }
    cv::normalize(current, current, 1, 0, cv::NORM_L1);
    return current;
}

cv::Mat makeHistogram(const cv::Mat& features, const cv::Mat& mask, BallHistogramSpace space,
                      const BallHistogramOptions& p) {
    const int channels[] = {0, 1, 2}, bins[] = {p.hueBins, p.saturationBins, p.toneBins};
    const float hue[] = {0, 180}, other[] = {0, 256};
    const float* ranges[] = {hue, other, other};
    cv::Mat histogram;
    cv::calcHist(&features, 1, channels, mask, histogram, space == BallHistogramSpace::HS ? 2 : 3, bins, ranges);
    return smoothHistogram(histogram, p, space != BallHistogramSpace::HS);
}
} // namespace

const char* histogramSpaceName(BallHistogramSpace space) {
    switch (space) {
        case BallHistogramSpace::HS: return "hs_histogram";
        case BallHistogramSpace::HSV: return "hsv_histogram";
        case BallHistogramSpace::HLS: return "hls_histogram";
    }
    throw std::invalid_argument("Unknown histogram space");
}

const char* bodyColourName(int colour) {
    static const char* names[] = {"unknown", "yellow", "blue", "red", "purple", "orange", "green", "maroon", "black"};
    return names[colour >= 1 && colour <= 8 ? colour : 0];
}

cv::Mat histogramBodyMask(const BallROI& ball, const BallHistogramOptions& p) {
    validate(p);
    if (ball.image.empty() || ball.image.type() != CV_8UC3 || ball.mask.type() != CV_8UC1 ||
        ball.image.size() != ball.mask.size()) throw std::invalid_argument("Invalid ball ROI");
    cv::Mat mask, hsv, light;
    cv::erode(ball.mask, mask, cv::getStructuringElement(cv::MORPH_ELLIPSE, {p.erosionKernel, p.erosionKernel}),
              {-1, -1}, 1, cv::BORDER_CONSTANT, 0);
    cv::cvtColor(ball.image, hsv, cv::COLOR_BGR2HSV);
    cv::inRange(hsv, cv::Scalar(0, 0, p.lightMinV), cv::Scalar(179, p.lightMaxS, 255), light);
    mask.setTo(0, light);
    return mask;
}

BallColourModel trainBallColourModel(const std::vector<LabelledColourSample>& samples,
    BallHistogramSpace space, const BallHistogramOptions& options) {
    validate(options);
    BallColourModel model{space, options, {}, {}};
    for (const auto& sample : samples) {
        if (sample.number < 0 || sample.number > 15) throw std::invalid_argument("Invalid training identity");
        if (sample.number == 0) continue;
        const auto mask = histogramBodyMask(sample.ball, options);
        if (cv::countNonZero(mask) < options.minimumBodyPixels)
            throw std::runtime_error("Insufficient body pixels for training ball " + std::to_string(sample.number));
        const int colour = (sample.number > 8 ? sample.number - 8 : sample.number) - 1;
        const auto histogram = makeHistogram(featureImage(sample.ball, space, options), mask, space, options);
        if (model.histograms[colour].empty()) model.histograms[colour] = histogram.clone();
        else model.histograms[colour] += histogram;
        ++model.examples[colour];
    }
    for (size_t i = 0; i < model.histograms.size(); ++i) {
        if (!model.examples[i]) throw std::runtime_error("No training example for " + std::string(bodyColourName(i + 1)));
        cv::normalize(model.histograms[i], model.histograms[i], 1, 0, cv::NORM_L1);
    }
    return model;
}

BallColourScores scoreBallColours(const BallROI& ball, const BallColourModel& model) {
    BallColourScores result;
    result.bodyMask = histogramBodyMask(ball, model.options);
    result.pixels = cv::countNonZero(result.bodyMask);
    if (result.pixels < model.options.minimumBodyPixels) return result;
    const auto features = featureImage(ball, model.space, model.options);
    const auto query = makeHistogram(features, result.bodyMask, model.space, model.options);
    cv::Mat floatFeatures;
    features.convertTo(floatFeatures, CV_32F); // Preserve sub-unit histogram masses in backprojection.
    const int channels[] = {0, 1, 2};
    const float hue[] = {0, 180}, other[] = {0, 256};
    const float* ranges[] = {hue, other, other};
    for (size_t i = 0; i < model.histograms.size(); ++i) {
        const auto& histogram = model.histograms[i];
        if (histogram.empty() || histogram.type() != CV_32F || histogram.total() != query.total())
            throw std::invalid_argument("Invalid trained colour histogram");
        // Flat views make comparison independent of OpenCV's handling of 3D Mats.
        const cv::Mat a(1, int(query.total()), CV_32F, const_cast<float*>(query.ptr<float>()));
        const cv::Mat b(1, int(histogram.total()), CV_32F, const_cast<float*>(histogram.ptr<float>()));
        result.similarity[i] = std::clamp(1 - cv::compareHist(a, b, cv::HISTCMP_BHATTACHARYYA), 0.0, 1.0);
        cv::calcBackProject(&floatFeatures, 1, channels, histogram, result.backprojections[i], ranges, 1);
        result.backprojections[i].setTo(0, ball.mask == 0);
    }
    const auto best = std::max_element(result.similarity.begin(), result.similarity.end());
    result.bestColour = int(best - result.similarity.begin()) + 1;
    double second = 0;
    for (size_t i = 0; i < result.similarity.size(); ++i)
        if (int(i + 1) != result.bestColour) second = std::max(second, result.similarity[i]);
    result.margin = *best - second;
    return result;
}

int histogramBallNumber(const std::string& predictedType, int colour) {
    if (predictedType == "White") return 0;
    if (colour < 1 || colour > 8) return -1;
    if (predictedType == "Solid") return colour;
    if (predictedType == "Striped" && colour <= 7) return colour + 8;
    return -1;
}
} // namespace experiments
