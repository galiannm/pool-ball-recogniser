#include "identityHeuristics.h"
#include "ballColourHistograms.h"
#include "config.h"
#include "detection/ballDetection.h"
#include "evaluation/evaluation.h"
#include "geometry/perspectiveTransform.h"
#include "io/finalResults.h"
#include "io/output.h"
#include "recognition/ballTypeClassifier.h"
#include "visualisation/combineImages.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace experiments {
namespace {
const std::array<std::string, 4> methods = {"A_baseline", "B_scores", "C_unique", "D_unique_geometry"};
const std::array<std::string, 4> names = {"A: K2 + HSV", "B: independent scores", "C: unique identities", "D: uniqueness + size"};
// Predeclared heuristics. No search, calibration or selection on Table1–6.
constexpr double typeWidth = 0.10, unknownScore = 0.35, maximumAlternativeDrop = 0.15;
constexpr double geometryFloor = 0.75, radiusToleranceRatio = 1.5, ballRadiusMm = 21.5;

std::ofstream output(const std::filesystem::path& path) {
    std::ofstream out(path);
    out.exceptions(std::ios::failbit | std::ios::badbit);
    out << std::fixed << std::setprecision(6);
    return out;
}
std::string label(int n) {return n < 0 ? "Unknown" : n == 0 ? "Cue" : std::to_string(n);}
std::string typeFor(int n) {return n < 0 ? "Unknown" : n == 0 ? "White" : n <= 8 ? "Solid" : "Striped";}
double ratio(double a, double b) {return b ? a / b : 0;}

BallColourModel loadModel(const std::filesystem::path& path) {
    if (!std::filesystem::exists(path)) throw std::runtime_error("Run --colour-histograms first; missing " + path.string());
    cv::FileStorage input(path.string(), cv::FileStorage::READ);
    if (!input.isOpened() || std::string(input["trainingImage"]) != "AllBalls.jpeg")
        throw std::runtime_error("Expected an AllBalls-trained histogram model");
    BallColourModel model{BallHistogramSpace::HSV, BallHistogramOptions{}, {}, {}};
    bool found = false;
    std::array<bool, 8> seen{};
    for (const auto& node : input["models"]) {
        if (std::string(node["space"]) != "hsv_histogram") continue;
        if (found) throw std::runtime_error("Repeated HSV model");
        found = true;
        for (const auto& entry : node["histograms"]) {
            const int c = int(entry["colour"]);
            if (c < 1 || c > 8 || seen[c - 1]) throw std::runtime_error("Invalid/repeated model colour");
            seen[c - 1] = true;
            entry["histogram"] >> model.histograms[c - 1];
            model.examples[c - 1] = int(entry["examples"]);
            const auto& h = model.histograms[c - 1];
            if (h.type() != CV_32F || h.dims != 3 || h.size[0] != 18 || h.size[1] != 8 || h.size[2] != 8 ||
                model.examples[c - 1] != (c == 8 ? 1 : 2) || !cv::checkRange(h) || std::abs(cv::sum(h)[0] - 1) > 1e-5)
                throw std::runtime_error("Unexpected HSV histogram parameters");
            for (size_t i = 0; i < h.total(); ++i)
                if (h.ptr<float>()[i] < 0) throw std::runtime_error("Negative histogram mass");
        }
    }
    if (!found || std::find(seen.begin(), seen.end(), false) != seen.end()) throw std::runtime_error("Incomplete HSV model");
    return model;
}

struct Geometry {
    bool valid = false;
    double rawRadius = 0, minorRadius = 0, angle = 0;
    double expectedRadius = 0, radiusRatio = 0, compatibility = 1, factor = 1;
};

Geometry measureGeometry(const cv::Vec3f& circle, cv::Point2f tableCentre,
                         const cv::Mat& inverse, double scale = 1) {
    Geometry g;
    const double x = tableCentre.x, y = tableCentre.y;
    const double w = inverse.at<double>(2, 0) * x + inverse.at<double>(2, 1) * y + inverse.at<double>(2, 2);
    if (!std::isfinite(w) || std::abs(w) < 1e-9 || circle[2] <= 0) return g;
    const double u = inverse.at<double>(0, 0) * x + inverse.at<double>(0, 1) * y + inverse.at<double>(0, 2);
    const double v = inverse.at<double>(1, 0) * x + inverse.at<double>(1, 1) * y + inverse.at<double>(1, 2);
    cv::Mat jacobian(2, 2, CV_64F);
    // Source pixels per physical table mm; preserve the existing endpoint convention.
    for (int row = 0; row < 2; ++row)
        for (int col = 0; col < 2; ++col) {
            const double numerator = row == 0 ? u : v;
            const double pixelsPerMm = col == 0 ? (config::tableWidth - 1.0) / 1434.0 : (config::tableHeight - 1.0) / 683.0;
            jacobian.at<double>(row, col) = (inverse.at<double>(row, col) * w - numerator * inverse.at<double>(2, col))
                                          / (w * w) * pixelsPerMm;
        }
    cv::Mat singular, directions, unused;
    cv::SVD::compute(jacobian, singular, directions, unused);
    g.rawRadius = ballRadiusMm * singular.at<double>(0);
    g.minorRadius = ballRadiusMm * singular.at<double>(1);
    g.angle = std::atan2(directions.at<double>(1, 0), directions.at<double>(0, 0)) * 180 / CV_PI;
    if (!std::isfinite(g.rawRadius) || g.rawRadius <= 0 || !std::isfinite(scale) || scale <= 0) return Geometry{};
    // Major plane-projection axis approximates the sphere's apparent image radius.
    g.expectedRadius = scale * g.rawRadius;
    g.radiusRatio = circle[2] / g.expectedRadius;
    const double residual = std::log(g.radiusRatio) / std::log(radiusToleranceRatio);
    g.compatibility = std::exp(-0.5 * residual * residual);
    g.factor = geometryFloor + (1 - geometryFloor) * g.compatibility;
    g.valid = true;
    return g;
}

double geometryCalibration(const std::filesystem::path& directory, const cv::Mat& inverse) {
    const cv::Mat image = cv::imread(config::mediaDirectory + "AllBalls.jpeg");
    if (image.empty()) throw std::runtime_error("Could not load AllBalls for geometry calibration");
    const auto circles = runDetection(image, HoughParams{}).filteredCircles;
    const auto centres = transformCentres(circles);
    auto csv = output(directory / "training_geometry.csv");
    csv << "candidate,x,y,detectedRadius,geometricRadius,radiusRatio\n";
    std::vector<double> ratios;
    for (size_t i = 0; i < circles.size(); ++i) {
        const auto g = measureGeometry(circles[i], centres[i], inverse);
        if (!g.valid) continue;
        ratios.push_back(g.radiusRatio);
        csv << i + 1 << ',' << circles[i][0] << ',' << circles[i][1] << ',' << circles[i][2] << ',' << g.rawRadius << ',' << g.radiusRatio << '\n';
    }
    if (ratios.empty()) throw std::runtime_error("No valid AllBalls radius calibration samples");
    std::sort(ratios.begin(), ratios.end());
    const size_t middle = ratios.size() / 2;
    return ratios.size() % 2 ? ratios[middle] : (ratios[middle - 1] + ratios[middle]) / 2;
}

// Rectangular Hungarian assignment: maximise total score, each column used at most once.
std::vector<int> maximumAssignment(const std::vector<std::vector<double>>& weights) {
    const int n = int(weights.size());
    if (!n) return {};
    const int m = int(weights[0].size());
    if (m < n) throw std::invalid_argument("Assignment requires at least as many columns as rows");
    for (const auto& row : weights) {
        if (int(row.size()) != m) throw std::invalid_argument("Ragged assignment matrix");
        for (double w : row) if (!std::isfinite(w)) throw std::invalid_argument("Non-finite assignment weight");
    }
    std::vector<double> u(n + 1), v(m + 1);
    std::vector<int> p(m + 1), way(m + 1);
    for (int i = 1; i <= n; ++i) {
        p[0] = i;
        int j0 = 0;
        std::vector<double> minimum(m + 1, std::numeric_limits<double>::infinity());
        std::vector<bool> used(m + 1, false);
        do {
            used[j0] = true;
            const int i0 = p[j0];
            int j1 = 0;
            double delta = std::numeric_limits<double>::infinity();
            for (int j = 1; j <= m; ++j) if (!used[j]) {
                const double reduced = -weights[i0 - 1][j - 1] - u[i0] - v[j];
                if (reduced < minimum[j]) {minimum[j] = reduced; way[j] = j0;}
                if (minimum[j] < delta) {delta = minimum[j]; j1 = j;}
            }
            for (int j = 0; j <= m; ++j) {
                if (used[j]) {u[p[j]] += delta; v[j] -= delta;}
                else minimum[j] -= delta;
            }
            j0 = j1;
        } while (p[j0] != 0);
        do {const int j1 = way[j0]; p[j0] = p[j1]; j0 = j1;} while (j0);
    }
    std::vector<int> assigned(n, -1);
    for (int j = 1; j <= m; ++j) if (p[j]) assigned[p[j] - 1] = j - 1;
    return assigned;
}

struct Candidate {
    cv::Vec3f circle;
    cv::Point2f tableCentre;
    BallROI roi;
    BallTypeResult baselineType;
    BallColourScores colours;
    Geometry geometry;
    std::array<double, 3> typeSupport{}; // Cue, solid, stripe; not calibrated probabilities.
    std::array<double, 16> identityScore{};
    std::array<bool, 16> eligible{};
    int localBest = -1;
    double bestScore = 0;
};

bool hasUsableEvidence(const Candidate& c) {
    return c.colours.bestColour >= 1 || c.baselineType.type == "White";
}

void identityEvidence(Candidate& c, bool bestEffort = false) {
    const double light = std::clamp(c.baselineType.lightPercent / 100, 0.0, 1.0);
    const auto compatibility = [](double distance) {return std::exp(-0.5 * std::pow(distance / typeWidth, 2));};
    // Distances to the original K2 light intervals: cue >=.85, solid <=.30, stripe [.30,.85].
    c.typeSupport = {compatibility(std::max(0.0, 0.85 - light)), compatibility(std::max(0.0, light - 0.30)),
                     compatibility(std::max({0.0, 0.30 - light, light - 0.85}))};
    c.identityScore[0] = light * c.typeSupport[0];
    for (int n = 1; n <= 15; ++n) {
        const int colour = n <= 8 ? n : n - 8;
        c.identityScore[n] = c.colours.bestColour < 0 ? 0 : c.colours.similarity[colour - 1] * c.typeSupport[n <= 8 ? 1 : 2];
    }
    const auto best = std::max_element(c.identityScore.begin(), c.identityScore.end());
    c.localBest = int(best - c.identityScore.begin());
    c.bestScore = *best;
    for (int n = 0; n <= 15; ++n)
        c.eligible[n] = bestEffort ? hasUsableEvidence(c) && c.identityScore[n] > 1e-9
            : c.identityScore[n] > unknownScore && c.identityScore[n] >= c.bestScore - maximumAlternativeDrop;
}

std::vector<int> assignUnique(const std::vector<Candidate>& candidates, bool geometry, bool bestEffort = false) {
    const double abstention = bestEffort ? 0 : unknownScore;
    std::vector<std::vector<double>> weights(candidates.size(), std::vector<double>(16 + candidates.size(), abstention + 1e-9));
    for (size_t i = 0; i < candidates.size(); ++i)
        for (int n = 0; n <= 15; ++n)
            weights[i][n] = candidates[i].eligible[n]
                ? candidates[i].identityScore[n] * (geometry ? candidates[i].geometry.factor : 1) : -1e6;
    auto result = maximumAssignment(weights);
    for (auto& n : result) if (n >= 16) n = -1; // Separate dummy columns allow any number of Unknowns.
    return result;
}

bool lowSupport(const Candidate& c, int number, bool geometry) {
    if (number < 0) return false;
    const double score = c.identityScore[number] * (geometry ? c.geometry.factor : 1);
    return score <= unknownScore || c.identityScore[number] < c.bestScore - maximumAlternativeDrop;
}

struct Counts {
    int gt = 0, tp = 0, fp = 0, fn = 0, type = 0, number = 0, unknown = 0, duplicates = 0;
    double errorSum = 0;
    void add(const Counts& c) {
        gt += c.gt; tp += c.tp; fp += c.fp; fn += c.fn; type += c.type; number += c.number;
        unknown += c.unknown; duplicates += c.duplicates; errorSum += c.errorSum;
    }
};
void writeCounts(std::ostream& out, const Counts& c) {
    out << c.gt << ',' << c.tp + c.fp << ',' << c.tp << ',' << c.fp << ',' << c.fn << ','
        << ratio(c.tp, c.tp + c.fp) << ',' << ratio(c.tp, c.gt) << ',' << ratio(2 * c.tp, 2 * c.tp + c.fp + c.fn) << ','
        << ratio(c.errorSum, c.tp) << ',' << c.type << ',' << c.number << ',' << ratio(c.type, c.tp) << ','
        << ratio(c.number, c.tp) << ',' << ratio(c.number, c.gt) << ',' << ratio(c.number, c.tp + c.fp) << ',' << c.unknown << ',' << c.duplicates;
}
std::string change(int before, int after, int actual) {
    if (before == after) return "unchanged";
    if (actual < 0) return "unmatched";
    if (after == actual) return "corrected";
    if (before == actual) return "regressed";
    return "changed_still_wrong";
}
cv::Mat enlarged(const cv::Mat& image) {
    cv::Mat result; cv::resize(image, result, {240, 240}, 0, 0, cv::INTER_NEAREST); return result;
}
void drawGeometry(cv::Mat& canvas, const Candidate& c, cv::Point offset = {}) {
    const cv::Point centre(cvRound(c.circle[0]) - offset.x, cvRound(c.circle[1]) - offset.y);
    cv::circle(canvas, centre, cvRound(c.circle[2]), {30, 100, 255}, 1, cv::LINE_AA);
    if (c.geometry.valid) {
        cv::circle(canvas, centre, cvRound(c.geometry.expectedRadius), {255, 255, 0}, 1, cv::LINE_AA);
        const double scale = c.geometry.expectedRadius / c.geometry.rawRadius;
        cv::ellipse(canvas, cv::RotatedRect(cv::Point2f(centre),
            cv::Size2f(float(2 * c.geometry.expectedRadius), float(2 * c.geometry.minorRadius * scale)), float(c.geometry.angle)),
            {180, 180, 180}, 1, cv::LINE_AA);
    }
}

struct ExpansionCloth {
    int hue = -1, samples = 0;
    bool valid = false;
};

ExpansionCloth expansionCloth(const cv::Mat& image, const std::vector<cv::Vec3f>& rawCircles) {
    cv::Mat mask = cv::Mat::zeros(image.size(), CV_8UC1), hsv, visible;
    cv::fillConvexPoly(mask, getTableCorners(), cv::Scalar(255));
    cv::erode(mask, mask, cv::getStructuringElement(cv::MORPH_ELLIPSE, {21, 21}));
    for (const auto& c : rawCircles)
        cv::circle(mask, {cvRound(c[0]), cvRound(c[1])}, cvCeil(c[2]) + 12, cv::Scalar(0), cv::FILLED);
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
    cv::inRange(hsv, cv::Scalar(0, 50, 40), cv::Scalar(179, 255, 255), visible);
    cv::bitwise_and(mask, visible, mask);
    std::array<int, 180> histogram{};
    ExpansionCloth result;
    for (int y = 0; y < image.rows; ++y)
        for (int x = 0; x < image.cols; ++x) if (mask.at<uchar>(y, x)) {
            ++histogram[hsv.at<cv::Vec3b>(y, x)[0]];
            ++result.samples;
        }
    if (result.samples < 1000) return result;
    int best = -1;
    for (int h = 0; h < 180; ++h) {
        int sum = 0;
        for (int d = -3; d <= 3; ++d) sum += histogram[(h + d + 180) % 180];
        if (sum > best) {best = sum; result.hue = h;}
    }
    result.valid = true;
    return result;
}

struct ExpansionAttempt {
    int padding = 0, addedPixels = 0, minimumAdded = 0;
    bool supported = false;
    Candidate candidate;
};

ExpansionAttempt expandWhiteCandidate(const cv::Mat& image, const std::vector<cv::Vec3f>& circles,
    size_t index, const Candidate& original, const BallColourModel& model, const ExpansionCloth& cloth, int padding) {
    ExpansionAttempt attempt;
    attempt.padding = padding;
    attempt.candidate = original;
    if (!padding || !cloth.valid) return attempt;
    cv::Vec3f expandedCircle = original.circle;
    expandedCircle[2] += padding;
    auto roi = extractBall(image, expandedCircle);
    const int radius = std::max(1, cvRound(expandedCircle[2]));
    const cv::Point origin(std::max(0, cvRound(original.circle[0]) - radius),
                           std::max(0, cvRound(original.circle[1]) - radius));
    const cv::Point centre(cvRound(original.circle[0]) - origin.x, cvRound(original.circle[1]) - origin.y);
    cv::Mat inner = cv::Mat::zeros(roi.mask.size(), CV_8UC1), hsv, coloured, light, boundary;
    cv::circle(inner, centre, std::max(1, cvRound(original.circle[2])), cv::Scalar(255), cv::FILLED);
    cv::cvtColor(roi.image, hsv, cv::COLOR_BGR2HSV);
    cv::inRange(hsv, cv::Scalar(0, 80, 40), cv::Scalar(179, 255, 255), coloured);
    cv::inRange(hsv, cv::Scalar(0, 0, 170), cv::Scalar(179, 165, 255), light);
    coloured.setTo(0, light);
    cv::bitwise_and(coloured, roi.mask, coloured);
    // Suppress cloth-like colours and other detections only when measuring new evidence.
    for (int y = 0; y < hsv.rows; ++y)
        for (int x = 0; x < hsv.cols; ++x) {
            const auto pixel = hsv.at<cv::Vec3b>(y, x);
            const int delta = std::abs(int(pixel[0]) - cloth.hue);
            if (pixel[1] >= 50 && std::min(delta, 180 - delta) <= 10) coloured.at<uchar>(y, x) = 0;
        }
    for (size_t other = 0; other < circles.size(); ++other) if (other != index)
        cv::circle(coloured, {cvRound(circles[other][0]) - origin.x, cvRound(circles[other][1]) - origin.y},
                   std::max(1, cvRound(circles[other][2])), cv::Scalar(0), cv::FILLED);
    const auto kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, {3, 3});
    cv::Mat outerBoundary, innerBoundary;
    cv::dilate(inner, outerBoundary, kernel);
    cv::erode(inner, innerBoundary, kernel, {-1, -1}, 1, cv::BORDER_CONSTANT, 0);
    cv::subtract(outerBoundary, innerBoundary, boundary);
    cv::Mat labels, stats, centroids, admitted = cv::Mat::zeros(roi.mask.size(), CV_8UC1);
    const int components = cv::connectedComponentsWithStats(coloured, labels, stats, centroids, 8, CV_32S);
    attempt.minimumAdded = std::max(12, cvCeil(0.02 * cv::countNonZero(inner)));
    for (int component = 1; component < components; ++component) {
        const cv::Mat region = labels == component;
        cv::Mat touch, added;
        cv::bitwise_and(region, boundary, touch);
        region.copyTo(added);
        added.setTo(0, inner);
        if (cv::countNonZero(touch) >= 3 && cv::countNonZero(added) >= attempt.minimumAdded)
            cv::bitwise_or(admitted, added, admitted);
    }
    attempt.addedPixels = cv::countNonZero(admitted);
    // Preserve every original crop pixel. Only supported coloured extensions join the mask.
    cv::bitwise_or(inner, admitted, roi.mask);
    attempt.candidate.roi = std::move(roi);
    attempt.candidate.baselineType = classifyBallType(segmentBallKMeans(attempt.candidate.roi, 2));
    attempt.candidate.colours = scoreBallColours(attempt.candidate.roi, model);
    identityEvidence(attempt.candidate, true);
    attempt.supported = attempt.addedPixels >= attempt.minimumAdded &&
        attempt.candidate.baselineType.type != "White" &&
        histogramBallNumber(attempt.candidate.baselineType.type, attempt.candidate.colours.bestColour) > 0;
    return attempt;
}

int chooseExpansion(const std::vector<ExpansionAttempt>& attempts) {
    for (size_t i = 1; i < attempts.size(); ++i)
        if (attempts[i - 1].padding + 1 == attempts[i].padding && attempts[i - 1].supported && attempts[i].supported &&
            attempts[i - 1].candidate.baselineType.type == attempts[i].candidate.baselineType.type &&
            attempts[i - 1].candidate.colours.bestColour == attempts[i].candidate.colours.bestColour)
            return int(i); // First confirming padding, not whichever scores best on evaluation labels.
    return -1;
}

Candidate rawWhitePadding(const cv::Mat& image, const Candidate& original, const BallColourModel& model, int padding) {
    Candidate grown = original;
    auto cropCircle = original.circle;
    cropCircle[2] += padding;
    grown.roi = extractBall(image, cropCircle);
    grown.baselineType = classifyBallType(segmentBallKMeans(grown.roi, 2));
    grown.colours = scoreBallColours(grown.roi, model);
    identityEvidence(grown, true);
    return grown; // Stored circle/centre remain the original Hough geometry.
}

struct RowRadiusModel {
    double intercept = 0, slope = 0, minimumRow = 0, maximumRow = 0;
    bool slopeClamped = false;
    double expected(double row) const {return std::max(1.0, intercept + slope * row);}
};
double median(std::vector<double> values) {
    if (values.empty()) throw std::invalid_argument("Cannot take median of empty samples");
    std::sort(values.begin(), values.end());
    const size_t middle = values.size() / 2;
    return values.size() % 2 ? values[middle] : .5 * (values[middle - 1] + values[middle]);
}
RowRadiusModel fitRowRadius(const std::vector<cv::Vec3f>& circles) {
    if (circles.empty()) throw std::invalid_argument("No circles for row radius fit");
    RowRadiusModel model;
    model.minimumRow = model.maximumRow = circles.front()[1];
    std::vector<double> slopes, residuals;
    for (size_t i = 0; i < circles.size(); ++i) {
        if (!std::isfinite(circles[i][1]) || !std::isfinite(circles[i][2]) || circles[i][2] <= 0)
            throw std::invalid_argument("Invalid row radius sample");
        model.minimumRow = std::min(model.minimumRow, double(circles[i][1]));
        model.maximumRow = std::max(model.maximumRow, double(circles[i][1]));
        for (size_t j = 0; j < i; ++j) {
            const double dy = double(circles[i][1]) - circles[j][1];
            if (std::abs(dy) >= 10) slopes.push_back((double(circles[i][2]) - circles[j][2]) / dy);
        }
    }
    const double rawSlope = slopes.empty() ? 0 : median(slopes);
    model.slope = std::max(0.0, rawSlope); // Physical prior: size does not shrink towards the near end.
    model.slopeClamped = rawSlope < 0;
    for (const auto& c : circles) residuals.push_back(c[2] - model.slope * c[1]);
    model.intercept = median(residuals);
    return model;
}
double cueSizeSupport(double radius, double expected) {
    if (!std::isfinite(radius) || !std::isfinite(expected) || radius <= 0 || expected <= 0)
        throw std::invalid_argument("Invalid cue radius");
    const double relative = std::min(1.0, radius / expected);
    return relative * relative; // Apparent area ratio, capped so oversized circles receive no bonus.
}
void checkRowCueSize() {
    const auto model = fitRowRadius({{0,100,15},{0,200,20},{0,300,25},{0,400,30},{0,250,100}});
    if (std::abs(model.slope - .05) > 1e-9 || std::abs(model.expected(250)-22.5) > 1e-9)
        throw std::runtime_error("Robust row radius fit failed");
    const auto flat = fitRowRadius({{0,100,20},{50,100,24}});
    if (flat.slope != 0 || flat.expected(500) != 22) throw std::runtime_error("Constant row fallback failed");
    const auto decreasing = fitRowRadius({{0,100,30},{0,200,20},{0,300,10}});
    if (!decreasing.slopeClamped || decreasing.slope != 0) throw std::runtime_error("Monotonic row prior failed");
    if (std::abs(cueSizeSupport(10,20)-.25)>1e-12 || cueSizeSupport(20,20)!=1 || cueSizeSupport(40,20)!=1 ||
        cueSizeSupport(12,20) >= cueSizeSupport(18,20)) throw std::runtime_error("Cue size support failed");
    // Cue-specific support must change a cue conflict without suppressing the stripe alternative.
    std::vector<Candidate> conflict(2);
    conflict[0].identityScore[0]=.90; conflict[0].identityScore[14]=.30;
    conflict[1].identityScore[0]=.98; conflict[1].identityScore[14]=.10;
    for (auto& c : conflict) c.eligible[0]=c.eligible[14]=true;
    auto before = assignUnique(conflict,false,true);
    conflict[1].identityScore[0] *= cueSizeSupport(10,20);
    auto after = assignUnique(conflict,false,true);
    if (before != std::vector<int>({14,0}) || after != std::vector<int>({0,14}))
        throw std::runtime_error("Targeted cue conflict check failed");
}
} // namespace

BallColourModel loadFrozenIdentityColourModel(const std::filesystem::path& path) { return loadModel(path); }

std::vector<int> recogniseUniqueBallEvidence(const std::vector<BallTypeResult>& types,
    const std::vector<BallColourScores>& colours, int reservedCue) {
    if (types.size()!=colours.size() || reservedCue < -1 || reservedCue>=int(types.size()))
        throw std::invalid_argument("Invalid shared recognition evidence");
    std::vector<Candidate> candidates(types.size());
    for (size_t i=0;i<types.size();++i) {
        auto& c=candidates[i]; c.baselineType=types[i]; c.colours=colours[i]; identityEvidence(c,true);
        if (reservedCue>=0) {
            if (int(i)==reservedCue) {c.eligible.fill(false); c.eligible[0]=c.identityScore[0]>1e-9;}
            else {c.eligible[0]=false; c.identityScore[0]=0;}
        }
    }
    return assignUnique(candidates,false,true);
}

std::vector<int> recogniseUniqueBallROIs(const std::vector<BallROI>& rois, const BallColourModel& model) {
    std::vector<Candidate> candidates;
    for (const auto& roi : rois) {
        Candidate c;
        c.roi = roi;
        c.baselineType = classifyBallType(segmentBallKMeans(roi, 2));
        c.colours = scoreBallColours(roi, model);
        identityEvidence(c, true);
        candidates.push_back(std::move(c));
    }
    return assignUnique(candidates, false, true);
}

static void runIdentityHeuristicsComparison(const std::filesystem::path& directory, bool bestEffort) {
    std::filesystem::create_directories(directory);
    const std::filesystem::path modelPath = "output/experiments/ball_colour_histograms/training/models.yml";
    const auto model = loadModel(modelPath);
    std::filesystem::copy_file(modelPath, directory / "frozen_models.yml", std::filesystem::copy_options::overwrite_existing);
    auto configuration = output(directory / "configuration.txt");
    configuration << "Fixed A/B/C/D experiment. One implementation, no main-pipeline changes.\n"
        << "Best-effort mode=" << bestEffort << ". Original conservative outputs are not overwritten.\n"
        << "A: current K2 type + frozen AllBalls HSV body histograms.\n"
        << "B: independent identity scores, no uniqueness. C: Hungarian unique identities. D: C plus size weighting.\n"
        << "Type compatibility exp(-0.5*(distance/0.10)^2), distance to light-fraction intervals: "
        << "cue [.85,1], solid [0,.30], stripe [.30,.85].\n"
        << "Numbered identity score = type compatibility * histogram similarity. Cue score = light fraction * cue compatibility.\n"
        << (bestEffort ? "Unknown score=0. All supported identities allowed; no .35 cutoff or .15 shortlist. "
            "Scores <=1e-9 count as numerical zero. Require usable histogram evidence or a baseline White classification.\n"
            : "Unknown score=0.35. Only alternatives within 0.15 of local best and strictly above 0.35 are admissible.\n")
        << "C/D: 16 identity columns plus one Unknown column per candidate. Missing identities allowed. "
        << "Exactly tied Unknown is preferred by 1e-9. No removal based only on overlap.\n"
        << "D: inverse homography Jacobian in source pixels/mm; 43mm diameter; major semiaxis estimates apparent sphere radius. "
        << "One median radius scale from AllBalls detections only.\n"
        << "D geometry support exp(-0.5*(log(radius/expected)/log(1.5))^2); identity multiplier .75+.25*support. "
        << "Invalid geometry is neutral (multiplier 1). Unknown score remains the policy's fixed value.\n"
        << "Geometry is approximate: homography describes the cloth plane, not sphere height or camera calibration.\n"
        << "All weights fixed before evaluation. Scores and normalised relative supports are not calibrated probabilities.\n"
        << "No K6 or tuned CCA. Frozen histogram body mask: 3x3 erosion, HSV V>=170/S<=165 light exclusion; "
        << "18x8x8 bins; chroma<15 hue canonicalisation; min12 body pixels; original smoothing/normalisation.\n";
    configuration.close();
    const cv::Mat inverse = getTableTransformation().inv();
    const double radiusScale = geometryCalibration(directory, inverse);
    auto calibration = output(directory / "geometry_calibration.txt");
    calibration << "AllBalls median detectedRadius / inverse-homography majorRadius = " << std::setprecision(12) << radiusScale << '\n';
    calibration.close();
    auto metrics = output(directory / "metrics.csv");
    metrics << "split,image,method,groundTruth,detections,TP,FP,FN,precision,recall,F1,meanError,correctTypes,correctNumbers,"
               "typeAccuracy,numberAccuracy,numberRecall,numberPrecision,unknownPredictions,extraDuplicateClaims\n";
    auto predictions = output(directory / "predictions.csv");
    predictions << "image,method,candidate,trueNumber,type,number,localBest,localBestScore,selectedRawScore,selectedGeometryScore,"
                   "selectedRank,typeCorrect,numberCorrect\n";
    auto evidence = output(directory / "evidence.csv");
    evidence << "image,candidate,number,type,colour,colourSupport,typeCompatibility,identityScore,relativeSupport,eligible,geometryWeightedScore\n";
    auto geometryCSV = output(directory / "geometry.csv");
    geometryCSV << "image,candidate,x,y,radius,tableX,tableY,valid,rawMajorRadius,rawMinorRadius,angle,expectedRadius,radiusRatio,compatibility,factor\n";
    auto typeCSV = output(directory / "type_scores.csv");
    typeCSV << "image,candidate,baselineType,lightPercent,cueCompatibility,solidCompatibility,stripeCompatibility\n";
    auto duplicatesCSV = output(directory / "duplicate_claims.csv");
    duplicatesCSV << "image,method,identity,count,candidates\n";
    auto changes = output(directory / "changes.csv");
    changes << "image,comparison,candidate,trueNumber,beforeType,afterType,beforeNumber,afterNumber,typeChange,numberChange\n";
    auto confidence = output(directory / "confidence.csv");
    confidence << "image,method,candidate,number,hasUsableEvidence,lowSupport,belowOldScoreCutoff,outsideOldShortlist\n";
    auto policy = output(directory / "policy_comparison.csv");
    policy << "image,method,candidate,trueNumber,conservativeNumber,currentNumber,change\n";
    const std::array<std::pair<int, int>, 5> comparisons = {{{0,1},{1,2},{2,3},{0,2},{0,3}}};
    std::map<std::pair<std::string, std::string>, int> changeCounts;
    std::array<Counts, 4> totals{};
    std::array<int, 4> conservativeCorrect{}, conservativeUnknown{}, policyFixes{}, policyRegressions{}, lowSupportTotals{};
    std::map<std::tuple<std::string, std::string, int, int>, int> confusion;

    for (const auto& filename : config::allImages()) {
        const std::string stem = std::filesystem::path(filename).stem().string();
        const std::string split = filename == "AllBalls.jpeg" ? "training_resubstitution" : "evaluation";
        const auto imageDirectory = directory / stem;
        std::filesystem::create_directories(imageDirectory);
        const cv::Mat image = cv::imread(config::mediaDirectory + filename);
        if (image.empty()) throw std::runtime_error("Could not load " + filename);
        const auto circles = runDetection(image, HoughParams{}).filteredCircles;
        const auto centres = transformCentres(circles);
        std::vector<Candidate> candidates;
        std::array<std::vector<int>, 4> numbers;
        std::array<std::vector<std::string>, 4> types;
        cv::Mat sizeOverlay = image.clone();
        for (size_t i = 0; i < circles.size(); ++i) {
            Candidate c;
            c.circle = circles[i]; c.tableCentre = centres[i]; c.roi = extractBall(image, circles[i]);
            c.baselineType = classifyBallType(segmentBallKMeans(c.roi, 2));
            c.colours = scoreBallColours(c.roi, model);
            c.geometry = measureGeometry(c.circle, c.tableCentre, inverse, radiusScale);
            identityEvidence(c, bestEffort);
            numbers[0].push_back(histogramBallNumber(c.baselineType.type, c.colours.bestColour));
            types[0].push_back(c.baselineType.type);
            numbers[1].push_back(bestEffort
                ? (hasUsableEvidence(c) && c.bestScore > 1e-9 ? c.localBest : -1)
                : (c.bestScore > unknownScore ? c.localBest : -1));
            const double sum = std::accumulate(c.identityScore.begin(), c.identityScore.end(), 0.0);
            for (int n = 0; n <= 15; ++n) {
                const int colour = n == 0 ? 0 : n <= 8 ? n : n - 8;
                evidence << filename << ',' << i + 1 << ',' << n << ',' << typeFor(n) << ',' << colour << ','
                    << (n == 0 ? c.baselineType.lightPercent / 100 : c.colours.similarity[colour - 1]) << ','
                    << c.typeSupport[n == 0 ? 0 : n <= 8 ? 1 : 2] << ',' << c.identityScore[n] << ','
                    << ratio(c.identityScore[n], sum) << ',' << c.eligible[n] << ',' << c.identityScore[n] * c.geometry.factor << '\n';
            }
            const auto& g = c.geometry;
            geometryCSV << filename << ',' << i + 1 << ',' << circles[i][0] << ',' << circles[i][1] << ',' << circles[i][2]
                << ',' << centres[i].x << ',' << centres[i].y << ',' << g.valid << ',' << g.rawRadius << ',' << g.minorRadius
                << ',' << g.angle << ',' << g.expectedRadius << ',' << g.radiusRatio << ',' << g.compatibility << ',' << g.factor << '\n';
            typeCSV << filename << ',' << i + 1 << ',' << c.baselineType.type << ',' << c.baselineType.lightPercent;
            for (double value : c.typeSupport) typeCSV << ',' << value;
            typeCSV << '\n';
            drawGeometry(sizeOverlay, c);
            cv::putText(sizeOverlay, std::to_string(i + 1), {cvRound(c.circle[0]), cvRound(c.circle[1])},
                        cv::FONT_HERSHEY_SIMPLEX, 0.35, {255, 255, 255}, 1);
            candidates.push_back(std::move(c));
        }
        numbers[2] = assignUnique(candidates, false, bestEffort);
        numbers[3] = assignUnique(candidates, true, bestEffort);
        // Recompute the conservative controls from identical cached evidence, with no new inference.
        auto conservativeNumbers = numbers;
        if (bestEffort) {
            auto control = candidates;
            conservativeNumbers[1].clear();
            for (auto& c : control) {
                identityEvidence(c, false);
                conservativeNumbers[1].push_back(c.bestScore > unknownScore ? c.localBest : -1);
            }
            conservativeNumbers[2] = assignUnique(control, false);
            conservativeNumbers[3] = assignUnique(control, true);
        }
        for (size_t m = 1; m < methods.size(); ++m)
            for (int n : numbers[m]) types[m].push_back(typeFor(n));
        for (size_t m = 0; m < methods.size(); ++m) {
            saveFinalResults(image, directory / "final_results" / methods[m] / stem, circles, centres, types[m], numbers[m]);
            cv::Mat flagged = cv::imread((directory / "final_results" / methods[m] / stem / "original_numbered.png").string());
            for (size_t i = 0; i < candidates.size(); ++i) {
                const auto& c = candidates[i];
                const int number = numbers[m][i];
                const bool weak = lowSupport(c, number, m == 3);
                confidence << filename << ',' << methods[m] << ',' << i + 1 << ',' << number << ',' << hasUsableEvidence(c) << ','
                    << weak << ',' << (number >= 0 && c.identityScore[number] * (m == 3 ? c.geometry.factor : 1) <= unknownScore) << ','
                    << (number >= 0 && c.identityScore[number] < c.bestScore - maximumAlternativeDrop) << '\n';
                if (weak) {
                    const cv::Point centre(cvRound(c.circle[0]), cvRound(c.circle[1]));
                    cv::circle(flagged, centre, cvRound(c.circle[2]) + 3, {0, 160, 255}, 2, cv::LINE_AA);
                    cv::putText(flagged, "?", centre + cv::Point(cvRound(c.circle[2]) + 4, 0),
                                cv::FONT_HERSHEY_SIMPLEX, 0.6, {0, 160, 255}, 2, cv::LINE_AA);
                }
            }
            cv::rectangle(flagged, {0, 0}, {flagged.cols - 1, 25}, {25, 25, 25}, cv::FILLED);
            cv::putText(flagged, "Amber rings / ?: guesses that fail the old score or shortlist rules (not calibrated confidence)",
                        {8, 18}, cv::FONT_HERSHEY_SIMPLEX, 0.43, {0, 190, 255}, 1, cv::LINE_AA);
            saveImage(directory / "final_results" / methods[m] / stem / "confidence_overlay.png", flagged);
        }
        saveImage(imageDirectory / "geometry.png", sizeOverlay);

        // No annotation is an input to detection, evidence, geometry weighting or assignment.
        const auto truth = loadLabelledGroundTruth(config::groundTruthPath, filename);
        if (truth.empty()) throw std::runtime_error("Missing evaluation annotations for " + filename);
        std::vector<cv::Point2f> truthCentres;
        for (const auto& ball : truth) truthCentres.push_back(ball.centre);
        const auto evaluated = evaluateDetections(centres, truthCentres, config::matchTolerance);
        for (size_t m = 0; m < methods.size(); ++m) {
            Counts count{int(truth.size()), evaluated.TP, evaluated.FP, evaluated.FN, 0, 0, 0, 0, evaluated.meanError * evaluated.TP};
            std::map<int, std::vector<int>> claims;
            for (size_t i = 0; i < candidates.size(); ++i) {
                const auto& c = candidates[i];
                const int match = evaluated.predictionToGroundTruth[i], actual = match < 0 ? -1 : truth[match].number;
                const int n = numbers[m][i];
                const int previous = conservativeNumbers[m][i];
                const auto policyChange = change(previous, n, actual);
                policy << filename << ',' << methods[m] << ',' << i + 1 << ',' << actual << ',' << previous << ',' << n << ',' << policyChange << '\n';
                if (split == "evaluation") {
                    conservativeCorrect[m] += actual >= 0 && previous == actual;
                    conservativeUnknown[m] += previous < 0;
                    policyFixes[m] += policyChange == "corrected";
                    policyRegressions[m] += policyChange == "regressed";
                    lowSupportTotals[m] += lowSupport(c, n, m == 3);
                }
                if (n < 0) ++count.unknown;
                else claims[n].push_back(int(i + 1));
                if (actual >= 0) {
                    count.type += types[m][i] == typeFor(actual); count.number += n == actual;
                    ++confusion[{split, methods[m], actual, n}];
                }
                int rank = 0;
                if (n >= 0) {rank = 1; for (double s : c.identityScore) rank += s > c.identityScore[n] + 1e-12;}
                predictions << filename << ',' << methods[m] << ',' << i + 1 << ',' << actual << ',' << types[m][i] << ',' << n
                    << ',' << c.localBest << ',' << c.bestScore << ',' << (n < 0 ? 0 : c.identityScore[n]) << ','
                    << (n < 0 ? 0 : c.identityScore[n] * c.geometry.factor) << ',' << rank << ',';
                if (actual >= 0) predictions << (types[m][i] == typeFor(actual)) << ',' << (n == actual);
                else predictions << ',';
                predictions << '\n';
            }
            for (const auto& [n, ids] : claims) if (ids.size() > 1) {
                count.duplicates += int(ids.size() - 1);
                duplicatesCSV << filename << ',' << methods[m] << ',' << n << ',' << ids.size() << ',';
                for (size_t j = 0; j < ids.size(); ++j) duplicatesCSV << (j ? ";" : "") << ids[j];
                duplicatesCSV << '\n';
            }
            if (m >= 2 && count.duplicates) throw std::runtime_error("Uniqueness assignment produced duplicate identities");
            metrics << split << ',' << filename << ',' << methods[m] << ','; writeCounts(metrics, count); metrics << '\n';
            if (split == "evaluation") totals[m].add(count);
        }
        for (size_t i = 0; i < candidates.size(); ++i) {
            const auto& c = candidates[i];
            const int match = evaluated.predictionToGroundTruth[i], actual = match < 0 ? -1 : truth[match].number;
            for (const auto& [before, after] : comparisons) {
                const std::string comparison = methods[before] + "_to_" + methods[after];
                const std::string numberChange = change(numbers[before][i], numbers[after][i], actual);
                const int beforeType = types[before][i] == "White" ? 0 : types[before][i] == "Solid" ? 1 : types[before][i] == "Striped" ? 2 : -1;
                const int afterType = types[after][i] == "White" ? 0 : types[after][i] == "Solid" ? 1 : types[after][i] == "Striped" ? 2 : -1;
                const int actualType = actual < 0 ? -1 : actual == 0 ? 0 : actual <= 8 ? 1 : 2;
                changes << filename << ',' << comparison << ',' << i + 1 << ',' << actual << ',' << types[before][i] << ',' << types[after][i]
                    << ',' << numbers[before][i] << ',' << numbers[after][i] << ',' << change(beforeType, afterType, actualType) << ',' << numberChange << '\n';
                if (split == "evaluation") ++changeCounts[{comparison, numberChange}];
            }
            std::vector<ImagePanel> panels;
            for (size_t m = 0; m < methods.size(); ++m) {
                cv::Mat crop = enlarged(c.roi.image);
                const cv::Scalar colour = actual < 0 ? cv::Scalar(180, 180, 180) : numbers[m][i] == actual ? cv::Scalar(50, 240, 50) : cv::Scalar(70, 70, 255);
                const std::string prediction = numbers[m][i] < 0 ? "Unknown" : types[m][i] + " " + label(numbers[m][i])
                    + (lowSupport(c, numbers[m][i], m == 3) ? " ?" : "");
                cv::putText(crop, prediction, {8, 25}, cv::FONT_HERSHEY_SIMPLEX, 0.55, {0, 0, 0}, 3, cv::LINE_AA);
                cv::putText(crop, prediction, {8, 25}, cv::FONT_HERSHEY_SIMPLEX, 0.55, colour, 1, cv::LINE_AA);
                panels.push_back({names[m], crop});
            }
            const int half = cvCeil(std::max(double(c.circle[2]), c.geometry.expectedRadius)) + 8;
            const cv::Rect cropRect = cv::Rect(cvRound(c.circle[0]) - half, cvRound(c.circle[1]) - half, 2 * half + 1, 2 * half + 1)
                                    & cv::Rect(0, 0, image.cols, image.rows);
            cv::Mat sizeCrop = image(cropRect).clone();
            drawGeometry(sizeCrop, c, cropRect.tl());
            panels.push_back({"Size: detected orange / expected cyan", enlarged(sizeCrop)});
            panels.push_back({"Frozen histogram body mask", enlarged(c.colours.bodyMask)});
            const auto grid = combineImages(panels, 3);
            cv::Mat preview(grid.rows + 280, grid.cols, CV_8UC3, cv::Scalar(32, 32, 32));
            grid.copyTo(preview(cv::Rect(0, 60, grid.cols, grid.rows)));
            cv::putText(preview, filename + " candidate " + std::to_string(i + 1) + " | actual " + (actual < 0 ? "unmatched" : label(actual)),
                {10, 22}, cv::FONT_HERSHEY_SIMPLEX, 0.5, {245, 245, 245}, 1);
            std::ostringstream description;
            description << std::fixed << std::setprecision(2) << "Radius ratio " << c.geometry.radiusRatio << " | D weight "
                << c.geometry.factor << " | green/red prediction = correct/wrong number";
            cv::putText(preview, description.str(), {10, 45}, cv::FONT_HERSHEY_SIMPLEX, 0.4, {220, 220, 220}, 1);
            const int chartY = grid.rows + 82, columnWidth = grid.cols / 2;
            cv::putText(preview, bestEffort ? "Identity scores (not probabilities); Unknown = 0; ? = fails old confidence rules"
                                          : "Identity compatibility scores (not probabilities); Unknown option = 0.35", {10, chartY},
                cv::FONT_HERSHEY_SIMPLEX, 0.42, {245, 245, 245}, 1);
            for (int n = 0; n <= 15; ++n) {
                const int x = (n / 8) * columnWidth + 10, y = chartY + 20 + (n % 8) * 22;
                cv::rectangle(preview, {x + 65, y - 10}, {x + 65 + cvRound(180 * c.identityScore[n]), y}, {190, 130, 50}, cv::FILLED);
                std::string markers;
                for (int m = 1; m < 4; ++m) if (numbers[m][i] == n) markers += " " + std::string(1, char('A' + m));
                std::ostringstream value; value << std::fixed << std::setprecision(2) << c.identityScore[n];
                cv::putText(preview, label(n), {x, y}, cv::FONT_HERSHEY_SIMPLEX, 0.4, {245, 245, 245}, 1);
                cv::putText(preview, value.str() + markers, {x + 250, y}, cv::FONT_HERSHEY_SIMPLEX, 0.4, {245, 245, 245}, 1);
            }
            saveImage(imageDirectory / ("candidate_" + std::to_string(i + 1) + ".png"), preview);
        }
        std::vector<ImagePanel> overview;
        for (size_t m = 0; m < methods.size(); ++m)
            overview.push_back({names[m], cv::imread((directory / "final_results" / methods[m] / stem / "table_numbered.png").string())});
        saveImage(imageDirectory / "table_comparison.png", combineImages(overview, 2));
        std::cout << filename << ": compared " << circles.size() << " candidates across A/B/C/D\n";
    }
    auto matrix = output(directory / "confusion.csv");
    matrix << "split,method,actual,predicted,count\n";
    for (const auto& split : {"training_resubstitution", "evaluation"})
        for (const auto& method : methods)
            for (int actual = 0; actual <= 15; ++actual)
                for (int predicted = -1; predicted <= 15; ++predicted)
                    matrix << split << ',' << method << ',' << actual << ',' << predicted << ',' << confusion[{split, method, actual, predicted}] << '\n';
    auto summary = output(directory / "summary.md");
    summary << (bestEffort ? "# Best-effort A/B/C/D identity heuristics\n\n" : "# A/B/C/D identity heuristics\n\n")
        << "Shared K=2 features and frozen AllBalls HSV histograms. No K6/CCA search settings used. "
        << "A is the existing classifier; B uses independent colour/type compatibility; C assigns identities "
        << "jointly with missing identities and unlimited Unknowns; D adds weak source-radius consistency.\n\n"
        << "| Method | Correct types | Correct numbers | Unknown predictions | Extra duplicate identity claims |\n|---|---:|---:|---:|---:|\n";
    for (size_t m = 0; m < methods.size(); ++m) {
        metrics << "evaluation,ALL," << methods[m] << ','; writeCounts(metrics, totals[m]); metrics << '\n';
        summary << "| " << names[m] << " | " << totals[m].type << '/' << totals[m].tp << " | " << totals[m].number << '/'
            << totals[m].tp << " | " << totals[m].unknown << " | " << totals[m].duplicates << " |\n";
    }
    summary << "\n| Comparison | Numbers corrected | Previously correct numbers lost |\n|---|---:|---:|\n";
    for (const auto& [before, after] : comparisons) {
        const auto comparison = methods[before] + "_to_" + methods[after];
        summary << "| " << names[before] << " -> " << names[after] << " | " << changeCounts[{comparison, "corrected"}]
            << " | " << changeCounts[{comparison, "regressed"}] << " |\n";
    }
    if (bestEffort) {
        summary << "\nOnly the abstention policy changes relative to the original experiment. B gives the best supported "
            << "guess; C/D permit every supported identity, Unknown=0, no .15 shortlist. Values <=1e-9 are numerical zero. "
            << "A candidate still needs usable body-colour evidence or a baseline White classification. "
            << "Uniqueness can require Unknown even with usable evidence, including when there are more than 16 candidates.\n\n"
            << "| Stage | Conservative correct | Best-effort correct | Conservative Unknown | Best-effort Unknown | Corrections | Regressions | Low-support guesses |\n"
            << "|---|---:|---:|---:|---:|---:|---:|---:|\n";
        for (size_t m = 0; m < methods.size(); ++m)
            summary << "| " << names[m] << " | " << conservativeCorrect[m] << " | " << totals[m].number << " | "
                << conservativeUnknown[m] << " | " << totals[m].unknown << " | " << policyFixes[m] << " | "
                << policyRegressions[m] << " | " << lowSupportTotals[m] << " |\n";
        summary << "\nControls were recomputed from the same cached candidates/scores, not refitted. "
            << "confidence.csv flags selected scores <=.35 or alternatives farther than .15 from local best "
            << "(D uses its weighted score for the .35 check). Amber rings in confidence_overlay.png and ? in "
            << "candidate comparisons mark these guesses without replacing their identity. Flags are heuristics, "
            << "not calibrated probabilities. policy_comparison.csv gives every conservative/current decision.\n";
    }
    summary << "\nTable1–6 totals exclude AllBalls, which is shown separately as training resubstitution. "
        << "All stages retain all Hough candidates, including Unknown identities; detection remains " << totals[0].tp << " TP, "
        << totals[0].fp << " FP and " << totals[0].fn << " FN. Identity uniqueness is not physical circle deduplication. "
        << "No candidate is removed just for overlapping another. False detections can still steal identities, "
        << "and global assignment can turn duplicates into different incorrect numbers.\n\n"
        << "Type evidence uses Gaussian distance to the original K2 decision intervals, width .10; numbered scores "
        << "multiply type and histogram support. Cue uses its light fraction as colour evidence. B chooses the best "
        << (bestEffort ? "supported identity above numerical zero. C/D consider every supported identity, with Unknown=0. Hungarian assignment "
                       : "identity above .35. C/D allow only identities within .15 of that best and above .35. Hungarian assignment ")
        << "maximises total score with one column per identity and separate Unknown columns. These are fixed heuristic "
        << "scores, not calibrated probabilities. Normalised relative supports in evidence.csv are diagnostic only.\n\n"
        << "Size comes from the inverse homography Jacobian in source pixels/mm and the known 43mm diameter. "
        << "The major projected semiaxis approximates sphere radius; the plane ellipse is also shown in grey. "
        << "A single median AllBalls detected/expected radius ratio, " << radiusScale << ", scales the prediction. "
        << "No evaluation locations or radii fit this model. D multiplies each candidate's identity scores by "
        << ".75+.25*exp(-.5*(log(radius/expected)/log(1.5))^2), leaving the policy's Unknown score unchanged. Invalid geometry is neutral. "
        << "The planar model is approximate because balls extend above the cloth and camera intrinsics are unknown.\n\n"
        << (bestEffort ? "This policy is an evaluation-informed follow-up after inspecting the conservative results. "
            "Colour/type/geometry scoring weights stay fixed; no additional adjustments were made during this run. "
            : "All weights were declared before this run, with no evaluation-based adjustment. ")
        << "The six images have "
        << "been inspected in earlier development, so this is not pristine unseen-data validation. "
        << "Choosing a stage after inspecting its scores is method selection, not a new independent test. "
        << "The main pipeline is unchanged.\n\n"
        << "Outputs: configuration.txt; frozen_models.yml; training_geometry.csv; geometry_calibration.txt; "
        << "metrics.csv; predictions.csv; evidence.csv (all 16 scores); type_scores.csv; geometry.csv; "
        << "duplicate_claims.csv; changes.csv; confusion.csv; final_results/<stage>/<image>/; "
        << "per-image geometry, candidate comparisons with score bars, and whole-table comparisons.\n";
    std::cout << "Saved A/B/C/D identity heuristics to " << directory << '\n';
}

void runIdentityHeuristics(const std::filesystem::path& directory) {
    runIdentityHeuristicsComparison(directory, false);
}

void runBestEffortIdentityHeuristics(const std::filesystem::path& directory) {
    runIdentityHeuristicsComparison(directory, true);
}

void runWhiteExpansionIdentityHeuristics(const std::filesystem::path& directory) {
    const std::array<std::string, 8> stage = {"A_baseline", "B_scores", "C_unique", "D_white_expansion",
        "D_raw_pad1", "D_raw_pad2", "D_raw_pad3", "D_raw_pad4"};
    const std::array<std::string, 8> title = {"A: K2 + HSV", "B: independent scores", "C: unique identities", "D: guarded white expansion",
        "D: raw white padding +1", "D: raw white padding +2", "D: raw white padding +3", "D: raw white padding +4"};
    const std::filesystem::path source = "output/experiments/ball_colour_histograms/training/models.yml";
    const auto model = loadModel(source);
    std::filesystem::create_directories(directory);
    if (std::filesystem::exists(directory / "summary.md") && !std::filesystem::exists(directory / "guarded_initial_summary.md"))
        std::filesystem::copy_file(directory / "summary.md", directory / "guarded_initial_summary.md");
    std::filesystem::copy_file(source, directory / "frozen_models.yml", std::filesystem::copy_options::overwrite_existing);
    auto settings = output(directory / "configuration.txt");
    settings << "A/B/C reproduce the best-effort experiment; D replaces size weighting with white-crop expansion.\n"
        << "Only candidates whose original K2 hard type is White are probed. Padding 0,1,2,3,4 source pixels.\n"
        << "No Hough centres, radii, counts or table coordinates change. Only D's recognition evidence can change.\n"
        << "Cloth: per-image dominant hue from inset table (ellipse21), raw Hough radius+12 excluded; sample S>=50,V>=40; "
        << "minimum1000 pixels, hue peak with +/-3 smoothing. Invalid cloth model skips expansion.\n"
        << "Growth: coloured S>=80,V>=40, excluding light V>=170,S<=165, cloth hue distance<=10 with S>=50, "
        << "and every other filtered candidate disk. Eight-connected coloured components must touch the original "
        << "boundary band (dilate3-erode3) at >=3 pixels and add >=max(12,ceil(2% original disk area)) pixels.\n"
        << "Recognition mask = complete original disk OR admitted coloured annulus; never all padded background.\n"
        << "Require non-cue valid number evidence at two consecutive paddings with identical body colour and type. "
        << "Select first confirming (larger) padding. Otherwise retain original evidence.\n"
        << "Recompute K2 type and query histogram with the frozen AllBalls HSV models on accepted expanded mask. "
        << "D reruns best-effort unique assignment; no size multiplier. Other candidates keep their original evidence.\n"
        << "Best-effort Unknown=0 (+1e-9 tie preference), all supported identities, unchanged type width .10.\n"
        << "Raw fixed-padding ablations: expand original White crops by +1,+2,+3,+4 with the full circular mask, no cloth/neighbour gates. "
        << "Each is reported separately; no padding is selected from test labels. Added after inspecting the guarded run.\n"
        << "No evaluation-image parameter search. This is an evaluation-informed follow-up to observed cue errors.\n"
        << "Risk: cloth hue exclusion can also remove genuine green-body evidence; neighbour masks can hide real "
        << "overlapping colour, and undetected neighbours/shadows can contaminate growth. All are logged.\n";
    settings.close();
    auto attemptsCSV = output(directory / "white_expansion.csv");
    attemptsCSV << "image,candidate,padding,clothValid,addedPixels,minimumAdded,type,lightPercent,bodyColour,bodyPixels,"
                   "localBest,localScore,supported,chosen\n";
    auto rawCSV = output(directory / "raw_padding.csv");
    rawCSV << "image,candidate,padding,type,lightPercent,bodyColour,bodyPixels,localBest,localScore\n";
    auto clothCSV = output(directory / "cloth_models.csv");
    clothCSV << "image,valid,hue,samples\n";
    auto evidence = output(directory / "d_evidence.csv");
    evidence << "image,method,candidate,number,originalScore,dScore,eligible,selectedPadding\n";
    auto predictions = output(directory / "predictions.csv");
    predictions << "image,method,candidate,trueNumber,type,number,bodyColour,lightPercent,localBest,localScore,"
                   "directExpansion,padding,typeCorrect,numberCorrect\n";
    auto changes = output(directory / "changes.csv");
    changes << "image,comparison,candidate,trueNumber,beforeNumber,afterNumber,change,directExpansion\n";
    auto metrics = output(directory / "metrics.csv");
    metrics << "split,image,method,groundTruth,detections,TP,FP,FN,precision,recall,F1,meanError,correctTypes,correctNumbers,"
               "typeAccuracy,numberAccuracy,numberRecall,numberPrecision,unknownPredictions,extraDuplicateClaims\n";
    auto applied = output(directory / "applied_expansions.csv");
    applied << "image,candidate,trueNumber,padding,originalLightPercent,expandedLightPercent,expandedType,expandedBodyColour,cNumber,dNumber\n";
    std::array<Counts, 8> totals{};
    std::array<int, 8> dFixes{}, dRegressions{};
    int triggeredTotal = 0, appliedTotal = 0, appliedTrueCues = 0, corrected = 0, regressed = 0;

    for (const auto& filename : config::allImages()) {
        const std::string stem = std::filesystem::path(filename).stem().string();
        const bool evaluation = filename != "AllBalls.jpeg";
        const std::string split = evaluation ? "evaluation" : "training_resubstitution";
        const cv::Mat image = cv::imread(config::mediaDirectory + filename);
        if (image.empty()) throw std::runtime_error("Could not load " + filename);
        const auto detection = runDetection(image, HoughParams{});
        const auto& circles = detection.filteredCircles;
        const auto centres = transformCentres(circles);
        const auto cloth = expansionCloth(image, detection.rawCircles);
        clothCSV << filename << ',' << cloth.valid << ',' << cloth.hue << ',' << cloth.samples << '\n';
        std::vector<Candidate> candidates;
        for (size_t i = 0; i < circles.size(); ++i) {
            Candidate c;
            c.circle = circles[i]; c.tableCentre = centres[i]; c.roi = extractBall(image, circles[i]);
            c.baselineType = classifyBallType(segmentBallKMeans(c.roi, 2));
            c.colours = scoreBallColours(c.roi, model);
            identityEvidence(c, true);
            candidates.push_back(std::move(c));
        }
        auto dCandidates = candidates;
        std::array<std::vector<Candidate>, 4> rawCandidates;
        for (auto& variant : rawCandidates) variant = candidates;
        std::vector<int> padding(candidates.size(), 0);
        std::filesystem::create_directories(directory / stem);
        std::filesystem::create_directories(directory / "white_expansion" / stem);
        for (size_t i = 0; i < candidates.size(); ++i) {
            const auto& c = candidates[i];
            if (c.baselineType.type != "White") continue;
            if (evaluation) ++triggeredTotal;
            std::vector<ExpansionAttempt> attempts;
            for (int p = 0; p <= 4; ++p)
                attempts.push_back(expandWhiteCandidate(image, circles, i, c, model, cloth, p));
            const int chosen = chooseExpansion(attempts);
            if (chosen >= 0) {
                dCandidates[i] = attempts[chosen].candidate;
                padding[i] = attempts[chosen].padding;
                if (evaluation) ++appliedTotal;
            }
            for (int p = 1; p <= 4; ++p) {
                auto grown = rawWhitePadding(image, c, model, p);
                rawCSV << filename << ',' << i + 1 << ',' << p << ',' << grown.baselineType.type << ','
                    << grown.baselineType.lightPercent << ',' << grown.colours.bestColour << ',' << grown.colours.pixels
                    << ',' << grown.localBest << ',' << grown.bestScore << '\n';
                rawCandidates[p - 1][i] = std::move(grown);
            }
            std::vector<ImagePanel> crops, masks;
            for (size_t p = 0; p < attempts.size(); ++p) {
                const auto& a = attempts[p];
                const auto& grown = a.candidate;
                attemptsCSV << filename << ',' << i + 1 << ',' << a.padding << ',' << cloth.valid << ',' << a.addedPixels << ','
                    << a.minimumAdded << ',' << grown.baselineType.type << ',' << grown.baselineType.lightPercent << ','
                    << grown.colours.bestColour << ',' << grown.colours.pixels << ',' << grown.localBest << ',' << grown.bestScore << ','
                    << a.supported << ',' << (int(p) == chosen) << '\n';
                const std::string caption = "+" + std::to_string(a.padding) + "px " + grown.baselineType.type + " / " +
                    bodyColourName(grown.colours.bestColour) + (int(p) == chosen ? " USED" : "");
                crops.push_back({caption, enlarged(grown.roi.image)});
                masks.push_back({"Mask / added " + std::to_string(a.addedPixels) + " pixels", enlarged(grown.roi.mask)});
            }
            crops.insert(crops.end(), masks.begin(), masks.end());
            for (int p = 0; p <= 4; ++p) {
                const auto& raw = p ? rawCandidates[p - 1][i] : c;
                crops.push_back({"Raw +" + std::to_string(p) + "px " + raw.baselineType.type + " / " + bodyColourName(raw.colours.bestColour), enlarged(raw.roi.image)});
            }
            saveImage(directory / "white_expansion" / stem / ("candidate_" + std::to_string(i + 1) + ".png"), combineImages(crops, 5));
        }
        std::array<std::vector<int>, 8> numbers;
        std::array<std::vector<std::string>, 8> types;
        for (const auto& c : candidates) {
            numbers[0].push_back(histogramBallNumber(c.baselineType.type, c.colours.bestColour));
            types[0].push_back(c.baselineType.type);
            numbers[1].push_back(hasUsableEvidence(c) && c.bestScore > 1e-9 ? c.localBest : -1);
        }
        numbers[2] = assignUnique(candidates, false, true);
        numbers[3] = assignUnique(dCandidates, false, true); // Deliberately no geometric weight.
        for (int p = 0; p < 4; ++p) numbers[p + 4] = assignUnique(rawCandidates[p], false, true);
        auto forStage = [&](size_t m) -> const std::vector<Candidate>& {
            return m == 3 ? dCandidates : m >= 4 ? rawCandidates[m - 4] : candidates;
        };
        auto paddingFor = [&](size_t m, size_t i) {
            return m == 3 ? padding[i] : m >= 4 && candidates[i].baselineType.type == "White" ? int(m - 3) : 0;
        };
        for (size_t m = 1; m < stage.size(); ++m)
            for (int n : numbers[m]) types[m].push_back(typeFor(n));
        for (size_t m = 0; m < stage.size(); ++m)
            saveFinalResults(image, directory / "final_results" / stage[m] / stem, circles, centres, types[m], numbers[m]);
        for (size_t m = 3; m < stage.size(); ++m)
            for (size_t i = 0; i < candidates.size(); ++i)
                for (int n = 0; n <= 15; ++n)
                    evidence << filename << ',' << stage[m] << ',' << i + 1 << ',' << n << ',' << candidates[i].identityScore[n] << ','
                        << forStage(m)[i].identityScore[n] << ',' << forStage(m)[i].eligible[n] << ',' << paddingFor(m, i) << '\n';

        // Evaluation annotations cannot affect trigger, padding selection, masks or assignments.
        const auto truth = loadLabelledGroundTruth(config::groundTruthPath, filename);
        if (truth.empty()) throw std::runtime_error("Missing annotations for " + filename);
        std::vector<cv::Point2f> truthCentres;
        for (const auto& ball : truth) truthCentres.push_back(ball.centre);
        const auto e = evaluateDetections(centres, truthCentres, config::matchTolerance);
        for (size_t m = 0; m < stage.size(); ++m) {
            Counts count{int(truth.size()), e.TP, e.FP, e.FN, 0, 0, 0, 0, e.meanError * e.TP};
            std::array<int, 16> claims{};
            for (size_t i = 0; i < candidates.size(); ++i) {
                const int match = e.predictionToGroundTruth[i], actual = match < 0 ? -1 : truth[match].number;
                const auto& c = forStage(m)[i];
                const int n = numbers[m][i];
                if (n < 0) ++count.unknown;
                else if (++claims[n] > 1) ++count.duplicates;
                if (actual >= 0) {count.type += types[m][i] == typeFor(actual); count.number += n == actual;}
                predictions << filename << ',' << stage[m] << ',' << i + 1 << ',' << actual << ',' << types[m][i] << ',' << n << ','
                    << c.colours.bestColour << ',' << c.baselineType.lightPercent << ',' << c.localBest << ',' << c.bestScore << ','
                    << (paddingFor(m, i) > 0) << ',' << paddingFor(m, i) << ',';
                if (actual >= 0) predictions << (types[m][i] == typeFor(actual)) << ',' << (n == actual);
                else predictions << ',';
                predictions << '\n';
            }
            if (m >= 2 && count.duplicates) throw std::runtime_error("Duplicate identity after assignment");
            metrics << split << ',' << filename << ',' << stage[m] << ','; writeCounts(metrics, count); metrics << '\n';
            if (evaluation) totals[m].add(count);
        }
        for (size_t i = 0; i < candidates.size(); ++i) {
            const int match = e.predictionToGroundTruth[i], actual = match < 0 ? -1 : truth[match].number;
            for (size_t after = 3; after < stage.size(); ++after) for (const int before : {0, 2}) {
                const auto result = change(numbers[before][i], numbers[after][i], actual);
                changes << filename << ',' << stage[before] << "_to_" << stage[after] << ',' << i + 1 << ',' << actual << ','
                    << numbers[before][i] << ',' << numbers[after][i] << ',' << result << ',' << (paddingFor(after, i) > 0) << '\n';
                if (evaluation && before == 2) {
                    dFixes[after] += result == "corrected"; dRegressions[after] += result == "regressed";
                    if (after == 3) {corrected += result == "corrected"; regressed += result == "regressed";}
                }
            }
            if (padding[i]) {
                if (evaluation && actual == 0) ++appliedTrueCues;
                applied << filename << ',' << i + 1 << ',' << actual << ',' << padding[i] << ',' << candidates[i].baselineType.lightPercent
                    << ',' << dCandidates[i].baselineType.lightPercent << ',' << dCandidates[i].baselineType.type << ','
                    << dCandidates[i].colours.bestColour << ',' << numbers[2][i] << ',' << numbers[3][i] << '\n';
            }
            std::vector<ImagePanel> panels;
            for (size_t m = 0; m < stage.size(); ++m) {
                cv::Mat crop = enlarged(candidates[i].roi.image);
                const auto colour = actual < 0 ? cv::Scalar(180, 180, 180) : numbers[m][i] == actual ? cv::Scalar(50, 240, 50) : cv::Scalar(70, 70, 255);
                const std::string text = numbers[m][i] < 0 ? "Unknown" : types[m][i] + " " + label(numbers[m][i]);
                cv::putText(crop, text, {8, 25}, cv::FONT_HERSHEY_SIMPLEX, 0.55, {0, 0, 0}, 3, cv::LINE_AA);
                cv::putText(crop, text, {8, 25}, cv::FONT_HERSHEY_SIMPLEX, 0.55, colour, 1, cv::LINE_AA);
                panels.push_back({title[m], crop});
            }
            panels.push_back({"Guarded crop +" + std::to_string(padding[i]) + "px", enlarged(dCandidates[i].roi.image)});
            panels.push_back({"Guarded recognition mask", enlarged(dCandidates[i].roi.mask)});
            const auto grid = combineImages(panels, 3);
            cv::Mat preview(grid.rows + 55, grid.cols, CV_8UC3, cv::Scalar(32, 32, 32));
            grid.copyTo(preview(cv::Rect(0, 55, grid.cols, grid.rows)));
            cv::putText(preview, filename + " candidate " + std::to_string(i + 1) + " | actual " + (actual < 0 ? "unmatched" : label(actual)) +
                " | guarded padding " + std::to_string(padding[i]), {10, 23}, cv::FONT_HERSHEY_SIMPLEX, 0.5, {245, 245, 245}, 1);
            cv::putText(preview, "Green/red text = correct/wrong number. Stored Hough centre/radius are unchanged.",
                {10, 45}, cv::FONT_HERSHEY_SIMPLEX, 0.4, {220, 220, 220}, 1);
            saveImage(directory / stem / ("candidate_" + std::to_string(i + 1) + ".png"), preview);
        }
        std::vector<ImagePanel> overview;
        for (size_t m = 0; m < stage.size(); ++m)
            overview.push_back({title[m], cv::imread((directory / "final_results" / stage[m] / stem / "table_numbered.png").string())});
        saveImage(directory / stem / "table_comparison.png", combineImages(overview, 2));
        std::cout << filename << ": tested white expansion on all original White candidates\n";
    }
    auto summary = output(directory / "summary.md");
    summary << "# D: selective expansion of white detections\n\n"
        << "A/B/C retain best-effort K2+HSV inference. D replaces size weighting with colour growth around "
        << "originally White candidates. Frozen colour models and all detection geometry are unchanged.\n\n"
        << "| Stage | Correct types | Correct numbers | Unknown | Extra duplicate claims |\n|---|---:|---:|---:|---:|\n";
    for (size_t m = 0; m < stage.size(); ++m) {
        metrics << "evaluation,ALL," << stage[m] << ','; writeCounts(metrics, totals[m]); metrics << '\n';
        summary << "| " << title[m] << " | " << totals[m].type << '/' << totals[m].tp << " | " << totals[m].number << '/'
            << totals[m].tp << " | " << totals[m].unknown << " | " << totals[m].duplicates << " |\n";
    }
    summary << "\nAcross Table1–6, " << triggeredTotal << " original White candidates were probed; " << appliedTotal
        << " passed the guarded expansion rule. Relative to C, guarded D corrected " << corrected << " numbers and lost " << regressed << ".\n\n"
        << "Of the accepted guarded changes, " << appliedTrueCues << " were genuine cue balls whose local "
        << "type was changed to non-cue. This is a guard failure even when global assignment hides the change.\n\n"
        << "Raw +1..4 variants use full circular padding on every originally White candidate, without added-pixel "
        << "gates or stable-pair selection. All four are reported, not selected from their evaluation scores. "
        << "These ablations were added after the initial guarded run; guarded_initial_summary.md preserves "
        << "the earlier report when one existed in this output folder.\n\n";
    summary << "| D variant vs C | Corrected numbers | Previously correct numbers lost |\n|---|---:|---:|\n";
    for (size_t m = 3; m < stage.size(); ++m)
        summary << "| " << title[m] << " | " << dFixes[m] << " | " << dRegressions[m] << " |\n";
    summary << "\nGuarded D: probe +1..4 pixels. Admit coloured connected extensions reaching the original boundary, excluding "
        << "cloth-hue pixels and other detected disks. Preserve the entire original disk. A component needs >=3 "
        << "boundary pixels and >=max(12,2% original area) new pixels. Use the first padding confirmed by the next "
        << "consecutive padding with the same valid non-cue type and body colour (choose the confirming padding). "
        << "Otherwise keep original evidence. D recomputes K2 type/histogram evidence only for accepted expansions "
        << "and reruns global assignment; non-expanded balls can change indirectly through uniqueness.\n\n"
        << "There is no size weighting, per-image exception or ground-truth padding selection. Cloth hue is "
        << "estimated independently per image, without annotations. Green-ball pixels can resemble cloth; "
        << "neighbour exclusions can hide real body colour and undetected neighbours can leak into growth. "
        << "The few-pixel search may not reach the actual coloured body. These are limits of this specific rule.\n\n"
        << "This is an evaluation-informed follow-up to observed cue/stripe errors. Parameters were fixed before "
        << "this run and not adjusted from its results. AllBalls is reported separately as training resubstitution. "
        << "Detection counts, radii and coordinates stay unchanged; a larger recognition mask does not claim "
        << "improved localisation. The main application and previous experiment outputs are unchanged.\n\n"
        << "Outputs: white_expansion.csv (guarded paddings), raw_padding.csv (all raw paddings), applied_expansions.csv, cloth_models.csv, "
        << "d_evidence.csv, predictions.csv, changes.csv, metrics.csv, frozen_models.yml, configuration.txt; "
        << "white_expansion/<image>/candidate_N.png (guarded/raw padding crops and masks); <image>/candidate_N.png "
        << "(A–D comparison); final_results/<stage>/<image>/ (standard four PNGs and balls.csv).\n";
    std::cout << "Saved white-expansion experiment to " << directory << '\n';
}
void runColourScoreDiagnostics(const std::filesystem::path& directory) {
    struct Setting {std::string name; int padding, minimumPixels; double cueWeight, width;};
    // Diagnostic ablations, declared together before running any evaluation. No winner is deployed.
    const std::vector<Setting> settings = {
        {"unpadded_control", 0, 12, 1, .10}, {"pad1_control", 1, 12, 1, .10},
        {"pad4_control", 4, 12, 1, .10}, {"pad1_min6", 1, 6, 1, .10},
        {"pad1_cue075", 1, 12, .75, .10}, {"pad1_cue050", 1, 12, .50, .10},
        {"pad1_width015", 1, 12, 1, .15}, {"pad1_min6_cue075", 1, 6, .75, .10},
        {"pad1_min6_width015", 1, 6, 1, .15}, {"pad1_min6_cue075_width015", 1, 6, .75, .15}
    };
    const std::filesystem::path source = "output/experiments/ball_colour_histograms/training/models.yml";
    const auto frozen = loadModel(source);
    std::filesystem::create_directories(directory);
    std::filesystem::copy_file(source, directory / "frozen_models.yml", std::filesystem::copy_options::overwrite_existing);
    auto configuration = output(directory / "configuration.csv");
    configuration << "method,whitePadding,minimumQueryBodyPixels,cueWeight,typeWidth\n";
    for (const auto& s : settings) configuration << s.name << ',' << s.padding << ',' << s.minimumPixels << ',' << s.cueWeight << ',' << s.width << '\n';
    auto evidence = output(directory / "evidence.csv");
    evidence << "image,method,candidate,number,originalWhite,padding,lightFraction,bodyPixels,bodyColour,colourSupport,typeCompatibility,cueWeight,identityScore,rank,eligible,selected\n";
    auto predictions = output(directory / "predictions.csv");
    predictions << "image,method,candidate,trueNumber,number,type,localBest,localScore,selectedScore,cueScore,bestNumberedScore,bodyPixels,bodyColour,lightPercent,originalWhite,directExpansion\n";
    auto metrics = output(directory / "metrics.csv");
    metrics << "split,image,method,groundTruth,detections,TP,FP,FN,precision,recall,F1,meanError,correctTypes,correctNumbers,typeAccuracy,numberAccuracy,numberRecall,numberPrecision,unknownPredictions,extraDuplicateClaims\n";
    auto changes = output(directory / "changes.csv");
    changes << "image,method,reference,candidate,trueNumber,beforeNumber,afterNumber,change,originalWhite\n";
    auto forced = output(directory / "cue_counterfactuals.csv");
    forced << "image,method,forcedCueCandidate,trueNumber,totalScore,lossAgainstFreeAssignment,changedAssignments,assignments\n";
    auto swaps = output(directory / "cue_swaps.csv");
    swaps << "image,method,currentCueCandidate,otherCandidate,currentPairScore,swappedPairScore,swapGain,alternativeForCurrentCue,otherCurrentNumber\n";
    std::vector<Counts> totals(settings.size());
    std::vector<int> fixes(settings.size()), regressions(settings.size());
    std::vector<int> fixesVsPad(settings.size()), regressionsVsPad(settings.size());
    std::vector<std::string> table1Findings;
    auto totalScore = [](const std::vector<Candidate>& candidates, const std::vector<int>& assignments) {
        double total = 0;
        for (size_t i = 0; i < candidates.size(); ++i)
            if (assignments[i] >= 0) total += candidates[i].identityScore[assignments[i]];
        return total;
    };
    for (const auto& filename : config::allImages()) {
        const std::string stem = std::filesystem::path(filename).stem().string();
        const bool evaluation = filename != "AllBalls.jpeg";
        const auto image = cv::imread(config::mediaDirectory + filename);
        if (image.empty()) throw std::runtime_error("Could not load " + filename);
        const auto circles = runDetection(image, HoughParams{}).filteredCircles;
        const auto centres = transformCentres(circles);
        std::vector<Candidate> original;
        for (size_t i = 0; i < circles.size(); ++i) {
            Candidate c;
            c.circle = circles[i]; c.tableCentre = centres[i]; c.roi = extractBall(image, circles[i]);
            c.baselineType = classifyBallType(segmentBallKMeans(c.roi, 2));
            c.colours = scoreBallColours(c.roi, frozen);
            identityEvidence(c, true);
            original.push_back(std::move(c));
        }
        std::vector<std::vector<Candidate>> variants;
        std::vector<std::vector<int>> numbers;
        for (const auto& s : settings) {
            auto candidates = original;
            auto queryModel = frozen;
            queryModel.options.minimumBodyPixels = s.minimumPixels;
            for (auto& c : candidates) {
                if (c.baselineType.type == "White" && s.padding) c = rawWhitePadding(image, c, queryModel, s.padding);
                else if (s.minimumPixels != 12) c.colours = scoreBallColours(c.roi, queryModel);
                const double light = std::clamp(c.baselineType.lightPercent / 100, 0.0, 1.0);
                const auto compat = [&](double d) {return std::exp(-.5 * std::pow(d / s.width, 2));};
                c.typeSupport = {compat(std::max(0.0, .85 - light)), compat(std::max(0.0, light - .30)),
                    compat(std::max({0.0, .30 - light, light - .85}))};
                c.identityScore[0] = s.cueWeight * light * c.typeSupport[0];
                for (int n = 1; n <= 15; ++n) c.identityScore[n] = c.colours.bestColour < 0 ? 0 :
                    c.colours.similarity[(n <= 8 ? n : n - 8) - 1] * c.typeSupport[n <= 8 ? 1 : 2];
                const auto best = std::max_element(c.identityScore.begin(), c.identityScore.end());
                c.localBest = int(best - c.identityScore.begin()); c.bestScore = *best;
                for (int n = 0; n <= 15; ++n) c.eligible[n] = hasUsableEvidence(c) && c.identityScore[n] > 1e-9;
            }
            numbers.push_back(assignUnique(candidates, false, true));
            variants.push_back(std::move(candidates));
        }
        // Every prediction/assignment is fixed before any evaluation annotation is read.
        const auto truth = loadLabelledGroundTruth(config::groundTruthPath, filename);
        if (truth.empty()) throw std::runtime_error("Missing annotations for " + filename);
        std::vector<cv::Point2f> truthCentres;
        for (const auto& t : truth) truthCentres.push_back(t.centre);
        const auto e = evaluateDetections(centres, truthCentres, config::matchTolerance);
        auto actualFor = [&](size_t i) {const int j = e.predictionToGroundTruth[i]; return j < 0 ? -1 : truth[j].number;};
        for (size_t m = 0; m < settings.size(); ++m) {
            const auto& s = settings[m]; const auto& candidates = variants[m]; const auto& result = numbers[m];
            const double freeScore = totalScore(candidates, result);
            std::vector<std::string> types;
            Counts count{int(truth.size()), e.TP, e.FP, e.FN, 0, 0, 0, 0, e.meanError * e.TP};
            std::array<bool, 16> used{};
            int cueRow = -1;
            for (size_t i = 0; i < candidates.size(); ++i) {
                const auto& c = candidates[i]; const int n = result[i], actual = actualFor(i);
                types.push_back(typeFor(n));
                if (n < 0) ++count.unknown;
                else {if (used[n]) throw std::runtime_error("Duplicate identity in diagnostic"); used[n] = true;}
                if (n == 0) cueRow = int(i);
                if (actual >= 0) {count.type += typeFor(n) == typeFor(actual); count.number += n == actual;}
                const bool white = original[i].baselineType.type == "White";
                predictions << filename << ',' << s.name << ',' << i + 1 << ',' << actual << ',' << n << ',' << typeFor(n) << ','
                    << c.localBest << ',' << c.bestScore << ',' << (n < 0 ? 0 : c.identityScore[n]) << ',' << c.identityScore[0] << ','
                    << *std::max_element(c.identityScore.begin() + 1, c.identityScore.end()) << ',' << c.colours.pixels << ','
                    << c.colours.bestColour << ',' << c.baselineType.lightPercent << ',' << white << ',' << (white && s.padding > 0) << '\n';
                for (int identity = 0; identity <= 15; ++identity) {
                    const int rank = 1 + int(std::count_if(c.identityScore.begin(), c.identityScore.end(), [&](double v) {return v > c.identityScore[identity];}));
                    const double colour = identity == 0 ? c.baselineType.lightPercent / 100 :
                        c.colours.similarity[(identity <= 8 ? identity : identity - 8) - 1];
                    evidence << filename << ',' << s.name << ',' << i + 1 << ',' << identity << ',' << white << ',' << (white ? s.padding : 0)
                        << ',' << c.baselineType.lightPercent / 100 << ',' << c.colours.pixels << ',' << c.colours.bestColour << ',' << colour << ','
                        << c.typeSupport[identity == 0 ? 0 : identity <= 8 ? 1 : 2] << ',' << (identity == 0 ? s.cueWeight : 1) << ','
                        << c.identityScore[identity] << ',' << rank << ',' << c.eligible[identity] << ',' << (n == identity) << '\n';
                }
                for (const int ref : {0, 1}) {
                    const auto outcome = change(numbers[ref][i], n, actual);
                    changes << filename << ',' << s.name << ',' << settings[ref].name << ',' << i + 1 << ',' << actual << ','
                        << numbers[ref][i] << ',' << n << ',' << outcome << ',' << white << '\n';
                    if (evaluation) {
                        (ref == 0 ? fixes : fixesVsPad)[m] += outcome == "corrected";
                        (ref == 0 ? regressions : regressionsVsPad)[m] += outcome == "regressed";
                    }
                }
                // Diagnostic constraints force each original White candidate to claim cue; never used for inference.
                if (white && c.eligible[0]) {
                    auto constrained = candidates;
                    for (size_t j = 0; j < constrained.size(); ++j) {
                        if (j != i) constrained[j].eligible[0] = false;
                        else {constrained[j].eligible.fill(false); constrained[j].eligible[0] = true;}
                    }
                    auto assignment = assignUnique(constrained, false, true);
                    if (assignment[i] != 0) throw std::runtime_error("Forced cue constraint failed");
                    const double score = totalScore(candidates, assignment);
                    if (score > freeScore + 1e-7) throw std::runtime_error("Constrained score exceeds free optimum");
                    int changed = 0;
                    std::ostringstream mapping;
                    for (size_t j = 0; j < assignment.size(); ++j) {changed += assignment[j] != result[j]; if (j) mapping << ';'; mapping << j + 1 << ':' << assignment[j];}
                    forced << filename << ',' << s.name << ',' << i + 1 << ',' << actual << ',' << score << ','
                        << freeScore - score << ',' << changed << ',' << mapping.str() << '\n';
                }
            }
            if (cueRow >= 0) for (size_t other = 0; other < candidates.size(); ++other) {
                const int alternative = result[other];
                if (int(other) == cueRow || alternative < 1 || !candidates[other].eligible[0] || !candidates[cueRow].eligible[alternative]) continue;
                const double before = candidates[cueRow].identityScore[0] + candidates[other].identityScore[alternative];
                const double after = candidates[cueRow].identityScore[alternative] + candidates[other].identityScore[0];
                swaps << filename << ',' << s.name << ',' << cueRow + 1 << ',' << other + 1 << ',' << before << ',' << after << ',' << after - before << ',' << alternative << ',' << result[other] << '\n';
            }
            saveFinalResults(image, directory / "final_results" / s.name / stem, circles, centres, types, result);
            metrics << (evaluation ? "evaluation" : "training_resubstitution") << ',' << filename << ',' << s.name << ',';
            writeCounts(metrics, count); metrics << '\n';
            if (evaluation) totals[m].add(count);
            if (filename == "Table1.jpeg") {
                std::ostringstream finding;
                finding << "| " << s.name << " |";
                for (size_t i = 0; i < candidates.size(); ++i) if (original[i].baselineType.type == "White")
                    finding << " #" << i + 1 << " actual " << label(actualFor(i)) << " -> " << label(result[i]) << ";";
                finding << " | " << count.number << '/' << count.tp << " |\n";
                table1Findings.push_back(finding.str());
            }
        }
        for (size_t i = 0; i < original.size(); ++i) {
            bool changed = false;
            for (const auto& result : numbers) changed |= result[i] != numbers[0][i];
            if (original[i].baselineType.type != "White" && !changed) continue;
            std::vector<ImagePanel> panels;
            for (size_t m = 0; m < settings.size(); ++m) {
                const auto& c = variants[m][i];
                cv::Mat tile(380, 400, CV_8UC3, cv::Scalar(28, 28, 28)), crop, mask;
                cv::resize(c.roi.image, crop, {180, 180}, 0, 0, cv::INTER_NEAREST);
                cv::resize(c.colours.bodyMask, mask, {180, 180}, 0, 0, cv::INTER_NEAREST);
                cv::cvtColor(mask, mask, cv::COLOR_GRAY2BGR);
                crop.copyTo(tile(cv::Rect(10, 10, 180, 180))); mask.copyTo(tile(cv::Rect(210, 10, 180, 180)));
                const int actual = actualFor(i), n = numbers[m][i];
                const cv::Scalar ink = actual < 0 ? cv::Scalar(210,210,210) : n == actual ? cv::Scalar(70,240,70) : cv::Scalar(70,70,255);
                auto line = [&](const std::string& text, int y, cv::Scalar colour = cv::Scalar(225,225,225)) {
                    cv::putText(tile, text, {10,y}, cv::FONT_HERSHEY_SIMPLEX, .45, colour, 1, cv::LINE_AA);
                };
                line("Assigned " + label(n) + " | actual " + label(actual), 212, ink);
                std::ostringstream detail; detail << std::fixed << std::setprecision(1) << "Light " << c.baselineType.lightPercent << "% | body " << c.colours.pixels << "px";
                line(detail.str(), 233);
                std::array<int,16> order{}; std::iota(order.begin(), order.end(), 0);
                std::stable_sort(order.begin(), order.end(), [&](int a, int b) {return c.identityScore[a] > c.identityScore[b];});
                for (int k = 0; k < 5; ++k) {
                    const int identity = order[k]; std::ostringstream row;
                    row << label(identity) << "  " << std::fixed << std::setprecision(4) << c.identityScore[identity];
                    line(row.str(), 257 + 22*k, identity == n ? ink : cv::Scalar(210,210,210));
                    cv::rectangle(tile, cv::Rect(160, 246 + 22*k, std::max(1, cvRound(220*c.identityScore[identity])), 10), {100,150,210}, cv::FILLED);
                }
                panels.push_back({settings[m].name, tile});
            }
            saveImage(directory / "candidate_diagnostics" / stem / ("candidate_" + std::to_string(i+1) + ".png"), combineImages(panels, 2));
        }
        std::cout << filename << ": colour-score diagnostic ablations complete\n";
    }
    auto summary = output(directory / "summary.md");
    summary << "# Colour-score weighting diagnostics\n\n"
        << "Fixed K2 and frozen AllBalls HSV histograms; recognition-only changes. All ten ablations are reported. "
        << "Only original White candidates receive padding. Query minimum affects every candidate, while cue/type weights apply globally. "
        << "Training histograms, light thresholds (V170/S165), erosion3, detections and coordinates stay fixed.\n\n"
        << "| Method | Correct types | Correct identities | Unknown | Fixes / losses vs unpadded | Fixes / losses vs +1 |\n|---|---:|---:|---:|---:|---:|\n";
    for (size_t m = 0; m < settings.size(); ++m) {
        metrics << "evaluation,ALL," << settings[m].name << ','; writeCounts(metrics, totals[m]); metrics << '\n';
        summary << "| " << settings[m].name << " | " << totals[m].type << '/' << totals[m].tp << " | " << totals[m].number << '/' << totals[m].tp
            << " | " << totals[m].unknown << " | " << fixes[m] << " / " << regressions[m] << " | " << fixesVsPad[m] << " / " << regressionsVsPad[m] << " |\n";
    }
    summary << "\n## Table1 original White candidates\n\n| Method | Candidate identities | All matched correct |\n|---|---|---:|\n";
    for (const auto& finding : table1Findings) summary << finding;
    summary << "\nCue score = cueWeight * lightFraction * Gaussian compatibility with [.85,1]. "
        << "Numbered score = frozen histogram similarity * Gaussian compatibility with solid [0,.30] or stripe [.30,.85]. "
        << "Width .10/.15 controls compatibility outside those intervals. Histogram similarity and light fraction "
        << "have different empirical distributions; neither their product nor the final scores are calibrated probabilities.\n\n"
        << "Lowering the minimum from12 to6 only admits tiny query samples; no masks/model bins or training samples change. "
        << "Such evidence may be cloth, shadows, a neighbour or real body colour. Cue weights .75/.50 rescale cue support "
        << "for every candidate; uniqueness allows at most one cue, so this does not automatically resolve which candidate deserves it.\n\n"
        << "cue_counterfactuals.csv forces each original White candidate to claim cue in turn, reruns the full assignment, "
        << "and measures total objective loss. This is a diagnostic, never a prediction rule. cue_swaps.csv gives feasible "
        << "two-candidate cue swaps; negative swapGain explains why the current objective prefers the existing allocation. "
        << "evidence.csv decomposes every identity score and rank. Predictions are fixed before evaluation annotations are loaded.\n\n"
        << "This is an evaluation-informed investigation of observed Table1 failures, including an explicit parameter comparison. "
        << "Training models use only AllBalls, but choosing a winning configuration using Table1–6 is test-set method/parameter selection. "
        << "The reported figures are exploratory, not an independent validation or a recommended deployed setting. "
        << "AllBalls is training resubstitution and excluded from totals. Main pipeline unchanged.\n\n"
        << "Outputs: configuration.csv, metrics.csv, predictions.csv, changes.csv, evidence.csv, cue_counterfactuals.csv, "
        << "cue_swaps.csv, frozen_models.yml, candidate_diagnostics/<image>/candidate_N.png, final_results/<method>/<image>/.\n";
    std::cout << "Saved colour-score diagnostics to " << directory << '\n';
}
void runRowCueSizeExperiment(const std::filesystem::path& directory) {
    checkRowCueSize();
    const std::array<std::string,6> stages = {"C_unique", "white_pad1_unique", "row_cue_unique", "pad1_row_cue_unique",
        "row_largest_cue", "pad1_row_largest_cue"};
    std::filesystem::create_directories(directory);
    const std::filesystem::path source = "output/experiments/ball_colour_histograms/training/models.yml";
    const auto colourModel = loadModel(source);
    std::filesystem::copy_file(source, directory / "frozen_models.yml", std::filesystem::copy_options::overwrite_existing);
    // Fit once from AllBalls Hough detections alone; no annotations or test samples.
    const auto training = cv::imread(config::mediaDirectory + "AllBalls.jpeg");
    if (training.empty()) throw std::runtime_error("Missing AllBalls for row fit");
    const auto trainingCircles = runDetection(training,HoughParams{}).filteredCircles;
    const auto model = fitRowRadius(trainingCircles);
    auto samples = output(directory / "training_samples.csv");
    samples << "candidate,x,row,radius,expectedRadius,residual\n";
    double absoluteResidual = 0;
    for (size_t i=0; i<trainingCircles.size(); ++i) {
        const auto& c=trainingCircles[i]; const double expected=model.expected(c[1]);
        absoluteResidual += std::abs(c[2]-expected);
        samples << i+1 << ',' << c[0] << ',' << c[1] << ',' << c[2] << ',' << expected << ',' << c[2]-expected << '\n';
    }
    samples.close();
    cv::FileStorage fit((directory / "row_model.yml").string(),cv::FileStorage::WRITE);
    if (!fit.isOpened()) throw std::runtime_error("Could not save row model");
    fit << "trainingImage" << "AllBalls.jpeg" << "sampleCount" << int(trainingCircles.size())
        << "intercept" << model.intercept << "slope" << model.slope << "minimumTrainingRow" << model.minimumRow
        << "maximumTrainingRow" << model.maximumRow << "negativeSlopeClamped" << int(model.slopeClamped)
        << "meanAbsoluteTrainingResidual" << absoluteResidual/trainingCircles.size();
    fit.release();
    auto curve = output(directory / "row_radius.csv");
    curve << "row,expectedRadius,extrapolated\n";
    for (int row=0; row<training.rows; ++row)
        curve << row << ',' << model.expected(row) << ',' << (row<model.minimumRow || row>model.maximumRow) << '\n';
    curve.close();
    cv::Mat plot(430,900,CV_8UC3,cv::Scalar(250,250,250));
    const double maximumRadius=std::max(40.0,model.expected(training.rows-1)+5);
    auto point = [&](double row,double radius) {return cv::Point(60+cvRound(800*row/(training.rows-1)),370-cvRound(300*radius/maximumRadius));};
    cv::rectangle(plot,cv::Rect(point(model.minimumRow,0).x,60,std::max(1,point(model.maximumRow,0).x-point(model.minimumRow,0).x),310),{235,245,235},cv::FILLED);
    cv::line(plot,{60,370},{860,370},{60,60,60},1); cv::line(plot,{60,370},{60,60},{60,60,60},1);
    cv::line(plot,point(0,model.expected(0)),point(training.rows-1,model.expected(training.rows-1)),{200,80,30},2,cv::LINE_AA);
    for (const auto& c : trainingCircles) cv::circle(plot,point(c[1],c[2]),4,{30,120,30},cv::FILLED,cv::LINE_AA);
    for (int row=0; row<training.rows; row+=100) cv::putText(plot,std::to_string(row),{point(row,0).x-10,393},cv::FONT_HERSHEY_SIMPLEX,.4,{40,40,40},1);
    for (int radius=0; radius<=int(maximumRadius); radius+=10) cv::putText(plot,std::to_string(radius),{20,point(0,radius).y+4},cv::FONT_HERSHEY_SIMPLEX,.4,{40,40,40},1);
    cv::putText(plot,"Radius (pixels) vs original-image row | green dots: AllBalls Hough samples",{25,25},cv::FONT_HERSHEY_SIMPLEX,.53,{25,25,25},1,cv::LINE_AA);
    cv::putText(plot,"Shaded rows: training support. Other rows use linear extrapolation.",{65,419},cv::FONT_HERSHEY_SIMPLEX,.45,{25,25,25},1,cv::LINE_AA);
    saveImage(directory / "row_radius_model.png",plot);
    auto configuration = output(directory / "configuration.txt");
    configuration << "Row-only size prior, fitted once from pocket-filtered AllBalls Hough circles without labels.\n"
        << "Theil-Sen slope: median pairwise dr/drow for pairs >=10 row pixels apart; slope floored at0. "
        << "Intercept: median(radius-slope*row). Expected radius=max(1,intercept+slope*row).\n"
        << "Model: radius=" << model.intercept << "+" << model.slope << "*row; training rows " << model.minimumRow << ".." << model.maximumRow << ".\n"
        << "Only images with >=2 original K2 White candidates activate size scoring. "
        << "On those images multiply every candidate's cue score only by min(1,originalHoughRadius/expectedRadius)^2. "
        << "This is an apparent area support heuristic, not a calibrated probability. All numbered scores stay unchanged.\n"
        << "Factor applies to every candidate to avoid moving cue claims to a non-White fallback. "
        << "No bonus for oversize circles. Pad1 profile expands original White ROIs only; size always uses original geometry.\n"
        << "Largest-cue ablations instead reserve cue for the original White candidate with greatest radius/expectedRadius, "
        << "and exclude every other cue claim; active only with>=2 original White candidates. This is an explicitly forced heuristic.\n"
        << "All six variants use unchanged K2, frozen AllBalls HSV histograms, type width.10, best-effort unique assignment. "
        << "No candidates removed, no stored centres/radii/table coordinates changed. No chamfer combination.\n"
        << "All settings fixed before evaluation; experiment follows observed Table1-6 errors, so results remain exploratory.\n";
    configuration.close();
    auto synthetic = output(directory / "synthetic_checks.txt");
    synthetic << "PASS: robust linear fit with a severe radius outlier; equal-row constant fallback; negative slope prior; "
        << "size factor monotonic/capped; targeted cue conflict switches correctly while preserving stripe alternatives.\n";
    synthetic.close();
    auto evidence = output(directory / "cue_size_scores.csv");
    evidence << "image,method,candidate,originalWhite,whiteCandidateCount,priorActive,row,originalRadius,expectedRadius,radiusRatio,extrapolated,sizeSupport,appliedFactor,originalCueScore,weightedCueScore,selectedNumber\n";
    auto predictions = output(directory / "predictions.csv");
    predictions << "image,method,candidate,trueNumber,number,type,originalWhite,localBest,selectedScore,bodyPixels,lightPercent\n";
    auto metrics = output(directory / "metrics.csv");
    metrics << "split,image,method,groundTruth,detections,TP,FP,FN,precision,recall,F1,meanError,correctTypes,correctNumbers,typeAccuracy,numberAccuracy,numberRecall,numberPrecision,unknownPredictions,extraDuplicateClaims\n";
    auto changes = output(directory / "changes.csv");
    changes << "image,method,reference,candidate,trueNumber,beforeNumber,afterNumber,change\n";
    std::array<Counts,6> totals{};
    std::array<int,6> fixes{},losses{},padFixes{},padLosses{};
    std::vector<std::string> table1;
    int activeImages=0;
    for (const auto& filename : config::allImages()) {
        const bool evaluation=filename!="AllBalls.jpeg";
        const std::string stem=std::filesystem::path(filename).stem().string();
        const auto image=cv::imread(config::mediaDirectory+filename);
        if (image.empty()) throw std::runtime_error("Could not load "+filename);
        const auto circles=runDetection(image,HoughParams{}).filteredCircles;
        const auto centres=transformCentres(circles);
        std::vector<Candidate> original,padded;
        int whites=0;
        for (size_t i=0; i<circles.size(); ++i) {
            Candidate c; c.circle=circles[i]; c.tableCentre=centres[i]; c.roi=extractBall(image,c.circle);
            c.baselineType=classifyBallType(segmentBallKMeans(c.roi,2)); c.colours=scoreBallColours(c.roi,colourModel);
            identityEvidence(c,true); original.push_back(c);
            const bool white=c.baselineType.type=="White";
            whites+=white;
            padded.push_back(white ? rawWhitePadding(image,c,colourModel,1) : c);
        }
        const bool active=whites>=2;
        if (evaluation) activeImages+=active;
        std::array<std::vector<Candidate>,6> variants={original,padded,original,padded,original,padded};
        for (int m : {2,3}) for (auto& c : variants[m]) {
            if (active) c.identityScore[0] *= cueSizeSupport(c.circle[2],model.expected(c.circle[1]));
            const auto best=std::max_element(c.identityScore.begin(),c.identityScore.end());
            c.localBest=int(best-c.identityScore.begin()); c.bestScore=*best;
            c.eligible[0]=hasUsableEvidence(c) && c.identityScore[0]>1e-9;
        }
        int largest=-1;
        double largestRatio=-1;
        for (size_t i=0;i<original.size();++i) if (original[i].baselineType.type=="White") {
            const double r=original[i].circle[2]/model.expected(original[i].circle[1]);
            if (r>largestRatio) {largest=int(i); largestRatio=r;}
        }
        if (active) for (const int m : {4,5}) for (size_t i=0;i<original.size();++i) {
            auto& c=variants[m][i];
            if (int(i)==largest) {c.eligible.fill(false); c.eligible[0]=true;}
            else {c.eligible[0]=false; c.identityScore[0]=0;}
            const auto best=std::max_element(c.identityScore.begin(),c.identityScore.end());
            c.localBest=int(best-c.identityScore.begin()); c.bestScore=*best;
        }
        std::array<std::vector<int>,6> numbers;
        for (size_t m=0; m<stages.size(); ++m) numbers[m]=assignUnique(variants[m],false,true);
        if (active && (numbers[4][largest]!=0 || numbers[5][largest]!=0)) throw std::runtime_error("Largest-cue reservation failed");
        // Evaluation-only annotations: the fit, activation and all assignments are already fixed.
        const auto truth=loadLabelledGroundTruth(config::groundTruthPath,filename);
        if (truth.empty()) throw std::runtime_error("Missing annotations for "+filename);
        std::vector<cv::Point2f> truthCentres;
        for (const auto& t : truth) truthCentres.push_back(t.centre);
        const auto e=evaluateDetections(centres,truthCentres,config::matchTolerance);
        auto actualFor=[&](size_t i) {const int j=e.predictionToGroundTruth[i]; return j<0 ? -1 : truth[j].number;};
        for (size_t m=0; m<stages.size(); ++m) {
            Counts count{int(truth.size()),e.TP,e.FP,e.FN,0,0,0,0,e.meanError*e.TP};
            std::vector<std::string> types;
            std::array<bool,16> used{};
            for (size_t i=0; i<circles.size(); ++i) {
                const auto& c=variants[m][i]; const int n=numbers[m][i],actual=actualFor(i);
                types.push_back(typeFor(n));
                if (n<0) ++count.unknown;
                else {if (used[n]) throw std::runtime_error("Duplicate identity in row-cue assignment"); used[n]=true;}
                if (actual>=0) {count.type+=typeFor(n)==typeFor(actual); count.number+=n==actual;}
                const double expected=model.expected(c.circle[1]), support=cueSizeSupport(c.circle[2],expected);
                const double factor=active ? (m==2 || m==3 ? support : m>=4 && int(i)!=largest ? 0 : 1) : 1;
                evidence << filename << ',' << stages[m] << ',' << i+1 << ',' << (original[i].baselineType.type=="White") << ',' << whites
                    << ',' << (m>=2 && active) << ',' << c.circle[1] << ',' << c.circle[2] << ',' << expected << ',' << c.circle[2]/expected
                    << ',' << (c.circle[1]<model.minimumRow || c.circle[1]>model.maximumRow) << ',' << support << ',' << factor << ','
                    << variants[m%2][i].identityScore[0] << ',' << c.identityScore[0] << ',' << n << '\n';
                predictions << filename << ',' << stages[m] << ',' << i+1 << ',' << actual << ',' << n << ',' << typeFor(n) << ','
                    << (original[i].baselineType.type=="White") << ',' << c.localBest << ',' << (n<0 ? 0 : c.identityScore[n]) << ','
                    << c.colours.pixels << ',' << c.baselineType.lightPercent << '\n';
                for (const int ref : {0,1}) {
                    const auto outcome=change(numbers[ref][i],n,actual);
                    changes << filename << ',' << stages[m] << ',' << stages[ref] << ',' << i+1 << ',' << actual << ',' << numbers[ref][i] << ',' << n << ',' << outcome << '\n';
                    if (evaluation) {(ref==0 ? fixes : padFixes)[m]+=outcome=="corrected"; (ref==0 ? losses : padLosses)[m]+=outcome=="regressed";}
                }
            }
            saveFinalResults(image,directory / "final_results" / stages[m] / stem,circles,centres,types,numbers[m]);
            metrics << (evaluation ? "evaluation" : "training_resubstitution") << ',' << filename << ',' << stages[m] << ',';
            writeCounts(metrics,count); metrics << '\n';
            if (evaluation) totals[m].add(count);
            if (filename=="Table1.jpeg") {
                std::ostringstream row; row << "| " << stages[m] << " | ";
                for (size_t i=0;i<circles.size();++i) if (original[i].baselineType.type=="White")
                    row << '#' << i+1 << " actual " << label(actualFor(i)) << " -> " << label(numbers[m][i]) << "; ";
                row << "| " << count.number << '/' << count.tp << " |\n"; table1.push_back(row.str());
            }
        }
        for (size_t i=0;i<circles.size();++i) {
            bool changed=false; for (const auto& ns : numbers) changed|=ns[i]!=numbers[0][i];
            if (!changed && original[i].baselineType.type!="White") continue;
            std::vector<ImagePanel> panels;
            for (size_t m=0;m<stages.size();++m) {
                const auto& c=variants[m][i]; const int n=numbers[m][i],actual=actualFor(i);
                cv::Mat tile(330,320,CV_8UC3,cv::Scalar(30,30,30)),crop;
                cv::resize(c.roi.image,crop,{230,230},0,0,cv::INTER_NEAREST); crop.copyTo(tile(cv::Rect(45,0,230,230)));
                const auto ink=actual<0 ? cv::Scalar(220,220,220) : n==actual ? cv::Scalar(60,240,60) : cv::Scalar(60,60,255);
                auto line=[&](const std::string& text,int y,cv::Scalar colour) {cv::putText(tile,text,{8,y},cv::FONT_HERSHEY_SIMPLEX,.43,colour,1,cv::LINE_AA);};
                line("Actual "+label(actual)+" | assigned "+label(n),250,ink);
                std::ostringstream size; size << std::fixed << std::setprecision(2) << "Radius " << c.circle[2] << " / expected " << model.expected(c.circle[1]);
                line(size.str(),273,{220,220,220});
                std::ostringstream cue; cue << std::fixed << std::setprecision(3) << "Cue " << variants[m%2][i].identityScore[0] << " -> " << c.identityScore[0];
                line(cue.str(),296,{220,220,220});
                line(std::string("Ambiguous-cue size prior: ")+(m>=2 && active ? "ON" : "OFF"),319,{220,220,220});
                panels.push_back({stages[m],tile});
            }
            saveImage(directory / "candidate_diagnostics" / stem / ("candidate_"+std::to_string(i+1)+".png"),combineImages(panels,2));
        }
        std::cout << filename << ": row-cue prior " << (active ? "active" : "inactive") << " (" << whites << " original White candidates)\n";
    }
    auto summary=output(directory / "summary.md");
    summary << "# Row-based size support for competing cue claims\n\n"
        << "One row-only radius model is fitted from " << trainingCircles.size() << " AllBalls Hough detections, without labels: "
        << "expectedRadius=max(1," << model.intercept << "+" << model.slope << "*originalImageRow). "
        << "Robust Theil-Sen slope uses pairs >=10 rows apart, floored at0; intercept is the median residual. "
        << "Every horizontal position uses the same expected radius at a given row.\n\n"
        << "The training row span is " << model.minimumRow << ".." << model.maximumRow << "; rows outside that span use "
        << "linear extrapolation, not interpolation. Hough radius bias/partial crops can bias the fit. Horizontal perspective "
        << "variation and sphere height are ignored. The mean absolute training residual is " << absoluteResidual/trainingCircles.size() << " pixels.\n\n"
        << "Size scoring activates only with >=2 original K2 White candidates (" << activeImages << " evaluation images). "
        << "Then every candidate's cue score is multiplied by min(1,originalRadius/expectedRadius)^2, an apparent-area "
        << "support heuristic. Numbered scores are unchanged. Applying this to all cue claims avoids merely handing "
        << "cue to another soft non-White candidate. Oversized circles get no bonus. No thresholds were selected from this run.\n\n"
        << "| Method | Correct types | Correct identities | Unknown | Fixes/losses vs C | Fixes/losses vs +1 |\n|---|---:|---:|---:|---:|---:|\n";
    for (size_t m=0;m<stages.size();++m) {
        metrics << "evaluation,ALL," << stages[m] << ','; writeCounts(metrics,totals[m]); metrics << '\n';
        summary << "| " << stages[m] << " | " << totals[m].type << '/' << totals[m].tp << " | " << totals[m].number << '/' << totals[m].tp
            << " | " << totals[m].unknown << " | " << fixes[m] << '/' << losses[m] << " | " << padFixes[m] << '/' << padLosses[m] << " |\n";
    }
    summary << "\n| Table1 method | Original White candidate identities | All matched correct |\n|---|---|---:|\n";
    for (const auto& row : table1) summary << row;
    summary << "\nThe two largest-cue ablations test the stronger hypothesis directly: reserve cue for the original White "
        << "candidate with greatest radius/expectedRadius and forbid every other cue claim. Ties keep the first candidate. "
        << "The selected candidate is forced to cue; this can be wrong when its detection is inflated or a real cue was not originally White. "
        << "This is compared separately from the soft area-support prior, with no selection from evaluation scores.\n\n"
        << "All methods share original K2, frozen AllBalls HSV histograms and best-effort unique assignment. "
        << "Padding1 expands only originally White recognition ROIs, while size uses original Hough radii/rows. "
        << "Detection geometry, count and matching are fixed. AllBalls results are training resubstitution, excluded from totals. "
        << "The main pipeline is unchanged. These are evaluation-informed exploratory comparisons, not independent validation.\n\n"
        << "Outputs: row_model.yml, training_samples.csv, row_radius.csv, row_radius_model.png, configuration.txt, "
        << "cue_size_scores.csv, predictions.csv, metrics.csv, changes.csv, candidate_diagnostics/<image>/candidate_N.png, "
        << "final_results/<method>/<image>/, frozen_models.yml and synthetic_checks.txt.\n";
    std::cout << "Saved row-cue size experiment to " << directory << '\n';
}
} // namespace experiments
