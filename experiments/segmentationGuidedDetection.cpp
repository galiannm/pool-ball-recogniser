#include "segmentationGuidedDetection.h"
#include "config.h"
#include "detection/ballDetection.h"
#include "detection/geometricFilter.h"
#include "geometry/perspectiveTransform.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace experiments {
namespace {

double median(std::vector<double> values) {
    if (values.empty())
        return 0;
    const size_t middle = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + middle, values.end());
    return values[middle];
}

cv::Point centre(const cv::Vec3f& c) { return {cvRound(c[0]), cvRound(c[1])}; }

// A perspective-aware size guide using only known geometry and configured
// Hough size bounds. No circles or annotations are needed to locate clusters.
double expectedRadius(double y) {
    const auto corners = getTableCorners();
    int top = corners.front().y, bottom = top;
    for (const auto& p : corners) {
        top = std::min(top, p.y);
        bottom = std::max(bottom, p.y);
    }
    const HoughParams hough;
    const double fraction = std::clamp((y - top) / std::max(1, bottom - top), 0.0, 1.0);
    return hough.minRadius + fraction * (hough.maxRadius - hough.minRadius);
}

struct ClothModel {
    cv::Vec3d colour;
    double threshold = 0;
    bool valid = false;
};

ClothModel clothModel(const cv::Mat& hsv, const cv::Mat& lab, const cv::Mat& table,
                      const cv::Rect& crop, const SegmentationHoughOptions& p) {
    std::vector<cv::Vec3b> samples;
    std::vector<double> channels[3];
    for (int y = crop.y; y < crop.br().y; y += 2)
        for (int x = crop.x; x < crop.br().x; x += 2) {
            const auto h = hsv.at<cv::Vec3b>(y, x);
            if (table.at<uchar>(y, x) && h[0] >= 35 && h[0] <= 95 && h[1] >= 35 && h[2] >= 30) {
                const auto pixel = lab.at<cv::Vec3b>(y, x);
                samples.push_back(pixel);
                for (int c = 0; c < 3; ++c)
                    channels[c].push_back(pixel[c]);
            }
        }
    ClothModel model;
    if (static_cast<int>(samples.size()) < p.minimumClothSamples)
        return model;
    model.valid = true;
    model.colour = {median(channels[0]), median(channels[1]), median(channels[2])};
    std::vector<double> distances, deviations;
    for (const auto& pixel : samples)
        distances.push_back(cv::norm(cv::Vec3d(pixel) - model.colour));
    const double middle = median(distances);
    for (double d : distances)
        deviations.push_back(std::abs(d - middle));
    model.threshold = std::clamp(middle + p.clothMADMultiplier * median(deviations),
                                 p.clothMinimumDistance, p.clothMaximumDistance);
    return model;
}

cv::Mat foreground(const cv::Mat& image, const cv::Mat& table, const SegmentationHoughOptions& p) {
    cv::Mat inset, hsv, lab;
    cv::erode(table, inset, cv::getStructuringElement(cv::MORPH_ELLIPSE, {p.tableInsetKernel, p.tableInsetKernel}));
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
    cv::cvtColor(image, lab, cv::COLOR_BGR2Lab);
    const auto fallback = clothModel(hsv, lab, inset, {0, 0, image.cols, image.rows}, p);
    cv::Mat mask = cv::Mat::zeros(image.size(), CV_8UC1);
    if (!fallback.valid)
        return mask; // No usable cloth model: the algorithm falls back to standard Hough everywhere.
    for (int top = 0; top < image.rows; top += p.clothTileSize)
        for (int left = 0; left < image.cols; left += p.clothTileSize) {
            const cv::Rect tile(left, top, std::min(p.clothTileSize, image.cols - left),
                                std::min(p.clothTileSize, image.rows - top));
            auto local = clothModel(hsv, lab, inset, tile, p);
            if (!local.valid)
                local = fallback;
            for (int y = tile.y; y < tile.br().y; ++y)
                for (int x = tile.x; x < tile.br().x; ++x)
                    if (inset.at<uchar>(y, x) &&
                        cv::norm(cv::Vec3d(lab.at<cv::Vec3b>(y, x)) - local.colour) > local.threshold)
                        mask.at<uchar>(y, x) = 255;
        }
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE,
        cv::getStructuringElement(cv::MORPH_ELLIPSE, {p.closingKernel, p.closingKernel}));
    cv::Mat holes;
    cv::copyMakeBorder(mask, holes, 1, 1, 1, 1, cv::BORDER_CONSTANT, cv::Scalar(0));
    cv::floodFill(holes, {0, 0}, cv::Scalar(255));
    cv::bitwise_not(holes, holes);
    cv::bitwise_or(mask, holes(cv::Rect(1, 1, image.cols, image.rows)), mask);
    cv::bitwise_and(mask, inset, mask);
    cv::Mat labels, stats, centres, clean = cv::Mat::zeros(mask.size(), CV_8UC1);
    const int count = cv::connectedComponentsWithStats(mask, labels, stats, centres, 8, CV_32S);
    for (int label = 1; label < count; ++label)
        if (stats.at<int>(label, cv::CC_STAT_AREA) >= p.minimumComponentArea)
            clean.setTo(255, labels == label);
    return clean;
}

std::vector<GuidedCluster> clusterWindows(const cv::Mat& mask, const cv::Mat& table,
                                         const SegmentationHoughOptions& p) {
    cv::Mat labels, stats, centres, inner, border;
    const int count = cv::connectedComponentsWithStats(mask, labels, stats, centres, 8, CV_32S);
    cv::erode(table, inner, cv::getStructuringElement(cv::MORPH_ELLIPSE, {41, 41}));
    cv::subtract(table, inner, border);
    std::vector<GuidedCluster> result;
    const double tableArea = cv::countNonZero(table);
    for (int label = 1; label < count; ++label) {
        const int area = stats.at<int>(label, cv::CC_STAT_AREA);
        const cv::Rect bounds(stats.at<int>(label, cv::CC_STAT_LEFT), stats.at<int>(label, cv::CC_STAT_TOP),
                              stats.at<int>(label, cv::CC_STAT_WIDTH), stats.at<int>(label, cv::CC_STAT_HEIGHT));
        const double radius = expectedRadius(centres.at<double>(label, 1));
        const double ratio = area / (CV_PI * radius * radius);
        const double aspect = double(std::max(bounds.width, bounds.height)) / std::min(bounds.width, bounds.height);
        if (ratio < p.clusterAreaRatio && !(aspect >= p.clusterAspectRatio && ratio >= p.elongatedMinimumAreaRatio))
            continue;
        cv::Mat atBorder;
        cv::bitwise_and(labels == label, border, atBorder);
        if (area > 0.10 * tableArea || (aspect > 4 && double(cv::countNonZero(atBorder)) / area > 0.6))
            continue; // Reject very large background failures and long rail-like strips.
        GuidedCluster region;
        region.core = bounds;
        region.expectedRadius = radius;
        const int padding = cvRound(p.cropPaddingRadii * radius);
        region.crop = cv::Rect(bounds.x - padding, bounds.y - padding,
                               bounds.width + 2 * padding, bounds.height + 2 * padding)
                      & cv::Rect(0, 0, mask.cols, mask.rows);
        result.push_back(region);
    }
    bool merged = true;
    while (merged) {
        merged = false;
        for (size_t i = 0; i < result.size() && !merged; ++i)
            for (size_t j = i + 1; j < result.size(); ++j)
                if (!(result[i].crop & result[j].crop).empty()) {
                    result[i].crop |= result[j].crop;
                    result[i].core |= result[j].core;
                    result[i].expectedRadius = expectedRadius(result[i].core.y + result[i].core.height / 2.0);
                    result.erase(result.begin() + j);
                    merged = true;
                    break;
                }
    }
    return result;
}

void edgeEvidence(GuidedProposal& proposal, const GuidedCluster& region,
                   const cv::Mat& edges, const cv::Mat& gx, const cv::Mat& gy,
                   const cv::Mat& foregroundMask, const SegmentationHoughOptions& p) {
    const auto& c = proposal.circle;
    const double cx = c[0] - region.crop.x, cy = c[1] - region.crop.y, radius = c[2];
    std::vector<bool> supported(p.angularSamples, false);
    const int tolerance = static_cast<int>(std::ceil(p.edgeTolerance));
    for (int i = 0; i < p.angularSamples; ++i) {
        const double angle = 2 * CV_PI * i / p.angularSamples;
        const double dx = std::cos(angle), dy = std::sin(angle);
        const double x = cx + radius * dx, y = cy + radius * dy;
        for (int oy = -tolerance; oy <= tolerance && !supported[i]; ++oy)
            for (int ox = -tolerance; ox <= tolerance; ++ox) {
                const int ex = cvRound(x) + ox, ey = cvRound(y) + oy;
                if (ex < 0 || ey < 0 || ex >= edges.cols || ey >= edges.rows || !edges.at<uchar>(ey, ex) ||
                    std::hypot(ex - x, ey - y) > p.edgeTolerance)
                    continue;
                const double sx = gx.at<float>(ey, ex), sy = gy.at<float>(ey, ex);
                const double strength = std::hypot(sx, sy);
                if (strength > 1 && std::abs(sx * dx + sy * dy) / strength >= p.minimumRadialAlignment) {
                    supported[i] = true;
                    break;
                }
            }
    }
    proposal.edgeSupport = double(std::count(supported.begin(), supported.end(), true)) / p.angularSamples;
    auto closed = supported;
    for (int i = 0; i < p.angularSamples; ++i)
        if (!supported[i] && supported[(i + p.angularSamples - 1) % p.angularSamples] &&
            supported[(i + 1) % p.angularSamples])
            closed[i] = true; // Bridge one missing sample, without changing raw support fraction.
    int longest = 0, run = 0;
    for (int i = 0; i < 2 * p.angularSamples; ++i) {
        run = closed[i % p.angularSamples] ? std::min(p.angularSamples, run + 1) : 0;
        longest = std::max(longest, run);
    }
    proposal.arcDegrees = 360.0 * longest / p.angularSamples;
    cv::Mat disk = cv::Mat::zeros(region.crop.size(), CV_8UC1), support;
    cv::circle(disk, {cvRound(cx), cvRound(cy)}, cvRound(radius), cv::Scalar(255), cv::FILLED);
    cv::bitwise_and(disk, foregroundMask(region.crop), support);
    proposal.foregroundSupport = double(cv::countNonZero(support)) / std::max(1, cv::countNonZero(disk));
    proposal.quality = 0.7 * proposal.edgeSupport + 0.3 * proposal.arcDegrees / 360;
}

bool duplicate(const cv::Vec3f& a, const std::vector<cv::Vec3f>& selected, const SegmentationHoughOptions& p) {
    for (const auto& b : selected)
        if (cv::norm(cv::Point2f(a[0], a[1]) - cv::Point2f(b[0], b[1])) <
            p.duplicateDistanceRadii * std::min(a[2], b[2]))
            return true;
    return false;
}

void watershedCheck(const cv::Mat& image, const cv::Mat& table,
                    SegmentationGuidedResult& result, const SegmentationHoughOptions& p) {
    std::vector<bool> keep(result.guided.size(), true);
    for (size_t r = 0; r < result.clusters.size(); ++r) {
        auto& region = result.clusters[r];
        cv::Mat foreground = result.foreground(region.crop).clone();
        for (size_t i = 0; i < result.guided.size(); ++i)
            if (region.crop.contains(centre(result.guided[i]))) {
                region.seedIndices.push_back(static_cast<int>(i));
                cv::circle(foreground, centre(result.guided[i]) - region.crop.tl(),
                           cvRound(0.5 * result.guided[i][2]), cv::Scalar(255), cv::FILLED);
            }
        if (region.seedIndices.empty())
            continue;
        cv::Mat expanded;
        cv::dilate(foreground, expanded, cv::getStructuringElement(cv::MORPH_ELLIPSE, {7, 7}));
        region.markers = cv::Mat::zeros(foreground.size(), CV_32SC1);
        region.markers.setTo(1, expanded == 0);
        region.markers.setTo(1, table(region.crop) == 0);
        int label = 2;
        for (int i : region.seedIndices)
            cv::circle(region.markers, centre(result.guided[i]) - region.crop.tl(), 3, cv::Scalar(label++), cv::FILLED);
        region.labels = region.markers.clone();
        cv::Mat smooth;
        cv::GaussianBlur(image(region.crop), smooth, {3, 3}, 0.8);
        cv::watershed(smooth, region.labels);
        for (auto& candidate : result.proposals) {
            if (!candidate.accepted || candidate.region != static_cast<int>(r + 1))
                continue;
            const auto index = std::find(region.seedIndices.begin(), region.seedIndices.end(), candidate.guidedIndex);
            if (index == region.seedIndices.end())
                throw std::runtime_error("Missing watershed marker for accepted proposal");
            cv::Mat basin, supported;
            cv::compare(region.labels, static_cast<int>(index - region.seedIndices.begin()) + 2, basin, cv::CMP_EQ);
            cv::bitwise_and(basin, foreground, supported);
            candidate.watershedAreaRatio = cv::countNonZero(supported) / (CV_PI * candidate.circle[2] * candidate.circle[2]);
            candidate.watershedAccepted = candidate.watershedAreaRatio >= p.watershedMinimumAreaRatio &&
                candidate.watershedAreaRatio <= p.watershedMaximumAreaRatio;
            candidate.watershedReason = candidate.watershedAccepted ? "accepted" : "basin_area";
            keep[candidate.guidedIndex] = candidate.watershedAccepted;
        }
    }
    for (size_t i = 0; i < result.guided.size(); ++i)
        if (keep[i])
            result.watershed.push_back(result.guided[i]);
}

} // namespace

SegmentationGuidedResult detectSegmentationGuided(const cv::Mat& image, const SegmentationHoughOptions& p) {
    SegmentationGuidedResult result;
    if (image.empty())
        return result;
    if (image.type() != CV_8UC3 || p.clothTileSize < 1 || p.angularSamples < 12 ||
        p.tableInsetKernel < 1 || p.tableInsetKernel % 2 != 1 || p.closingKernel < 1 || p.closingKernel % 2 != 1 ||
        p.clothMinimumDistance > p.clothMaximumDistance || p.minimumClothSamples < 1)
        throw std::invalid_argument("Invalid segmentation-guided detection image/options");
    cv::Mat table = cv::Mat::zeros(image.size(), CV_8UC1);
    cv::fillConvexPoly(table, getTableCorners(), cv::Scalar(255));
    // These operations deliberately precede all Hough calls.
    result.foreground = foreground(image, table, p);
    result.clusters = clusterWindows(result.foreground, table, p);
    result.routingMask = cv::Mat::zeros(image.size(), CV_8UC1);
    for (const auto& region : result.clusters) {
        const int margin = cvRound(p.routingOverlapRadii * region.expectedRadius);
        const cv::Rect interior(region.core.x + margin, region.core.y + margin,
                                std::max(0, region.core.width - 2 * margin),
                                std::max(0, region.core.height - 2 * margin));
        if (!interior.empty())
            result.routingMask(interior).setTo(255);
    }
    const auto standard = runDetection(image, HoughParams{});
    result.baseline = standard.filteredCircles;
    for (const auto& c : result.baseline)
        if (!result.routingMask.at<uchar>(centre(c)))
            result.standardOutside.push_back(c);
    result.guided = result.standardOutside;
    for (size_t r = 0; r < result.clusters.size(); ++r) {
        auto& region = result.clusters[r];
        HoughParams local;
        local.param2 = p.localAccumulatorThreshold;
        local.minDist = p.localMinCentreDistance;
        local.minRadius = region.minRadius = std::max(p.absoluteMinLocalRadius,
            static_cast<int>(std::floor(p.localMinRadiusRatio * region.expectedRadius)));
        local.maxRadius = region.maxRadius = std::min(p.absoluteMaxLocalRadius,
            static_cast<int>(std::ceil(p.localMaxRadiusRatio * region.expectedRadius)));
        cv::Mat grey, smooth, edges, gx, gy;
        cv::cvtColor(image(region.crop), grey, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(grey, smooth, {5, 5}, 1.5);
        cv::Canny(smooth, edges, local.param1 / 2, local.param1);
        cv::Sobel(smooth, gx, CV_32F, 1, 0, 3);
        cv::Sobel(smooth, gy, CV_32F, 0, 1, 3);
        const auto circles = detectBalls(image(region.crop), local);
        for (const auto& c : circles) {
            GuidedProposal candidate;
            candidate.region = static_cast<int>(r + 1);
            candidate.circle = {c[0] + region.crop.x, c[1] + region.crop.y, c[2]};
            edgeEvidence(candidate, region, edges, gx, gy, result.foreground, p);
            if (!table.at<uchar>(centre(candidate.circle)))
                candidate.reason = "outside_table";
            else if (candidate.edgeSupport < p.minimumEdgeSupport)
                candidate.reason = "edge_support";
            else if (candidate.arcDegrees < p.minimumArcDegrees)
                candidate.reason = "visible_arc";
            else if (candidate.foregroundSupport < p.minimumForegroundSupport)
                candidate.reason = "foreground_support";
            else if (filterPocketDetections({candidate.circle}).empty())
                candidate.reason = "pocket_filter";
            result.proposals.push_back(candidate);
        }
    }
    std::vector<size_t> ranking(result.proposals.size());
    std::iota(ranking.begin(), ranking.end(), 0);
    std::stable_sort(ranking.begin(), ranking.end(), [&](size_t a, size_t b) {
        return result.proposals[a].quality > result.proposals[b].quality;
    });
    for (size_t i : ranking) {
        auto& candidate = result.proposals[i];
        if (!candidate.reason.empty())
            continue;
        if (duplicate(candidate.circle, result.guided, p))
            candidate.reason = "duplicate";
        else {
            candidate.reason = "accepted";
            candidate.accepted = true;
            candidate.guidedIndex = static_cast<int>(result.guided.size());
            result.guided.push_back(candidate.circle);
        }
    }
    // Fall back only when a cluster's local search accepts nothing, without looking at labels.
    for (size_t r = 0; r < result.clusters.size(); ++r) {
        auto& region = result.clusters[r];
        const bool found = std::any_of(result.proposals.begin(), result.proposals.end(), [&](const auto& c) {
            return c.accepted && c.region == static_cast<int>(r + 1);
        });
        if (!found) {
            region.usedEmptySearchFallback = true;
            for (const auto& c : result.baseline)
                if (region.core.contains(centre(c)) && !duplicate(c, result.guided, p))
                    result.guided.push_back(c);
        }
    }
    watershedCheck(image, table, result, p);
    return result;
}

} // namespace experiments
