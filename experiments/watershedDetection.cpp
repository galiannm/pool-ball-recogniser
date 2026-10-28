#include "watershedDetection.h"

#include "ccaGridSearch.h"
#include "recognitionGridSearch.h"
#include "config.h"
#include "detection/ballDetection.h"
#include "detection/geometricFilter.h"
#include "evaluation/evaluation.h"
#include "geometry/perspectiveTransform.h"
#include "visualisation/combineImages.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
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

cv::Point centre(const cv::Vec3f& circle) {
    return {cvRound(circle[0]), cvRound(circle[1])};
}

bool duplicate(cv::Point2f point, double radius, const std::vector<cv::Vec3f>& circles,
               const WatershedOptions& p) {
    for (const auto& c : circles)
        if (cv::norm(point - cv::Point2f(c[0], c[1])) <
            p.minimumMarkerSeparationRadii * std::max(radius, double(c[2])))
            return true;
    return false;
}

std::vector<WatershedRegion> crowdedRegions(const cv::Mat& image,
    const std::vector<cv::Vec3f>& circles, const WatershedOptions& p) {
    std::vector<int> parent(circles.size());
    std::iota(parent.begin(), parent.end(), 0);
    const auto root = [&](int i) { while (parent[i] != i) i = parent[i]; return i; };
    for (size_t i = 0; i < circles.size(); ++i)
        for (size_t j = i + 1; j < circles.size(); ++j)
            if (cv::norm(cv::Point2f(circles[i][0], circles[i][1]) -
                         cv::Point2f(circles[j][0], circles[j][1])) <=
                p.groupDistanceFactor * (circles[i][2] + circles[j][2]))
                parent[root(static_cast<int>(j))] = root(static_cast<int>(i));
    std::map<int, std::vector<int>> groups;
    for (size_t i = 0; i < circles.size(); ++i)
        groups[root(static_cast<int>(i))].push_back(static_cast<int>(i));
    std::vector<cv::Rect> crops;
    for (const auto& group : groups) {
        if (group.second.size() < 2)
            continue;
        cv::Rect bounds;
        std::vector<double> radii;
        for (int i : group.second) {
            const int r = std::max(1, cvRound(circles[i][2]));
            const auto c = centre(circles[i]);
            const cv::Rect disk(c.x - r, c.y - r, 2 * r + 1, 2 * r + 1);
            bounds = bounds.empty() ? disk : bounds | disk;
            radii.push_back(r);
        }
        const int padding = cvRound(p.cropPaddingRadii * median(radii));
        bounds = cv::Rect(bounds.x - padding, bounds.y - padding,
                          bounds.width + 2 * padding, bounds.height + 2 * padding)
                 & cv::Rect(0, 0, image.cols, image.rows);
        if (!bounds.empty())
            crops.push_back(bounds);
    }
    // Merge overlapping windows so the same cluster is not independently split twice.
    bool changed = true;
    while (changed) {
        changed = false;
        for (size_t i = 0; i < crops.size() && !changed; ++i)
            for (size_t j = i + 1; j < crops.size(); ++j)
                if (!(crops[i] & crops[j]).empty()) {
                    crops[i] |= crops[j];
                    crops.erase(crops.begin() + j);
                    changed = true;
                    break;
                }
    }
    std::vector<WatershedRegion> result;
    for (const auto& crop : crops) {
        WatershedRegion region;
        region.crop = crop;
        std::vector<double> radii;
        for (size_t i = 0; i < circles.size(); ++i)
            if (crop.contains(centre(circles[i]))) {
                region.houghIndices.push_back(static_cast<int>(i));
                radii.push_back(circles[i][2]);
            }
        region.expectedRadius = median(radii);
        result.push_back(std::move(region));
    }
    return result;
}

bool foreground(const cv::Mat& image, const cv::Mat& table,
                const std::vector<cv::Vec3f>& baseline, WatershedRegion& region,
                const WatershedOptions& p) {
    const auto local = image(region.crop);
    cv::Mat hsv, lab, samplesAllowed = table(region.crop).clone();
    cv::cvtColor(local, hsv, cv::COLOR_BGR2HSV);
    cv::cvtColor(local, lab, cv::COLOR_BGR2Lab);
    for (int i : region.houghIndices)
        cv::circle(samplesAllowed, centre(baseline[i]) - region.crop.tl(),
                   cvRound(1.25 * baseline[i][2]), cv::Scalar(0), cv::FILLED);
    std::vector<cv::Vec3b> samples;
    std::vector<double> channels[3];
    // A two-pixel sampling stride reduces the influence of adjacent duplicate samples.
    for (int y = 0; y < local.rows; y += 2)
        for (int x = 0; x < local.cols; x += 2) {
            const auto h = hsv.at<cv::Vec3b>(y, x);
            if (samplesAllowed.at<uchar>(y, x) && h[0] >= 35 && h[0] <= 95 && h[1] >= 35 && h[2] >= 30) {
                const auto pixel = lab.at<cv::Vec3b>(y, x);
                samples.push_back(pixel);
                for (int c = 0; c < 3; ++c)
                    channels[c].push_back(pixel[c]);
            }
        }
    region.clothSamples = static_cast<int>(samples.size());
    if (region.clothSamples < p.minClothSamples) {
        region.status = "insufficient_cloth_samples";
        return false;
    }
    const cv::Vec3d cloth(median(channels[0]), median(channels[1]), median(channels[2]));
    std::vector<double> distances, deviations;
    for (const auto& pixel : samples)
        distances.push_back(cv::norm(cv::Vec3d(pixel) - cloth));
    const double typical = median(distances);
    for (double d : distances)
        deviations.push_back(std::abs(d - typical));
    region.clothDistanceThreshold = std::clamp(typical + p.clothMADMultiplier * median(deviations),
                                               p.clothMinDistance, p.clothMaxDistance);
    region.foreground = cv::Mat::zeros(local.size(), CV_8UC1);
    for (int y = 0; y < local.rows; ++y)
        for (int x = 0; x < local.cols; ++x)
            if (cv::norm(cv::Vec3d(lab.at<cv::Vec3b>(y, x)) - cloth) > region.clothDistanceThreshold)
                region.foreground.at<uchar>(y, x) = 255;
    // Preserve a small foreground core around known detections, including green balls.
    for (int i : region.houghIndices)
        cv::circle(region.foreground, centre(baseline[i]) - region.crop.tl(),
                   cvRound(p.houghCoreRadius * baseline[i][2]), cv::Scalar(255), cv::FILLED);
    cv::morphologyEx(region.foreground, region.foreground, cv::MORPH_CLOSE,
        cv::getStructuringElement(cv::MORPH_ELLIPSE, {p.closingKernel, p.closingKernel}));
    // Fill enclosed holes; a padded exterior avoids assuming the crop corner is background.
    cv::Mat filled;
    cv::copyMakeBorder(region.foreground, filled, 1, 1, 1, 1, cv::BORDER_CONSTANT, cv::Scalar(0));
    cv::floodFill(filled, {0, 0}, cv::Scalar(255));
    cv::bitwise_not(filled, filled);
    cv::bitwise_or(region.foreground, filled(cv::Rect(1, 1, local.cols, local.rows)), region.foreground);
    cv::bitwise_and(region.foreground, table(region.crop), region.foreground);
    region.status = "segmented";
    return true;
}

void findPeaks(WatershedRegion& region, const std::vector<cv::Vec3f>& baseline,
               const WatershedOptions& p) {
    cv::distanceTransform(region.foreground, region.distance, cv::DIST_L2, cv::DIST_MASK_PRECISE);
    cv::Mat maxima, peakMask;
    const int half = std::max(1, cvRound(0.5 * p.peakNeighbourhoodRadii * region.expectedRadius));
    cv::dilate(region.distance, maxima,
               cv::getStructuringElement(cv::MORPH_RECT, {2 * half + 1, 2 * half + 1}));
    cv::compare(region.distance, maxima, peakMask, cv::CMP_GE);
    cv::Mat high;
    cv::compare(region.distance, p.minimumPeakRadii * region.expectedRadius, high, cv::CMP_GE);
    cv::bitwise_and(peakMask, high, peakMask);
    cv::Mat labels, stats, centroids;
    const int n = cv::connectedComponentsWithStats(peakMask, labels, stats, centroids, 8, CV_32S);
    for (int label = 1; label < n; ++label) {
        WatershedProposal peak;
        // Select an actual peak pixel nearest its plateau centroid, never a point outside the plateau.
        double best = std::numeric_limits<double>::infinity();
        for (int y = stats.at<int>(label, cv::CC_STAT_TOP);
             y < stats.at<int>(label, cv::CC_STAT_TOP) + stats.at<int>(label, cv::CC_STAT_HEIGHT); ++y)
            for (int x = stats.at<int>(label, cv::CC_STAT_LEFT);
                 x < stats.at<int>(label, cv::CC_STAT_LEFT) + stats.at<int>(label, cv::CC_STAT_WIDTH); ++x)
                if (labels.at<int>(y, x) == label) {
                    const double d = cv::norm(cv::Point2d(x, y) -
                        cv::Point2d(centroids.at<double>(label, 0), centroids.at<double>(label, 1)));
                    if (d < best) {
                        best = d;
                        peak.localCentre = {x, y};
                    }
                }
        peak.peakDistance = region.distance.at<float>(peak.localCentre);
        region.proposals.push_back(peak);
    }
    std::stable_sort(region.proposals.begin(), region.proposals.end(), [](const auto& a, const auto& b) {
        return a.peakDistance > b.peakDistance;
    });
    std::vector<cv::Vec3f> seeds = baseline;
    int nextLabel = static_cast<int>(region.houghIndices.size()) + 2;
    for (auto& peak : region.proposals) {
        const auto full = peak.localCentre + region.crop.tl();
        const double margin = p.cropEdgeMarginRadii * region.expectedRadius;
        if (duplicate(full, region.expectedRadius, baseline, p))
            peak.reason = "hough_duplicate";
        else if (duplicate(full, region.expectedRadius, seeds, p))
            peak.reason = "new_seed_duplicate";
        else if (peak.localCentre.x < margin || peak.localCentre.y < margin ||
                 peak.localCentre.x >= region.crop.width - margin || peak.localCentre.y >= region.crop.height - margin)
            peak.reason = "near_crop_edge";
        else {
            peak.label = nextLabel++;
            seeds.emplace_back(full.x, full.y, static_cast<float>(region.expectedRadius));
        }
    }
}

cv::Mat markers(const WatershedRegion& region, const cv::Mat& table,
                 const std::vector<cv::Vec3f>& baseline, bool hybrid, const WatershedOptions& p) {
    cv::Mat expanded;
    cv::dilate(region.foreground, expanded,
               cv::getStructuringElement(cv::MORPH_ELLIPSE, {7, 7}));
    cv::Mat result = cv::Mat::zeros(region.foreground.size(), CV_32SC1);
    result.setTo(1, expanded == 0); // Sure background; the collar remains unknown.
    result.setTo(1, table(region.crop) == 0);
    int label = 2;
    for (int index : region.houghIndices)
        cv::circle(result, centre(baseline[index]) - region.crop.tl(), p.seedRadius,
                   cv::Scalar(label++), cv::FILLED);
    if (hybrid)
        for (const auto& peak : region.proposals)
            if (peak.label)
                cv::circle(result, peak.localCentre, p.seedRadius, cv::Scalar(peak.label), cv::FILLED);
    return result;
}

void acceptBasins(WatershedRegion& region, std::vector<cv::Vec3f>& circles,
                   const WatershedOptions& p) {
    for (auto& peak : region.proposals) {
        if (!peak.label)
            continue;
        cv::Mat basin;
        cv::compare(region.hybridLabels, peak.label, basin, cv::CMP_EQ);
        cv::bitwise_and(basin, region.foreground, basin);
        const double idealArea = CV_PI * region.expectedRadius * region.expectedRadius;
        peak.areaRatio = cv::countNonZero(basin) / idealArea;
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(basin.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        if (!contours.empty()) {
            const auto largest = std::max_element(contours.begin(), contours.end(), [](const auto& a, const auto& b) {
                return cv::contourArea(a) < cv::contourArea(b);
            });
            const double perimeter = cv::arcLength(*largest, true);
            peak.circularity = perimeter > 0 ? 4 * CV_PI * cv::contourArea(*largest) / (perimeter * perimeter) : 0;
            const auto bounds = cv::boundingRect(*largest);
            peak.aspectRatio = double(std::max(bounds.width, bounds.height)) / std::min(bounds.width, bounds.height);
        }
        cv::Mat disk = cv::Mat::zeros(basin.size(), CV_8UC1), support;
        cv::circle(disk, peak.localCentre, cvRound(region.expectedRadius), cv::Scalar(255), cv::FILLED);
        cv::bitwise_and(disk, region.foreground, support);
        peak.foregroundSupport = double(cv::countNonZero(support)) / std::max(1, cv::countNonZero(disk));
        const auto full = peak.localCentre + region.crop.tl();
        const cv::Vec3f circle(full.x, full.y, static_cast<float>(region.expectedRadius));
        if (peak.areaRatio < p.minBasinAreaRatio || peak.areaRatio > p.maxBasinAreaRatio)
            peak.reason = "basin_area";
        else if (peak.circularity < p.minCircularity)
            peak.reason = "basin_circularity";
        else if (peak.aspectRatio > p.maxAspectRatio)
            peak.reason = "basin_aspect";
        else if (peak.foregroundSupport < p.minForegroundSupport)
            peak.reason = "foreground_support";
        else if (duplicate(full, region.expectedRadius, circles, p))
            peak.reason = "final_duplicate";
        else if (filterPocketDetections({circle}).empty())
            peak.reason = "pocket_filter";
        else {
            peak.accepted = true;
            peak.reason = "accepted";
            circles.push_back(circle);
        }
    }
}

cv::Scalar labelColour(int label) {
    if (label <= 1)
        return {0, 0, 0};
    return {double(60 + label * 73 % 196), double(60 + label * 137 % 196), double(60 + label * 193 % 196)};
}

cv::Mat labelView(const cv::Mat& image, const cv::Mat& labels, bool overlay) {
    if (labels.empty())
        return {};
    cv::Mat result = overlay ? image.clone() : cv::Mat::zeros(image.size(), CV_8UC3);
    for (int y = 0; y < labels.rows; ++y)
        for (int x = 0; x < labels.cols; ++x) {
            const int label = labels.at<int>(y, x);
            if (label > 1) {
                const auto c = labelColour(label);
                auto& pixel = result.at<cv::Vec3b>(y, x);
                for (int channel = 0; channel < 3; ++channel)
                    pixel[channel] = cv::saturate_cast<uchar>(overlay ? 0.45 * pixel[channel] + 0.55 * c[channel] : c[channel]);
            } else if (label == -1)
                result.at<cv::Vec3b>(y, x) = {255, 255, 255};
        }
    return result;
}

void save(const std::filesystem::path& path, const cv::Mat& image) {
    if (!cv::imwrite(path.string(), image))
        throw std::runtime_error("Could not save " + path.string());
}

std::ofstream output(const std::filesystem::path& path) {
    std::ofstream stream(path);
    stream.exceptions(std::ios::failbit | std::ios::badbit);
    stream << std::fixed << std::setprecision(6);
    return stream;
}

std::string trueType(int n) {
    return n < 0 ? "Unknown" : n == 0 ? "White" : n <= 8 ? "Solid" : "Striped";
}

struct Measured {
    int gt = 0, tp = 0, fp = 0, fn = 0, types = 0, numbers = 0, recovered = 0, lost = 0;
    double totalError = 0;
};

double ratio(double a, double b) { return b ? a / b : 0; }

void writeMetrics(std::ostream& out, const Measured& m) {
    out << m.gt << ',' << m.tp + m.fp << ',' << m.tp << ',' << m.fp << ',' << m.fn << ','
        << ratio(m.tp, m.tp + m.fp) << ',' << ratio(m.tp, m.gt) << ','
        << ratio(2 * m.tp, 2 * m.tp + m.fp + m.fn) << ',' << ratio(m.totalError, m.tp) << ','
        << m.recovered << ',' << m.lost << ',' << m.types << ',' << m.numbers << ','
        << ratio(m.types, m.tp) << ',' << ratio(m.numbers, m.tp) << ',' << ratio(m.numbers, m.gt);
}

const char* metricHeader = "groundTruth,detections,TP,FP,FN,precision,recall,F1,meanError,"
    "recovered,lost,typeCorrect,numberCorrect,typeAccuracy,numberAccuracy,endToEndNumberRecall";

cv::Mat detectionView(const cv::Mat& image, const std::vector<cv::Vec3f>& circles,
    const EvaluationResult& evaluation, const std::vector<LabelledBall>& truth,
    const std::vector<cv::Point2f>& sourceTruth) {
    cv::Mat view = image.clone();
    for (size_t i = 0; i < circles.size(); ++i) {
        const int match = evaluation.predictionToGroundTruth[i];
        const cv::Scalar colour = match < 0 ? cv::Scalar(80, 80, 255) : cv::Scalar(70, 240, 70);
        cv::circle(view, centre(circles[i]), cvRound(circles[i][2]), colour, 2);
        cv::putText(view, std::to_string(i + 1), centre(circles[i]) + cv::Point(3, -3),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, colour, 1);
    }
    for (size_t g = 0; g < truth.size(); ++g)
        if (evaluation.groundTruthToPrediction[g] < 0) {
            const cv::Point point(cvRound(sourceTruth[g].x), cvRound(sourceTruth[g].y));
            cv::drawMarker(view, point, {255, 150, 0}, cv::MARKER_CROSS, 18, 2);
            cv::putText(view, "miss " + std::to_string(truth[g].number), point + cv::Point(5, 15),
                        cv::FONT_HERSHEY_SIMPLEX, 0.45, {255, 150, 0}, 1);
        }
    return view;
}

void regionDiagnostics(const cv::Mat& image, const std::vector<cv::Vec3f>& baseline,
                       const WatershedDetectionResult& result, const std::filesystem::path& directory) {
    for (size_t i = 0; i < result.regions.size(); ++i) {
        const auto& r = result.regions[i];
        cv::Mat original = image(r.crop).clone(), accepted = original.clone();
        for (int index : r.houghIndices)
            cv::circle(original, centre(baseline[index]) - r.crop.tl(), cvRound(baseline[index][2]), {60, 220, 60}, 1);
        for (const auto& p : r.proposals) {
            cv::circle(accepted, p.localCentre, 3, p.accepted ? cv::Scalar(0, 255, 255) : cv::Scalar(80, 80, 255), cv::FILLED);
            if (p.accepted)
                cv::circle(accepted, p.localCentre, cvRound(r.expectedRadius), {0, 255, 255}, 2);
        }
        cv::Mat distance;
        if (!r.distance.empty())
            cv::normalize(r.distance, distance, 0, 255, cv::NORM_MINMAX, CV_8U);
        const std::vector<ImagePanel> panels = {
            {"Original / Hough circles", original}, {"Foreground / " + r.status, r.foreground},
            {"Distance transform", distance}, {"Hough markers", labelView(image(r.crop), r.seededMarkers, false)},
            {"Hybrid markers", labelView(image(r.crop), r.hybridMarkers, false)},
            {"Hough-seeded watershed", labelView(image(r.crop), r.seededLabels, true)},
            {"Hybrid watershed", labelView(image(r.crop), r.hybridLabels, true)},
            {"Yellow: accepted new / red: rejected peaks", accepted}};
        save(directory / ("region_" + std::to_string(i + 1) + "_diagnostics.png"), combineImages(panels, 3));
    }
}

} // namespace

WatershedDetectionResult segmentCrowdedBalls(const cv::Mat& image,
    const std::vector<cv::Vec3f>& baseline, const WatershedOptions& p) {
    if (image.empty())
        return {baseline, {}};
    if (image.type() != CV_8UC3 || p.closingKernel < 1 || p.closingKernel % 2 != 1 ||
        p.seedRadius < 1 || p.minClothSamples < 1 || p.clothMinDistance > p.clothMaxDistance)
        throw std::invalid_argument("Invalid watershed image/options");
    for (const auto& c : baseline)
        if (!std::isfinite(c[0]) || !std::isfinite(c[1]) || !std::isfinite(c[2]) || c[2] <= 0)
            throw std::invalid_argument("Invalid Hough circle");
    WatershedDetectionResult result;
    result.hybridCircles = baseline;
    result.regions = crowdedRegions(image, baseline, p);
    cv::Mat table = cv::Mat::zeros(image.size(), CV_8UC1);
    cv::fillConvexPoly(table, getTableCorners(), cv::Scalar(255));
    for (auto& region : result.regions) {
        if (!foreground(image, table, baseline, region, p))
            continue;
        findPeaks(region, baseline, p);
        region.seededMarkers = markers(region, table, baseline, false, p);
        region.hybridMarkers = markers(region, table, baseline, true, p);
        region.seededLabels = region.seededMarkers.clone();
        region.hybridLabels = region.hybridMarkers.clone();
        cv::Mat smoothed;
        cv::GaussianBlur(image(region.crop), smoothed, {3, 3}, 0.8);
        cv::watershed(smoothed, region.seededLabels);
        cv::watershed(smoothed, region.hybridLabels);
        acceptBasins(region, result.hybridCircles, p);
    }
    return result;
}

void runWatershedDetectionExperiment(const std::filesystem::path& directory) {
    std::filesystem::create_directories(directory);
    const WatershedOptions options;
    auto optionCSV = output(directory / "parameters.csv");
    optionCSV << "parameter,value\n";
    const std::vector<std::pair<std::string, double>> optionValues = {
        {"groupDistanceFactor", options.groupDistanceFactor}, {"cropPaddingRadii", options.cropPaddingRadii},
        {"minClothSamples", options.minClothSamples}, {"clothMinDistance", options.clothMinDistance},
        {"clothMaxDistance", options.clothMaxDistance}, {"clothMADMultiplier", options.clothMADMultiplier},
        {"houghCoreRadius", options.houghCoreRadius}, {"closingKernel", options.closingKernel},
        {"peakNeighbourhoodRadii", options.peakNeighbourhoodRadii}, {"minimumPeakRadii", options.minimumPeakRadii},
        {"minimumMarkerSeparationRadii", options.minimumMarkerSeparationRadii},
        {"cropEdgeMarginRadii", options.cropEdgeMarginRadii}, {"seedRadius", options.seedRadius},
        {"minBasinAreaRatio", options.minBasinAreaRatio}, {"maxBasinAreaRatio", options.maxBasinAreaRatio},
        {"minCircularity", options.minCircularity}, {"maxAspectRatio", options.maxAspectRatio},
        {"minForegroundSupport", options.minForegroundSupport}};
    for (const auto& setting : optionValues)
        optionCSV << setting.first << ',' << setting.second << '\n';
    auto metrics = output(directory / "metrics.csv");
    metrics << "method,split,image," << metricHeader << '\n';
    auto audit = output(directory / "ground_truth_audit.csv");
    audit << "image,trueNumber,tableX,tableY,sourceX,sourceY,rawHoughMatched,baselineMatched,"
        "insideCrowdedROI,nearestBaselineError,status\n";
    auto regionCSV = output(directory / "regions.csv");
    regionCSV << "image,region,x,y,width,height,houghSeeds,expectedRadius,clothSamples,clothDistanceThreshold,status\n";
    auto proposals = output(directory / "proposals.csv");
    proposals << "image,region,peak,sourceX,sourceY,radius,peakDistance,watershedLabel,areaRatio,"
        "circularity,aspectRatio,foregroundSupport,accepted,reason\n";
    auto predictions = output(directory / "predictions.csv");
    predictions << "method,image,candidate,origin,x,y,radius,tableX,tableY,trueNumber,type,number,typeCorrect,numberCorrect,error\n";
    auto changes = output(directory / "changes.csv");
    changes << "method,image,trueNumber,baselineMatched,methodMatched,change\n";
    auto settings = output(directory / "parameters.md");
    settings << "# Fixed prototype parameters\n\nSee experiments/watershedDetection.h for defaults. "
        << "No parameter grid or annotation-driven marker selection is used.\n\n"
        << "- Nearby-circle grouping: distance <= 1.4 times sum of radii; at least two detections.\n"
        << "- Crop padding: 2 median radii; overlapping crops merge.\n"
        << "- Cloth: HSV H 35-95, S>=35, V>=30; sample stride 2; exclude disks of 1.25 Hough radii.\n"
        << "- Lab: per-channel median cloth; threshold = median distance + 2.5 MAD, clipped to [16,45].\n"
        << "- Require 40 cloth samples; keep known foreground cores of 0.65 Hough radius.\n"
        << "- Close with a 3x3 ellipse, fill enclosed holes, intersect with table polygon.\n"
        << "- Distance maxima: neighbourhood approximately 0.9 median radii wide; distance >=0.45 radius.\n"
        << "- Markers: radius 3; separation >=1.15 times the larger radius; crop edge margin 0.6 radius.\n"
        << "- Sure background: outside a 7x7 dilation of foreground; unknown collar inside it.\n"
        << "- Watershed image: original BGR crop smoothed with 3x3 Gaussian, sigma=0.8.\n"
        << "- Basin area / expected disk area: [0.25,1.8]; circularity >=0.35; aspect ratio <=2.2.\n"
        << "- Foreground coverage of expected disk >=0.35; existing pocket filter applies to additions.\n"
        << "- New centre: distance peak, not basin centroid. New radius: local median Hough radius.\n"
        << "- Recognition: configuration 2153 plus selected solid-only CCA (2% cutoff, 5x5 band, "
        << "non-light>=70%, light boundary<=20%, exactly one non-light component). "
        << "Standard circular masks are used for all methods; watershed masks are diagnostic only.\n";
    std::map<std::pair<std::string, std::string>, Measured> totals;
    std::map<std::string, int> proposalCounts;
    std::vector<std::string> misses;
    for (const auto& name : config::allImages()) {
        const cv::Mat image = cv::imread(config::mediaDirectory + name);
        if (image.empty())
            throw std::runtime_error("Could not load " + name);
        const auto truth = loadLabelledGroundTruth(config::groundTruthPath, name);
        if (truth.empty())
            throw std::runtime_error("No ground truth for " + name);
        std::vector<cv::Point2f> gt;
        for (const auto& t : truth)
            gt.push_back(t.centre);
        std::vector<cv::Point2f> sourceGT;
        cv::perspectiveTransform(gt, sourceGT, getTableTransformation().inv());
        const auto hough = runDetection(image, HoughParams{});
        const auto baselineEvaluation = evaluateDetections(transformCentres(hough.filteredCircles), gt, config::matchTolerance);
        const auto rawEvaluation = evaluateDetections(transformCentres(hough.rawCircles), gt, config::matchTolerance);
        const auto segmented = segmentCrowdedBalls(image, hough.filteredCircles, options);
        const auto imageDirectory = directory / std::filesystem::path(name).stem();
        std::filesystem::create_directories(imageDirectory);
        for (size_t g = 0; g < truth.size(); ++g) {
            bool inRegion = false;
            for (const auto& region : segmented.regions)
                inRegion |= region.crop.contains(cv::Point(cvRound(sourceGT[g].x), cvRound(sourceGT[g].y)));
            double nearest = std::numeric_limits<double>::infinity();
            for (const auto& point : transformCentres(hough.filteredCircles))
                nearest = std::min(nearest, cv::norm(point - gt[g]));
            const bool matched = baselineEvaluation.groundTruthToPrediction[g] >= 0;
            const bool rawMatched = rawEvaluation.groundTruthToPrediction[g] >= 0;
            if (!matched)
                misses.push_back(name + ": ball " + std::to_string(truth[g].number) +
                    (inRegion ? " (inside a crowded ROI)" : " (outside crowded ROIs)") +
                    (rawMatched ? "; matched before pocket filtering" : "; already missed before pocket filtering"));
            audit << name << ',' << truth[g].number << ',' << gt[g].x << ',' << gt[g].y << ','
                << sourceGT[g].x << ',' << sourceGT[g].y << ',' << rawMatched << ',' << matched << ','
                << inRegion << ',' << nearest << ',' << (matched ? "matched" : rawMatched ? "removed_by_pocket_filter"
                    : inRegion ? "miss_inside_crowded_roi" : "miss_outside_crowded_roi") << '\n';
        }
        for (size_t r = 0; r < segmented.regions.size(); ++r) {
            const auto& region = segmented.regions[r];
            regionCSV << name << ',' << r + 1 << ',' << region.crop.x << ',' << region.crop.y << ','
                << region.crop.width << ',' << region.crop.height << ',' << region.houghIndices.size() << ','
                << region.expectedRadius << ',' << region.clothSamples << ',' << region.clothDistanceThreshold << ',' << region.status << '\n';
            for (size_t i = 0; i < region.proposals.size(); ++i) {
                const auto& peak = region.proposals[i];
                ++proposalCounts[peak.reason];
                const auto full = peak.localCentre + region.crop.tl();
                proposals << name << ',' << r + 1 << ',' << i + 1 << ',' << full.x << ',' << full.y << ','
                    << region.expectedRadius << ',' << peak.peakDistance << ',' << peak.label << ',' << peak.areaRatio << ','
                    << peak.circularity << ',' << peak.aspectRatio << ',' << peak.foregroundSupport << ','
                    << peak.accepted << ',' << peak.reason << '\n';
            }
        }
        regionDiagnostics(image, hough.filteredCircles, segmented, imageDirectory);
        std::vector<ImagePanel> overview;
        const std::vector<std::pair<std::string, std::vector<cv::Vec3f>>> methods = {
            {"hough", hough.filteredCircles}, {"hough_seeded", hough.filteredCircles},
            {"hybrid", segmented.hybridCircles}};
        for (const auto& method : methods) {
            const auto tableCentres = transformCentres(method.second);
            const auto evaluation = evaluateDetections(tableCentres, gt, config::matchTolerance);
            Measured m;
            m.gt = static_cast<int>(truth.size());
            m.tp = evaluation.TP;
            m.fp = evaluation.FP;
            m.fn = evaluation.FN;
            m.totalError = evaluation.meanError * evaluation.TP;
            for (size_t g = 0; g < truth.size(); ++g) {
                const bool old = baselineEvaluation.groundTruthToPrediction[g] >= 0;
                const bool now = evaluation.groundTruthToPrediction[g] >= 0;
                m.recovered += !old && now;
                m.lost += old && !now;
                changes << method.first << ',' << name << ',' << truth[g].number << ',' << old << ',' << now
                    << ',' << (!old && now ? "recovered" : old && !now ? "lost" : "unchanged") << '\n';
            }
            for (size_t i = 0; i < method.second.size(); ++i) {
                const auto& circle = method.second[i];
                const auto ball = extractBall(image, circle);
                const auto clusters = segmentBallKMeans(ball, 6);
                const auto recognition = ccaRecognitionBase();
                const auto baseType = classifyExperimentalType(clusters, recognition);
                CCAParameters cca;
                cca.cue = cca.stripe = false;
                cca.solidMaxBoundary = 20;
                const auto features = measureExperimentalCCA(ball, clusters, recognition, cca);
                const auto type = classifyExperimentalCCA(baseType, features, cca);
                const int number = classifyExperimentalNumber(clusters, type.type, recognition);
                const int match = evaluation.predictionToGroundTruth[i];
                const int actual = match < 0 ? -1 : truth[match].number;
                predictions << method.first << ',' << name << ',' << i + 1 << ','
                    << (i < hough.filteredCircles.size() ? "hough" : "watershed") << ',' << circle[0] << ','
                    << circle[1] << ',' << circle[2] << ',' << tableCentres[i].x << ',' << tableCentres[i].y << ','
                    << actual << ',' << type.type << ',' << number << ',';
                if (match >= 0) {
                    const bool typeCorrect = type.type == trueType(actual), numberCorrect = number == actual;
                    m.types += typeCorrect;
                    m.numbers += numberCorrect;
                    predictions << typeCorrect << ',' << numberCorrect << ',' << cv::norm(tableCentres[i] - gt[match]);
                } else {
                    predictions << ",,";
                }
                predictions << '\n';
            }
            const std::string split = name == config::tuningImage ? "reference" : "evaluation";
            metrics << method.first << ',' << split << ',' << name << ',';
            writeMetrics(metrics, m);
            metrics << '\n';
            auto& total = totals[{method.first, split}];
            total.gt += m.gt; total.tp += m.tp; total.fp += m.fp; total.fn += m.fn;
            total.types += m.types; total.numbers += m.numbers; total.totalError += m.totalError;
            total.recovered += m.recovered; total.lost += m.lost;
            overview.push_back({method.first + " / TP " + std::to_string(m.tp) + " FP " + std::to_string(m.fp)
                + " FN " + std::to_string(m.fn) + " / green match, red FP, blue missed GT",
                detectionView(image, method.second, evaluation, truth, sourceGT)});
        }
        save(imageDirectory / "detection_comparison.png", combineImages(overview, 1));
        std::cout << name << ": " << segmented.regions.size() << " crowded regions, "
            << segmented.hybridCircles.size() - hough.filteredCircles.size() << " new candidates\n" << std::flush;
    }
    auto summary = output(directory / "summary.md");
    summary << "# Watershed detection experiment\n\nFixed prototype; no grid search. "
        << "Ground truth is used only for evaluation and diagnostics, never masks or seeds.\n\n"
        << "| Method | Split | TP | FP | FN | Precision | Recall | F1 | Mean error | Recovered/lost | Correct type/number |\n"
        << "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n";
    for (const auto& row : totals) {
        const auto& m = row.second;
        metrics << row.first.first << ',' << row.first.second << ",ALL,";
        writeMetrics(metrics, m);
        metrics << '\n';
        summary << '|' << row.first.first << '|' << row.first.second << '|' << m.tp << '|' << m.fp << '|' << m.fn
            << '|' << ratio(m.tp, m.tp + m.fp) << '|' << ratio(m.tp, m.gt) << '|'
            << ratio(2 * m.tp, 2 * m.tp + m.fp + m.fn) << '|' << ratio(m.totalError, m.tp) << '|'
            << m.recovered << '/' << m.lost << '|' << m.types << '/' << m.numbers << "|\n";
    }
    summary << "\nHough-seeded watershed is a segmentation control: its circles and standard extraction masks "
        << "are deliberately unchanged, so detection and recognition scores should equal Hough. "
        << "Hybrid can append accepted peak-centred candidates but does not relocate or delete Hough circles. "
        << "Recognition is frozen to configuration 2153 plus the selected solid-only CCA rule. "
        << "Watershed labels are diagnostic and are not substituted for circular recognition masks.\n\n"
        << "Baseline misses and pocket-filter status are in ground_truth_audit.csv. Parameters are in "
        << "parameters.md and watershedDetection.h. Basin rejection reasons are in proposals.csv. "
        << "A local window requires at least two nearby filtered Hough detections; wholly missed or isolated "
        << "balls outside those windows cannot be recovered. Green balls can still merge into the cloth mask. "
        << "Watershed cannot reconstruct an invisible boundary or guarantee an unbiased centre for an occluded ball. "
        << "No pixel-level annotations are available, so segmentation quality is inspected visually. "
        << "Recognition parameters were previously tuned on Table1-6, so recognition scores are not independent test estimates.\n";
    summary << "\nBaseline misses:\n\n";
    for (const auto& miss : misses)
        summary << "- " << miss << '\n';
    summary << "\nPeak outcomes across all seven images:\n\n";
    for (const auto& count : proposalCounts)
        summary << "- " << count.first << ": " << count.second << '\n';
}

} // namespace experiments
