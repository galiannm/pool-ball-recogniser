#include "watershedDetection.h"
#include "ccaGridSearch.h"
#include "recognitionGridSearch.h"
#include "config.h"
#include "detection/ballDetection.h"
#include "detection/geometricFilter.h"
#include "evaluation/evaluation.h"
#include "geometry/perspectiveTransform.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

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

bool duplicate(cv::Point2f point, double radius, const std::vector<cv::Vec3f>& circles, const WatershedOptions& p) {
    for (const auto& c : circles)
        if (cv::norm(point - cv::Point2f(c[0], c[1])) < p.minimumMarkerSeparationRadii * std::max(radius, double(c[2])))
            return true;
    return false;
}

std::vector<WatershedRegion> crowdedRegions(const cv::Mat& image, const std::vector<cv::Vec3f>& circles, const WatershedOptions& p) {
    std::vector<int> parent(circles.size());
    std::iota(parent.begin(), parent.end(), 0);
    const auto root = [&](int i) {
        while (parent[i] != i)
            i = parent[i];
        return i;
    };
    for (size_t i = 0; i < circles.size(); ++i)
        for (size_t j = i + 1; j < circles.size(); ++j)
            if (cv::norm(cv::Point2f(circles[i][0], circles[i][1]) - cv::Point2f(circles[j][0], circles[j][1])) <= p.groupDistanceFactor * (circles[i][2] + circles[j][2]))
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
        bounds = cv::Rect(bounds.x - padding, bounds.y - padding, bounds.width + 2 * padding, bounds.height + 2 * padding) & cv::Rect(0, 0, image.cols, image.rows);
        if (!bounds.empty())
            crops.push_back(bounds);
    }
    // Merge overlapping windows so clusters are not processed twice.
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

bool foreground(const cv::Mat& image, const cv::Mat& table, const std::vector<cv::Vec3f>& baseline, WatershedRegion& region, const WatershedOptions& p) {
    const auto local = image(region.crop);
    cv::Mat hsv, lab, samplesAllowed = table(region.crop).clone();
    cv::cvtColor(local, hsv, cv::COLOR_BGR2HSV);
    cv::cvtColor(local, lab, cv::COLOR_BGR2Lab);
    for (int i : region.houghIndices)
        cv::circle(samplesAllowed, centre(baseline[i]) - region.crop.tl(), cvRound(1.25 * baseline[i][2]), cv::Scalar(0), cv::FILLED);
    std::vector<cv::Vec3b> samples;
    std::vector<double> channels[3];
    // Sample cloth every second pixel.
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
    region.clothDistanceThreshold = std::clamp(typical + p.clothMADMultiplier * median(deviations), p.clothMinDistance, p.clothMaxDistance);
    region.foreground = cv::Mat::zeros(local.size(), CV_8UC1);
    for (int y = 0; y < local.rows; ++y)
        for (int x = 0; x < local.cols; ++x)
            if (cv::norm(cv::Vec3d(lab.at<cv::Vec3b>(y, x)) - cloth) > region.clothDistanceThreshold)
                region.foreground.at<uchar>(y, x) = 255;
    // Retain known Hough centres in the foreground mask.
    for (int i : region.houghIndices)
        cv::circle(region.foreground, centre(baseline[i]) - region.crop.tl(), cvRound(p.houghCoreRadius * baseline[i][2]), cv::Scalar(255), cv::FILLED);
    cv::morphologyEx(region.foreground, region.foreground, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_ELLIPSE, {p.closingKernel, p.closingKernel}));
    // Fill holes using a padded image.
    cv::Mat filled;
    cv::copyMakeBorder(region.foreground, filled, 1, 1, 1, 1, cv::BORDER_CONSTANT, cv::Scalar(0));
    cv::floodFill(filled, {0, 0}, cv::Scalar(255));
    cv::bitwise_not(filled, filled);
    cv::bitwise_or(region.foreground, filled(cv::Rect(1, 1, local.cols, local.rows)), region.foreground);
    cv::bitwise_and(region.foreground, table(region.crop), region.foreground);
    region.status = "segmented";
    return true;
}

void findPeaks(WatershedRegion& region, const std::vector<cv::Vec3f>& baseline, const WatershedOptions& p) {
    cv::distanceTransform(region.foreground, region.distance, cv::DIST_L2, cv::DIST_MASK_PRECISE);
    cv::Mat maxima, peakMask;
    const int half = std::max(1, cvRound(0.5 * p.peakNeighbourhoodRadii * region.expectedRadius));
    cv::dilate(region.distance, maxima, cv::getStructuringElement(cv::MORPH_RECT, {2 * half + 1, 2 * half + 1}));
    cv::compare(region.distance, maxima, peakMask, cv::CMP_GE);
    cv::Mat high;
    cv::compare(region.distance, p.minimumPeakRadii * region.expectedRadius, high, cv::CMP_GE);
    cv::bitwise_and(peakMask, high, peakMask);
    cv::Mat labels, stats, centroids;
    const int n = cv::connectedComponentsWithStats(peakMask, labels, stats, centroids, 8, CV_32S);
    for (int label = 1; label < n; ++label) {
        WatershedProposal peak;
        // Keep a peak pixel inside the plateau.
        double best = std::numeric_limits<double>::infinity();
        for (int y = stats.at<int>(label, cv::CC_STAT_TOP); y < stats.at<int>(label, cv::CC_STAT_TOP) + stats.at<int>(label, cv::CC_STAT_HEIGHT); ++y)
            for (int x = stats.at<int>(label, cv::CC_STAT_LEFT); x < stats.at<int>(label, cv::CC_STAT_LEFT) + stats.at<int>(label, cv::CC_STAT_WIDTH); ++x)
                if (labels.at<int>(y, x) == label) {
                    const double d = cv::norm(cv::Point2d(x, y) - cv::Point2d(centroids.at<double>(label, 0), centroids.at<double>(label, 1)));
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
        else if (peak.localCentre.x < margin || peak.localCentre.y < margin || peak.localCentre.x >= region.crop.width - margin || peak.localCentre.y >= region.crop.height - margin)
            peak.reason = "near_crop_edge";
        else {
            peak.label = nextLabel++;
            seeds.emplace_back(full.x, full.y, static_cast<float>(region.expectedRadius));
        }
    }
}

cv::Mat markers(const WatershedRegion& region, const cv::Mat& table, const std::vector<cv::Vec3f>& baseline, bool hybrid, const WatershedOptions& p) {
    cv::Mat expanded;
    cv::dilate(region.foreground, expanded, cv::getStructuringElement(cv::MORPH_ELLIPSE, {7, 7}));
    cv::Mat result = cv::Mat::zeros(region.foreground.size(), CV_32SC1);
    result.setTo(1, expanded == 0); // Sure background; leave a narrow unknown border.
    result.setTo(1, table(region.crop) == 0);
    int label = 2;
    for (int index : region.houghIndices)
        cv::circle(result, centre(baseline[index]) - region.crop.tl(), p.seedRadius, cv::Scalar(label++), cv::FILLED);
    if (hybrid)
        for (const auto& peak : region.proposals)
            if (peak.label)
                cv::circle(result, peak.localCentre, p.seedRadius, cv::Scalar(peak.label), cv::FILLED);
    return result;
}

void acceptBasins(WatershedRegion& region, std::vector<cv::Vec3f>& circles, const WatershedOptions& p) {
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

double ratio(double a, double b) {
    return b ? a / b : 0;
}

void writeMetrics(std::ostream& out, const Measured& m) {
    out << m.gt << ',' << m.tp + m.fp << ',' << m.tp << ',' << m.fp << ',' << m.fn << ',' << ratio(m.tp, m.tp + m.fp) << ',' << ratio(m.tp, m.gt) << ',' << ratio(2 * m.tp, 2 * m.tp + m.fp + m.fn) << ',' << ratio(m.totalError, m.tp) << ','
        << m.recovered << ',' << m.lost << ',' << m.types << ',' << m.numbers << ',' << ratio(m.types, m.tp) << ',' << ratio(m.numbers, m.tp) << ',' << ratio(m.numbers, m.gt);
}
const char* metricHeader = "groundTruth,detections,TP,FP,FN,precision,recall,F1,meanError,"
                           "recovered,lost,typeCorrect,numberCorrect,typeAccuracy,numberAccuracy,endToEndNumberRecall";

} // namespace


WatershedDetectionResult segmentCrowdedBalls(const cv::Mat& image, const std::vector<cv::Vec3f>& baseline, const WatershedOptions& p) {
    if (image.empty())
        return {baseline, {}};
    if (image.type() != CV_8UC3 || p.closingKernel < 1 || p.closingKernel % 2 != 1 || p.seedRadius < 1 || p.minClothSamples < 1 || p.clothMinDistance > p.clothMaxDistance)
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

// Comparison and diagnostics: same methods, same recognition settings, fewer exports.

void runWatershedDetectionExperiment(const std::filesystem::path& directory) {
    std::filesystem::create_directories(directory);
    const WatershedOptions options;

    auto metrics = output(directory / "metrics.csv");
    metrics << "method,split,image," << metricHeader << '\n';

    // Retain the evidence needed to explain why no new balls were accepted.
    auto proposals = output(directory / "proposals.csv");
    proposals << "image,region,peak,sourceX,sourceY,radius,peakDistance,watershedLabel,"
                 "areaRatio,circularity,aspectRatio,foregroundSupport,accepted,reason\n";
    auto misses = output(directory / "misses.csv");
    misses << "image,trueNumber,inCrowdedRegion,matchedBeforePocketFilter\n";
    auto changes = output(directory / "changes.csv");
    changes << "method,image,trueNumber,baselineMatched,methodMatched,change\n";

    std::map<std::pair<std::string, std::string>, Measured> totals;
    for (const auto& name : config::allImages()) {
        const cv::Mat image = cv::imread(config::mediaDirectory + name);
        if (image.empty())
            throw std::runtime_error("Could not load " + name);

        const auto truth = loadLabelledGroundTruth(config::groundTruthPath, name);
        if (truth.empty())
            throw std::runtime_error("No ground truth for " + name);

        std::vector<cv::Point2f> gt;
        for (const auto& ball : truth)
            gt.push_back(ball.centre);

        const auto hough = runDetection(image, HoughParams{});
        const auto baseline = evaluateDetections(
            transformCentres(hough.filteredCircles), gt, config::matchTolerance);
        const auto segmented = segmentCrowdedBalls(image, hough.filteredCircles, options);

        // Audit just the missed balls; this tells us whether the cluster selection
        // made recovery possible at all.
        const auto raw = evaluateDetections(
            transformCentres(hough.rawCircles), gt, config::matchTolerance);
        std::vector<cv::Point2f> sourceGT;
        cv::perspectiveTransform(gt, sourceGT, getTableTransformation().inv());
        for (size_t g = 0; g < gt.size(); ++g) {
            if (baseline.groundTruthToPrediction[g] >= 0)
                continue;
            bool inside = false;
            for (const auto& region : segmented.regions)
                inside |= region.crop.contains({cvRound(sourceGT[g].x), cvRound(sourceGT[g].y)});
            misses << name << ',' << truth[g].number << ',' << inside << ','
                   << (raw.groundTruthToPrediction[g] >= 0) << '\n';
        }

        for (size_t r = 0; r < segmented.regions.size(); ++r) {
            const auto& region = segmented.regions[r];
            for (size_t i = 0; i < region.proposals.size(); ++i) {
                const auto& peak = region.proposals[i];
                const auto full = peak.localCentre + region.crop.tl();
                proposals << name << ',' << r + 1 << ',' << i + 1 << ','
                          << full.x << ',' << full.y << ',' << region.expectedRadius << ','
                          << peak.peakDistance << ',' << peak.label << ',' << peak.areaRatio << ','
                          << peak.circularity << ',' << peak.aspectRatio << ','
                          << peak.foregroundSupport << ',' << peak.accepted << ','
                          << peak.reason << '\n';
            }
        }

        // The Hough-seeded watershed is a control: its circles are deliberately
        // unchanged. The hybrid may append accepted watershed proposals.
        const std::vector<std::pair<std::string, std::vector<cv::Vec3f>>> methods = {
            {"hough", hough.filteredCircles},
            {"hough_seeded", hough.filteredCircles},
            {"hybrid", segmented.hybridCircles}
        };
        const std::string split = name == config::tuningImage ? "reference" : "evaluation";

        for (const auto& [method, circles] : methods) {
            const auto centres = transformCentres(circles);
            const auto evaluation = evaluateDetections(centres, gt, config::matchTolerance);
            Measured m;
            m.gt = static_cast<int>(truth.size());
            m.tp = evaluation.TP;
            m.fp = evaluation.FP;
            m.fn = evaluation.FN;
            m.totalError = evaluation.meanError * evaluation.TP;

            for (size_t g = 0; g < gt.size(); ++g) {
                const bool before = baseline.groundTruthToPrediction[g] >= 0;
                const bool after = evaluation.groundTruthToPrediction[g] >= 0;
                m.recovered += !before && after;
                m.lost += before && !after;
                changes << method << ',' << name << ',' << truth[g].number << ','
                        << before << ',' << after << ','
                        << (!before && after ? "recovered" : before && !after ? "lost" : "unchanged")
                        << '\n';
            }

            // Preserve the original K=6 / solid-only CCA recognition evaluation.
            for (size_t i = 0; i < circles.size(); ++i) {
                const auto ball = extractBall(image, circles[i]);
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
                if (match < 0)
                    continue;
                const int actual = truth[match].number;
                m.types += type.type == trueType(actual);
                m.numbers += number == actual;
            }

            metrics << method << ',' << split << ',' << name << ',';
            writeMetrics(metrics, m);
            metrics << '\n';

            auto& total = totals[{method, split}];
            total.gt += m.gt;
            total.tp += m.tp;
            total.fp += m.fp;
            total.fn += m.fn;
            total.types += m.types;
            total.numbers += m.numbers;
            total.totalError += m.totalError;
            total.recovered += m.recovered;
            total.lost += m.lost;
        }
        std::cout << name << ": " << segmented.regions.size()
                  << " crowded regions, "
                  << segmented.hybridCircles.size() - hough.filteredCircles.size()
                  << " new candidates\n";
    }

    // Same aggregate metrics rows as the original program.
    for (const auto& [key, value] : totals) {
        metrics << key.first << ',' << key.second << ",ALL,";
        writeMetrics(metrics, value);
        metrics << '\n';
    }
}

} 
