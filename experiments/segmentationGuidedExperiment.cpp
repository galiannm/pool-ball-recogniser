#include "segmentationGuidedDetection.h"
#include "ccaGridSearch.h"
#include "recognitionGridSearch.h"
#include "config.h"
#include "evaluation/evaluation.h"
#include "geometry/perspectiveTransform.h"
#include "visualisation/combineImages.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>

namespace experiments {
namespace {

std::ofstream output(const std::filesystem::path& path) {
    std::ofstream stream(path);
    stream.exceptions(std::ios::failbit | std::ios::badbit);
    stream << std::fixed << std::setprecision(6);
    return stream;
}

void save(const std::filesystem::path& path, const cv::Mat& image) {
    if (!cv::imwrite(path.string(), image))
        throw std::runtime_error("Could not save " + path.string());
}

cv::Point centre(const cv::Vec3f& c) { return {cvRound(c[0]), cvRound(c[1])}; }
double ratio(double a, double b) { return b ? a / b : 0; }
std::string trueType(int n) { return n < 0 ? "Unknown" : n == 0 ? "White" : n <= 8 ? "Solid" : "Striped"; }

struct Metrics {
    int gt = 0, tp = 0, fp = 0, fn = 0, recovered = 0, lost = 0, types = 0, numbers = 0;
    double error = 0;
};

void writeMetrics(std::ostream& out, const Metrics& m) {
    out << m.gt << ',' << m.tp + m.fp << ',' << m.tp << ',' << m.fp << ',' << m.fn << ','
        << ratio(m.tp, m.tp + m.fp) << ',' << ratio(m.tp, m.gt) << ','
        << ratio(2 * m.tp, 2 * m.tp + m.fp + m.fn) << ',' << ratio(m.error, m.tp) << ','
        << m.recovered << ',' << m.lost << ',' << m.types << ',' << m.numbers << ','
        << ratio(m.types, m.tp) << ',' << ratio(m.numbers, m.tp) << ',' << ratio(m.numbers, m.gt);
}

cv::Mat watershedView(const cv::Mat& image, const cv::Mat& labels) {
    cv::Mat result = image.clone();
    if (labels.empty())
        return result;
    for (int y = 0; y < labels.rows; ++y)
        for (int x = 0; x < labels.cols; ++x) {
            const int label = labels.at<int>(y, x);
            if (label == -1)
                result.at<cv::Vec3b>(y, x) = {255, 255, 255};
            else if (label > 1) {
                const cv::Vec3b colour(60 + label * 73 % 196, 60 + label * 137 % 196, 60 + label * 193 % 196);
                auto& pixel = result.at<cv::Vec3b>(y, x);
                for (int c = 0; c < 3; ++c)
                    pixel[c] = cv::saturate_cast<uchar>(0.45 * pixel[c] + 0.55 * colour[c]);
            }
        }
    return result;
}

void diagnostics(const cv::Mat& image, const SegmentationGuidedResult& result,
                  const std::filesystem::path& directory) {
    cv::Mat windows = image.clone();
    for (size_t r = 0; r < result.clusters.size(); ++r) {
        const auto& region = result.clusters[r];
        cv::rectangle(windows, region.core, {0, 255, 255}, 2);
        cv::rectangle(windows, region.crop, {255, 200, 0}, 1);
        cv::putText(windows, std::to_string(r + 1), region.crop.tl() + cv::Point(5, 15),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, {255, 255, 255}, 1);
        cv::Mat raw = image(region.crop).clone(), accepted = raw.clone(), refined = raw.clone();
        for (const auto& candidate : result.proposals) {
            if (candidate.region != static_cast<int>(r + 1))
                continue;
            const auto point = centre(candidate.circle) - region.crop.tl();
            cv::circle(raw, point, cvRound(candidate.circle[2]), {0, 180, 255}, 1);
            if (candidate.accepted) {
                cv::circle(accepted, point, cvRound(candidate.circle[2]), {0, 255, 255}, 1);
                if (candidate.watershedAccepted)
                    cv::circle(refined, point, cvRound(candidate.circle[2]), {0, 255, 255}, 1);
            }
        }
        save(directory / ("region_" + std::to_string(r + 1) + "_diagnostics.png"), combineImages({
            {"Original / segmentation-selected window", image(region.crop)},
            {"Foreground, no Hough cores", result.foreground(region.crop)},
            {"Raw permissive Hough candidates", raw},
            {"Accepted arc-supported local candidates", accepted},
            {"Watershed labels / candidate-centre seeds", watershedView(image(region.crop), region.labels)},
            {"Local candidates retained after watershed", refined}}, 3));
    }
    save(directory / "segmentation_routing.png", combineImages({{"Original / yellow cores, blue windows", windows},
        {"Foreground before Hough", result.foreground}, {"Interior routed away from standard Hough", result.routingMask}}, 1));
}

cv::Mat matches(const cv::Mat& image, const std::vector<cv::Vec3f>& circles,
                const EvaluationResult& matching, const std::vector<LabelledBall>& truth,
                const std::vector<cv::Point2f>& sourceGT) {
    cv::Mat view = image.clone();
    for (size_t i = 0; i < circles.size(); ++i) {
        const bool matched = matching.predictionToGroundTruth[i] >= 0;
        const cv::Scalar colour = matched ? cv::Scalar(70, 240, 70) : cv::Scalar(80, 80, 255);
        cv::circle(view, centre(circles[i]), cvRound(circles[i][2]), colour, 2);
        cv::putText(view, std::to_string(i + 1), centre(circles[i]) + cv::Point(3, -3),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, colour, 1);
    }
    for (size_t i = 0; i < truth.size(); ++i)
        if (matching.groundTruthToPrediction[i] < 0) {
            const cv::Point point(cvRound(sourceGT[i].x), cvRound(sourceGT[i].y));
            cv::drawMarker(view, point, {255, 150, 0}, cv::MARKER_CROSS, 18, 2);
            cv::putText(view, "miss " + std::to_string(truth[i].number), point + cv::Point(5, 15),
                        cv::FONT_HERSHEY_SIMPLEX, 0.45, {255, 150, 0}, 1);
        }
    return view;
}

void parameters(const std::filesystem::path& directory, const SegmentationHoughOptions& p) {
    auto csv = output(directory / "parameters.csv");
    csv << "parameter,value\n";
    const std::vector<std::pair<std::string, double>> values = {
        {"clothTileSize", p.clothTileSize}, {"minimumClothSamples", p.minimumClothSamples},
        {"clothMinimumDistance", p.clothMinimumDistance}, {"clothMaximumDistance", p.clothMaximumDistance},
        {"clothMADMultiplier", p.clothMADMultiplier}, {"tableInsetKernel", p.tableInsetKernel},
        {"closingKernel", p.closingKernel}, {"minimumComponentArea", p.minimumComponentArea},
        {"clusterAreaRatio", p.clusterAreaRatio}, {"clusterAspectRatio", p.clusterAspectRatio},
        {"elongatedMinimumAreaRatio", p.elongatedMinimumAreaRatio}, {"cropPaddingRadii", p.cropPaddingRadii},
        {"routingOverlapRadii", p.routingOverlapRadii}, {"localAccumulatorThreshold", p.localAccumulatorThreshold},
        {"localMinCentreDistance", p.localMinCentreDistance}, {"localMinRadiusRatio", p.localMinRadiusRatio},
        {"localMaxRadiusRatio", p.localMaxRadiusRatio}, {"absoluteMinLocalRadius", p.absoluteMinLocalRadius},
        {"absoluteMaxLocalRadius", p.absoluteMaxLocalRadius}, {"angularSamples", p.angularSamples},
        {"edgeTolerance", p.edgeTolerance}, {"minimumRadialAlignment", p.minimumRadialAlignment},
        {"minimumEdgeSupport", p.minimumEdgeSupport}, {"minimumArcDegrees", p.minimumArcDegrees},
        {"minimumForegroundSupport", p.minimumForegroundSupport}, {"duplicateDistanceRadii", p.duplicateDistanceRadii},
        {"watershedMinimumAreaRatio", p.watershedMinimumAreaRatio}, {"watershedMaximumAreaRatio", p.watershedMaximumAreaRatio}};
    for (const auto& v : values)
        csv << v.first << ',' << v.second << '\n';
}

} // namespace

void runSegmentationGuidedExperiment(const std::filesystem::path& directory) {
    std::filesystem::create_directories(directory);
    const SegmentationHoughOptions options;
    parameters(directory, options);
    auto metrics = output(directory / "metrics.csv");
    metrics << "method,split,image,groundTruth,detections,TP,FP,FN,precision,recall,F1,meanError,"
        "recovered,lost,typeCorrect,numberCorrect,typeAccuracy,numberAccuracy,endToEndNumberRecall\n";
    auto regions = output(directory / "regions.csv");
    regions << "image,region,coreX,coreY,coreWidth,coreHeight,cropX,cropY,cropWidth,cropHeight,"
        "expectedRadius,minRadius,maxRadius,emptySearchFallback\n";
    auto proposals = output(directory / "proposals.csv");
    proposals << "image,region,x,y,radius,edgeSupport,arcDegrees,foregroundSupport,quality,accepted,reason,"
        "guidedIndex,watershedAreaRatio,watershedAccepted,watershedReason\n";
    auto predictions = output(directory / "predictions.csv");
    predictions << "method,image,candidate,origin,x,y,radius,tableX,tableY,trueNumber,type,number,typeCorrect,numberCorrect,error\n";
    auto changes = output(directory / "changes.csv");
    changes << "method,image,trueNumber,baselineMatched,methodMatched,change\n";
    auto coverage = output(directory / "coverage.csv");
    coverage << "image,trueNumber,baselineMatched,insideClusterCrop,insideRoutedInterior\n";
    std::map<std::pair<std::string, std::string>, Metrics> totals;
    std::vector<std::string> changeDescriptions;
    for (const auto& name : config::allImages()) {
        const auto image = cv::imread(config::mediaDirectory + name);
        if (image.empty())
            throw std::runtime_error("Could not load " + name);
        // Cluster construction in this function precedes its first Hough call.
        const auto result = detectSegmentationGuided(image, options);
        const auto truth = loadLabelledGroundTruth(config::groundTruthPath, name);
        if (truth.empty())
            throw std::runtime_error("Missing ground truth for " + name);
        std::vector<cv::Point2f> gt, sourceGT;
        for (const auto& t : truth)
            gt.push_back(t.centre);
        cv::perspectiveTransform(gt, sourceGT, getTableTransformation().inv());
        const auto baseline = evaluateDetections(transformCentres(result.baseline), gt, config::matchTolerance);
        const auto imageDirectory = directory / std::filesystem::path(name).stem();
        std::filesystem::create_directories(imageDirectory);
        diagnostics(image, result, imageDirectory);
        for (size_t g = 0; g < truth.size(); ++g) {
            const cv::Point point(cvRound(sourceGT[g].x), cvRound(sourceGT[g].y));
            bool inside = false;
            for (const auto& region : result.clusters)
                inside |= region.crop.contains(point);
            const bool routed = point.x >= 0 && point.y >= 0 && point.x < image.cols && point.y < image.rows &&
                result.routingMask.at<uchar>(point);
            coverage << name << ',' << truth[g].number << ',' << (baseline.groundTruthToPrediction[g] >= 0)
                << ',' << inside << ',' << routed << '\n';
        }
        for (size_t i = 0; i < result.clusters.size(); ++i) {
            const auto& r = result.clusters[i];
            regions << name << ',' << i + 1 << ',' << r.core.x << ',' << r.core.y << ',' << r.core.width << ','
                << r.core.height << ',' << r.crop.x << ',' << r.crop.y << ',' << r.crop.width << ',' << r.crop.height
                << ',' << r.expectedRadius << ',' << r.minRadius << ',' << r.maxRadius << ',' << r.usedEmptySearchFallback << '\n';
        }
        for (const auto& p : result.proposals)
            proposals << name << ',' << p.region << ',' << p.circle[0] << ',' << p.circle[1] << ',' << p.circle[2]
                << ',' << p.edgeSupport << ',' << p.arcDegrees << ',' << p.foregroundSupport << ',' << p.quality
                << ',' << p.accepted << ',' << p.reason << ',' << p.guidedIndex << ',' << p.watershedAreaRatio
                << ',' << p.watershedAccepted << ',' << p.watershedReason << '\n';
        const std::vector<std::pair<std::string, std::vector<cv::Vec3f>>> methods = {
            {"hough", result.baseline}, {"segmentation_guided", result.guided},
            {"segmentation_guided_ws", result.watershed}};
        std::vector<ImagePanel> views;
        std::vector<EvaluationResult> outcomes;
        for (const auto& method : methods) {
            const auto points = transformCentres(method.second);
            const auto evaluated = evaluateDetections(points, gt, config::matchTolerance);
            outcomes.push_back(evaluated);
            Metrics m;
            m.gt = static_cast<int>(truth.size());
            m.tp = evaluated.TP; m.fp = evaluated.FP; m.fn = evaluated.FN;
            m.error = evaluated.meanError * evaluated.TP;
            for (size_t g = 0; g < truth.size(); ++g) {
                const bool old = baseline.groundTruthToPrediction[g] >= 0, now = evaluated.groundTruthToPrediction[g] >= 0;
                m.recovered += !old && now; m.lost += old && !now;
                const std::string change = !old && now ? "recovered" : old && !now ? "lost" : "unchanged";
                changes << method.first << ',' << name << ',' << truth[g].number << ',' << old << ',' << now << ',' << change << '\n';
                if (change != "unchanged")
                    changeDescriptions.push_back(method.first + " / " + name + " / ball " + std::to_string(truth[g].number) + ": " + change);
            }
            for (size_t i = 0; i < method.second.size(); ++i) {
                const auto& c = method.second[i];
                const auto ball = extractBall(image, c);
                const auto clusters = segmentBallKMeans(ball, 6);
                const auto recognition = ccaRecognitionBase();
                const auto baseType = classifyExperimentalType(clusters, recognition);
                CCAParameters cca;
                cca.cue = cca.stripe = false;
                cca.solidMaxBoundary = 20;
                const auto f = measureExperimentalCCA(ball, clusters, recognition, cca);
                const auto type = classifyExperimentalCCA(baseType, f, cca);
                const int number = classifyExperimentalNumber(clusters, type.type, recognition);
                const int matched = evaluated.predictionToGroundTruth[i];
                const int actual = matched < 0 ? -1 : truth[matched].number;
                const bool local = std::any_of(result.proposals.begin(), result.proposals.end(), [&](const auto& p) {
                    return p.accepted && p.circle == c;
                });
                predictions << method.first << ',' << name << ',' << i + 1 << ','
                    << (method.first == "hough" || !local ? "standard_or_fallback" : "permissive") << ','
                    << c[0] << ',' << c[1] << ',' << c[2] << ',' << points[i].x << ',' << points[i].y << ','
                    << actual << ',' << type.type << ',' << number << ',';
                if (matched >= 0) {
                    const bool correctType = type.type == trueType(actual), correctNumber = number == actual;
                    m.types += correctType; m.numbers += correctNumber;
                    predictions << correctType << ',' << correctNumber << ',' << cv::norm(points[i] - gt[matched]);
                } else
                    predictions << ",,";
                predictions << '\n';
            }
            const std::string split = name == config::tuningImage ? "reference" : "evaluation";
            metrics << method.first << ',' << split << ',' << name << ',';
            writeMetrics(metrics, m);
            metrics << '\n';
            auto& total = totals[{method.first, split}];
            total.gt += m.gt; total.tp += m.tp; total.fp += m.fp; total.fn += m.fn; total.error += m.error;
            total.types += m.types; total.numbers += m.numbers; total.recovered += m.recovered; total.lost += m.lost;
            views.push_back({method.first + " / TP " + std::to_string(m.tp) + " FP " + std::to_string(m.fp) + " FN "
                + std::to_string(m.fn) + " / green matches, red FP, blue missed GT", matches(image, method.second, evaluated, truth, sourceGT)});
        }
        save(imageDirectory / "detection_comparison.png", combineImages(views, 1));
        // Annotation-centred crops are diagnostics only, created after all methods finish.
        const std::vector<std::string> shortNames = {"Hough", "Guided", "Guided + watershed"};
        for (size_t g = 0; g < truth.size(); ++g) {
            if (baseline.groundTruthToPrediction[g] >= 0 ||
                std::none_of(outcomes.begin(), outcomes.end(), [&](const auto& e) {
                    return e.groundTruthToPrediction[g] >= 0;
                }))
                continue;
            const cv::Rect crop = cv::Rect(cvRound(sourceGT[g].x) - 90, cvRound(sourceGT[g].y) - 90, 181, 181)
                & cv::Rect(0, 0, image.cols, image.rows);
            std::vector<ImagePanel> zoom;
            for (size_t m = 0; m < views.size(); ++m)
                zoom.push_back({shortNames[m] + (outcomes[m].groundTruthToPrediction[g] >= 0 ? " / recovered" : " / missed"),
                                views[m].image(crop)});
            save(imageDirectory / ("recovered_ball_" + std::to_string(truth[g].number) + "_comparison.png"), combineImages(zoom, 3));
        }
        std::cout << name << ": " << result.clusters.size() << " segmentation-selected windows / "
            << result.baseline.size() << " baseline / " << result.guided.size() << " guided / "
            << result.watershed.size() << " after watershed\n" << std::flush;
    }
    auto summary = output(directory / "summary.md");
    summary << "# Segmentation-guided Hough experiment\n\n"
        << "Foreground and cluster windows are constructed before Hough. Ground truth is used only after detection. "
        << "This is one fixed configuration, not a grid search.\n\n"
        << "| Method | Split | TP | FP | FN | Precision | Recall | F1 | Mean error | Recovered/lost | Correct type/number |\n"
        << "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n";
    for (const auto& row : totals) {
        const auto& m = row.second;
        metrics << row.first.first << ',' << row.first.second << ",ALL,";
        writeMetrics(metrics, m);
        metrics << '\n';
        summary << '|' << row.first.first << '|' << row.first.second << '|' << m.tp << '|' << m.fp << '|' << m.fn
            << '|' << ratio(m.tp, m.tp + m.fp) << '|' << ratio(m.tp, m.gt) << '|'
            << ratio(2 * m.tp, 2 * m.tp + m.fp + m.fn) << '|' << ratio(m.error, m.tp) << '|'
            << m.recovered << '/' << m.lost << '|' << m.types << '/' << m.numbers << "|\n";
    }
    summary << "\nChanged annotation matches:\n\n";
    for (const auto& change : changeDescriptions)
        summary << "- " << change << '\n';
    summary << "\nStandard Hough is evaluated on the unmodified table-masked image, then its cluster-interior "
        << "centres are routed to local permissive searches. Boundary overlap is retained to avoid routing gaps. "
        << "Local candidates require radial Canny-edge support and a visible circular arc. Duplicate suppression "
        << "uses 0.8 times the smaller radius, allowing overlapping balls with distinct centres. "
        << "If a local region accepts no candidate, its available standard detections are retained as an explicit fallback.\n\n"
        << "The optional watershed pass uses accepted circles as markers and rejects local candidates whose basin "
        << "foreground area falls outside [0.15,2.0] expected disk areas. It does not invent centres or alter radii. "
        << "Standard/fallback detections remain untouched. Comparing guided and guided_ws measures this check's contribution.\n\n"
        << "Recognition stays fixed to configuration 2153 plus selected solid-only CCA, with circular extraction masks. "
        << "Recognition scores on Table1-6 are exploratory because its thresholds were previously tuned there. "
        << "No pixel-level segmentation accuracy is claimed. Main application files are unchanged. "
        << "parameters.csv, regions.csv and proposals.csv record settings, routes, arc evidence and watershed checks.\n";
}

} // namespace experiments
