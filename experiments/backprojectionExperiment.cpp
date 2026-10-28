#include "clothBackprojection.h"
#include "config.h"
#include "detection/ballDetection.h"
#include "evaluation/evaluation.h"
#include "geometry/perspectiveTransform.h"
#include "visualisation/combineImages.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <tuple>

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

cv::Mat tableMask(cv::Size size) {
    cv::Mat mask = cv::Mat::zeros(size, CV_8UC1);
    cv::fillConvexPoly(mask, getTableCorners(), cv::Scalar(255));
    return mask;
}

std::vector<cv::Point2f> centres(const std::vector<LabelledBall>& truth) {
    std::vector<cv::Point2f> points;
    for (const auto& ball : truth)
        points.push_back(ball.centre);
    return points;
}

struct Counts {
    int gt = 0, tp = 0, fp = 0, fn = 0;
    double error = 0;
};

Counts counts(const EvaluationResult& e, size_t groundTruth) {
    return {static_cast<int>(groundTruth), e.TP, e.FP, e.FN, e.meanError * e.TP};
}

void writeCounts(std::ostream& out, const Counts& c) {
    out << c.gt << ',' << c.tp + c.fp << ',' << c.tp << ',' << c.fp << ',' << c.fn << ','
        << ratio(c.tp, c.tp + c.fp) << ',' << ratio(c.tp, c.gt) << ','
        << ratio(2 * c.tp, 2 * c.tp + c.fp + c.fn) << ',' << ratio(c.error, c.tp);
}

const char* metricsHeader = "groundTruth,detections,TP,FP,FN,precision,recall,F1,meanError";

struct Choice {
    int id = 0;
    ClothBackprojectionOptions p;
    Counts count;
    int rejectedTP = 0, rejectedFP = 0;
    bool validModel = false;
    bool safe() const { return validModel && rejectedTP == 0; }
};

const char* parameterHeader = "hBins,sBins,similarityThreshold,maximumClothFraction";
void parameters(std::ostream& out, const ClothBackprojectionOptions& p) {
    out << p.hBins << ',' << p.sBins << ',' << p.similarityThreshold << ',' << p.maximumClothFraction;
}

cv::Mat histogramView(const ClothBackprojectionModel& model) {
    if (model.histogram.empty())
        return {};
    cv::Mat bytes, enlarged, coloured;
    model.histogram.convertTo(bytes, CV_8U);
    cv::resize(bytes, enlarged, {bytes.cols * 12, bytes.rows * 12}, 0, 0, cv::INTER_NEAREST);
    cv::applyColorMap(enlarged, coloured, cv::COLORMAP_VIRIDIS);
    return coloured; // Rows are H bins, columns are S bins, on the stored [0,255] scale.
}

void debug(const std::filesystem::path& directory, const cv::Mat& image,
           const ClothBackprojectionModel& model, const ClothVerificationResult& verified,
           const std::vector<cv::Vec3f>& candidates, const ClothBackprojectionOptions& p) {
    save(directory / "sampling_mask.png", model.samplingMask);
    cv::Mat annotated = image.clone();
    for (size_t i = 0; i < candidates.size(); ++i) {
        const cv::Scalar colour = verified.scores[i].retained ? cv::Scalar(70, 240, 70) : cv::Scalar(80, 80, 255);
        cv::circle(annotated, centre(candidates[i]), cvRound(candidates[i][2]), colour, 2);
        cv::putText(annotated, std::to_string(i + 1), centre(candidates[i]) + cv::Point(3, -3),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, colour, 1);
    }
    save(directory / "verification.png", annotated);
    cv::FileStorage yaml((directory / "cloth_model.yml").string(), cv::FileStorage::WRITE);
    if (!yaml.isOpened())
        throw std::runtime_error("Could not write cloth histogram");
    yaml << "hBins" << p.hBins << "sBins" << p.sBins << "samples" << model.samples
         << "valid" << int(model.valid) << "status" << model.status << "histogram" << model.histogram;
    std::vector<ImagePanel> panels = {{"Sampling mask (candidate/pocket holes)", model.samplingMask}};
    if (model.valid) {
        cv::Mat high;
        cv::compare(model.similarity, p.similarityThreshold, high, cv::CMP_GE);
        save(directory / "similarity.png", model.similarity8U);
        save(directory / "high_similarity.png", high);
        save(directory / "histogram.png", histogramView(model));
        panels.push_back({"Cloth similarity / fixed 0-255 display", model.similarity8U});
        panels.push_back({"High-similarity pixels / fixed threshold", high});
    }
    panels.push_back({"Green retained / red rejected (not GT correctness)", annotated});
    save(directory / "diagnostics.png", combineImages(panels, 2));
}

cv::Mat matches(const cv::Mat& image, const std::vector<cv::Vec3f>& circles,
                const EvaluationResult& e, const std::vector<LabelledBall>& truth) {
    cv::Mat result = image.clone();
    for (size_t i = 0; i < circles.size(); ++i) {
        const cv::Scalar colour = e.predictionToGroundTruth[i] >= 0 ? cv::Scalar(70, 240, 70) : cv::Scalar(80, 80, 255);
        cv::circle(result, centre(circles[i]), cvRound(circles[i][2]), colour, 2);
    }
    auto source = centres(truth);
    cv::perspectiveTransform(source, source, getTableTransformation().inv());
    for (size_t g = 0; g < truth.size(); ++g)
        if (e.groundTruthToPrediction[g] < 0) {
            const cv::Point point(cvRound(source[g].x), cvRound(source[g].y));
            cv::drawMarker(result, point, {255, 150, 0}, cv::MARKER_CROSS, 16, 2);
            cv::putText(result, "miss " + std::to_string(truth[g].number), point + cv::Point(4, 14),
                        cv::FONT_HERSHEY_SIMPLEX, 0.4, {255, 150, 0}, 1);
        }
    return result;
}

} // namespace

void runBackprojectionExperiment(const std::filesystem::path& directory) {
    std::filesystem::create_directories(directory / "screening");
    const cv::Mat screeningImage = cv::imread(config::mediaDirectory + config::tuningImage);
    if (screeningImage.empty())
        throw std::runtime_error("Could not load AllBalls for recall screening");
    const auto screeningTruth = loadLabelledGroundTruth(config::groundTruthPath, config::tuningImage);
    if (screeningTruth.empty())
        throw std::runtime_error("Missing AllBalls ground truth");
    const auto screenHough = runDetection(screeningImage, HoughParams{});
    const auto screenTable = tableMask(screeningImage.size());
    const auto screenPockets = makeClothPocketMask(screeningImage.size());
    const auto screenEvaluation = evaluateDetections(transformCentres(screenHough.filteredCircles),
        centres(screeningTruth), config::matchTolerance);
    auto gridCSV = output(directory / "grid_search.csv");
    gridCSV << "configId," << parameterHeader << ",modelValid,samples,rejectedTP,rejectedFP,recallSafe,"
            << metricsHeader << '\n';
    auto screenScores = output(directory / "screening_scores.csv");
    screenScores << "configId,candidate,trueNumber,validPixels,coverage,meanSimilarity,clothFraction,retained,reason\n";
    std::vector<Choice> choices;
    std::map<std::pair<int, int>, ClothBackprojectionModel> models;
    for (int h : {18, 30})
        for (int s : {16, 32}) {
            ClothBackprojectionOptions p;
            p.hBins = h; p.sBins = s;
            auto model = estimateClothBackprojection(screeningImage, screenTable, screenPockets, screenHough.rawCircles, p);
            if (model.valid) {
                const std::string prefix = "H" + std::to_string(h) + "_S" + std::to_string(s);
                save(directory / "screening" / (prefix + "_similarity.png"), model.similarity8U);
                save(directory / "screening" / (prefix + "_histogram.png"), histogramView(model));
            }
            models.emplace(std::make_pair(h, s), std::move(model));
        }
    for (int h : {18, 30})
        for (int s : {16, 32})
            for (double similarity : {0.6, 0.7, 0.8})
                for (double fraction : {0.7, 0.8, 0.9}) {
                    Choice choice;
                    choice.id = static_cast<int>(choices.size() + 1);
                    choice.p.hBins = h; choice.p.sBins = s;
                    choice.p.similarityThreshold = similarity; choice.p.maximumClothFraction = fraction;
                    const auto& model = models.at({h, s});
                    choice.validModel = model.valid;
                    const auto verified = verifyClothCandidates(model, screenTable, screenHough.filteredCircles, choice.p);
                    for (size_t i = 0; i < verified.scores.size(); ++i) {
                        const auto& score = verified.scores[i];
                        const int match = screenEvaluation.predictionToGroundTruth[i];
                        if (!score.retained)
                            (match >= 0 ? choice.rejectedTP : choice.rejectedFP)++;
                        screenScores << choice.id << ',' << i + 1 << ',' << (match < 0 ? -1 : screeningTruth[match].number)
                            << ',' << score.validPixels << ',' << score.coverage << ',' << score.meanSimilarity
                            << ',' << score.clothFraction << ',' << score.retained << ',' << score.reason << '\n';
                    }
                    choice.count = counts(evaluateDetections(transformCentres(verified.retained), centres(screeningTruth),
                                                               config::matchTolerance), screeningTruth.size());
                    gridCSV << choice.id << ',';
                    parameters(gridCSV, choice.p);
                    gridCSV << ',' << model.valid << ',' << model.samples << ',' << choice.rejectedTP << ','
                            << choice.rejectedFP << ',' << choice.safe() << ',';
                    writeCounts(gridCSV, choice.count);
                    gridCSV << '\n';
                    choices.push_back(choice);
                }
    // Predeclared conservative ordering, not precision ranking on held-out images.
    auto preference = [](const Choice& c) {
        return std::make_tuple(c.safe(), c.p.similarityThreshold, c.p.maximumClothFraction,
                               -c.p.hBins, -c.p.sBins, -c.id);
    };
    const Choice best = *std::max_element(choices.begin(), choices.end(), [&](const Choice& a, const Choice& b) {
        return preference(a) < preference(b);
    });
    const int safeCount = static_cast<int>(std::count_if(choices.begin(), choices.end(), [](const Choice& c) { return c.safe(); }));
    ClothBackprojectionOptions selected = best.safe() ? best.p : ClothBackprojectionOptions{};
    selected.enabled = best.safe();
    const int selectedId = best.safe() ? best.id : 0;
    auto selection = output(directory / "selection.csv");
    selection << "configId,enabled," << parameterHeader
        << ",samplingInsetKernel,exclusionMargin,minimumSamples,minimumCircleCoverage,safeConfigurations,selectionRule\n";
    selection << selectedId << ',' << selected.enabled << ',';
    parameters(selection, selected);
    selection << ',' << selected.samplingInsetKernel << ',' << selected.exclusionMargin << ',' << selected.minimumSamples
              << ',' << selected.minimumCircleCoverage << ',' << safeCount
              << ",AllBalls_recall_screen_then_high_similarity_high_fraction_coarse_bins\n";
    selection.close(); // Frozen on disk before any Table1-6 image/label evaluation.
    gridCSV.close(); screenScores.close();
    std::cout << "AllBalls recall-safe configurations: " << safeCount << "/36. Frozen config "
              << selectedId << " (enabled=" << selected.enabled << ") before Table1-6 evaluation.\n" << std::flush;

    auto metrics = output(directory / "evaluation.csv");
    metrics << "method,split,image," << metricsHeader << '\n';
    auto scoresCSV = output(directory / "candidate_scores.csv");
    scoresCSV << "image,candidate,rawCandidate,trueNumber,baselineMatched,x,y,radius,validPixels,coverage,"
        "meanSimilarity,clothFraction,retained,reason\n";
    auto predictions = output(directory / "predictions.csv");
    predictions << "method,image,candidate,x,y,radius,trueNumber,error\n";
    auto changes = output(directory / "changes.csv");
    changes << "image,trueNumber,baselineMatched,filteredMatched,change\n";
    auto rejections = output(directory / "rejections.csv");
    rejections << "split,image,rejectedBaselineTP,rejectedBaselineFP,netTPChange,netFPChange\n";
    auto modelCSV = output(directory / "models.csv");
    modelCSV << "image,hBins,sBins,samples,modelValid,status,rawCandidates,pocketFilteredCandidates,retainedCandidates\n";
    std::map<std::pair<std::string, std::string>, Counts> totals;
    std::map<std::string, std::pair<int, int>> rejectedTotals;
    std::vector<std::string> rejectedDescriptions;
    for (const auto& name : config::allImages()) {
        const cv::Mat image = name == config::tuningImage ? screeningImage : cv::imread(config::mediaDirectory + name);
        if (image.empty())
            throw std::runtime_error("Could not load " + name);
        const auto hough = name == config::tuningImage ? screenHough : runDetection(image, HoughParams{});
        const auto table = tableMask(image.size());
        const auto model = name == config::tuningImage ? models.at({selected.hBins, selected.sBins})
            : estimateClothBackprojection(image, table, makeClothPocketMask(image.size()), hough.rawCircles, selected);
        const auto verified = verifyClothCandidates(model, table, hough.filteredCircles, selected);
        // Table1-6 labels are loaded only after their filtering decisions are complete.
        const auto truth = name == config::tuningImage ? screeningTruth : loadLabelledGroundTruth(config::groundTruthPath, name);
        if (truth.empty())
            throw std::runtime_error("Missing labels for " + name);
        const auto gt = centres(truth);
        const auto baseline = evaluateDetections(transformCentres(hough.filteredCircles), gt, config::matchTolerance);
        const auto filtered = evaluateDetections(transformCentres(verified.retained), gt, config::matchTolerance);
        const std::string split = name == config::tuningImage ? "screening" : "evaluation";
        int rejectedTP = 0, rejectedFP = 0;
        for (size_t i = 0; i < verified.scores.size(); ++i) {
            const auto& score = verified.scores[i];
            const int match = baseline.predictionToGroundTruth[i];
            const auto raw = std::find(hough.rawCircles.begin(), hough.rawCircles.end(), hough.filteredCircles[i]);
            if (raw == hough.rawCircles.end())
                throw std::runtime_error("Pocket-filtered circle is not an original Hough candidate");
            const auto& c = hough.filteredCircles[i];
            scoresCSV << name << ',' << i + 1 << ',' << raw - hough.rawCircles.begin() + 1 << ','
                << (match < 0 ? -1 : truth[match].number) << ',' << (match >= 0) << ',' << c[0] << ',' << c[1] << ',' << c[2]
                << ',' << score.validPixels << ',' << score.coverage << ',' << score.meanSimilarity << ','
                << score.clothFraction << ',' << score.retained << ',' << score.reason << '\n';
            if (!score.retained) {
                (match >= 0 ? rejectedTP : rejectedFP)++;
                rejectedDescriptions.push_back(name + " / candidate " + std::to_string(i + 1) +
                    (match < 0 ? " / baseline false detection" : " / genuine ball " + std::to_string(truth[match].number)));
            }
        }
        rejections << split << ',' << name << ',' << rejectedTP << ',' << rejectedFP << ','
            << filtered.TP - baseline.TP << ',' << filtered.FP - baseline.FP << '\n';
        rejectedTotals[split].first += rejectedTP;
        rejectedTotals[split].second += rejectedFP;
        for (size_t g = 0; g < truth.size(); ++g) {
            const bool old = baseline.groundTruthToPrediction[g] >= 0, now = filtered.groundTruthToPrediction[g] >= 0;
            changes << name << ',' << truth[g].number << ',' << old << ',' << now << ','
                << (old && !now ? "lost" : !old && now ? "recovered_by_reassignment" : "unchanged") << '\n';
        }
        modelCSV << name << ',' << selected.hBins << ',' << selected.sBins << ',' << model.samples << ','
            << model.valid << ',' << model.status << ',' << hough.rawCircles.size() << ','
            << hough.filteredCircles.size() << ',' << verified.retained.size() << '\n';
        const auto imageDirectory = directory / std::filesystem::path(name).stem();
        std::filesystem::create_directories(imageDirectory);
        debug(imageDirectory, image, model, verified, hough.filteredCircles, selected);
        std::vector<ImagePanel> views;
        for (const auto& method : std::vector<std::pair<std::string, std::vector<cv::Vec3f>>>{
                {"A_hough", hough.rawCircles}, {"B_pocket", hough.filteredCircles}, {"C_backprojection", verified.retained}}) {
            const auto points = transformCentres(method.second);
            const auto e = evaluateDetections(points, gt, config::matchTolerance);
            const auto n = counts(e, truth.size());
            metrics << method.first << ',' << split << ',' << name << ',';
            writeCounts(metrics, n);
            metrics << '\n';
            auto& total = totals[{method.first, split}];
            total.gt += n.gt; total.tp += n.tp; total.fp += n.fp; total.fn += n.fn; total.error += n.error;
            for (size_t i = 0; i < method.second.size(); ++i) {
                const auto& c = method.second[i];
                const int match = e.predictionToGroundTruth[i];
                predictions << method.first << ',' << name << ',' << i + 1 << ',' << c[0] << ',' << c[1] << ','
                    << c[2] << ',' << (match < 0 ? -1 : truth[match].number) << ',';
                if (match >= 0)
                    predictions << cv::norm(points[i] - gt[match]);
                predictions << '\n';
            }
            views.push_back({method.first + " / TP " + std::to_string(n.tp) + " FP " + std::to_string(n.fp)
                + " FN " + std::to_string(n.fn) + " / green match, red FP, blue missed GT", matches(image, method.second, e, truth)});
        }
        save(imageDirectory / "detection_comparison.png", combineImages(views, 1));
        std::cout << name << ": " << hough.filteredCircles.size() << " baseline -> " << verified.retained.size()
            << " retained; rejected " << rejectedTP << " matched balls, " << rejectedFP << " false detections\n" << std::flush;
    }
    auto summary = output(directory / "summary.md");
    summary << "# Cloth histogram backprojection verification\n\n"
        << "Screened 36 configurations on AllBalls only. Recall-safe: " << safeCount << "/36. Frozen config: "
        << selectedId << ", enabled=" << selected.enabled << ", H bins=" << selected.hBins << ", S bins=" << selected.sBins
        << ", similarity threshold=" << selected.similarityThreshold << ", maximum cloth fraction=" << selected.maximumClothFraction << ".\n\n"
        << "Selection preserves every originally matched AllBalls candidate, then uses a predeclared preference: "
        << "higher similarity threshold, higher allowed cloth fraction, fewer H bins, fewer S bins, earlier ID. "
        << "If no valid recall-safe model exists, verification is disabled. AllBalls has no post-pocket false "
        << "detections in this run, so this is recall screening, not learned precision optimisation. "
        << "selection.csv is frozen before Table1-6 images/labels are evaluated; only that setting is evaluated there.\n\n"
        << "| Method | Split | TP | FP | FN | Precision | Recall | F1 | Mean error |\n"
        << "|---|---|---:|---:|---:|---:|---:|---:|---:|\n";
    for (const auto& row : totals) {
        const auto& n = row.second;
        metrics << row.first.first << ',' << row.first.second << ",ALL,";
        writeCounts(metrics, n);
        metrics << '\n';
        summary << '|' << row.first.first << '|' << row.first.second << '|' << n.tp << '|' << n.fp << '|' << n.fn
            << '|' << ratio(n.tp, n.tp + n.fp) << '|' << ratio(n.tp, n.gt) << '|'
            << ratio(2 * n.tp, 2 * n.tp + n.fp + n.fn) << '|' << ratio(n.error, n.tp) << "|\n";
    }
    summary << "\nRejected candidates:\n\n";
    if (rejectedDescriptions.empty())
        summary << "None.\n";
    for (const auto& description : rejectedDescriptions)
        summary << "- " << description << '\n';
    const auto removed = rejectedTotals["evaluation"];
    summary << "\nTable1-6: " << removed.first << " baseline matched candidates rejected, " << removed.second
        << " baseline false detections rejected.\n\n"
        << "Cloth models are re-estimated per image. Sampling: original table mask eroded by an 11x11 ellipse, "
        << "exclude all raw Hough circles with a four-pixel radius margin and known 25-table-pixel pocket regions. "
        << "At least 1000 sampling pixels are required. No H/S eligibility threshold or GT exclusions are used.\n\n"
        << "H/S ranges: [0,180), [0,256). Histogram min/max normalised to [0,255]; 8-bit backprojection "
        << "then converted to float by dividing by 255. No per-map renormalisation. Values are relative "
        << "histogram similarities, not probabilities. Scoring uses the original circle intersected with the "
        << "table/image, never the sampling mask. Less than 80% circle coverage causes conservative retention. "
        << "High-similarity pixels use score >= threshold; rejection uses clothFraction > maximum. "
        << "Invalid/insufficient models retain all candidates. No edge criterion was added in this first experiment.\n\n"
        << "The filter never moves, adds or changes a circle radius. Main localisation/recognition code is unchanged. "
        << "Table1-6 were excluded from this filter's tuning but informed previous project experiments, "
        << "so these results are a frozen-setting comparison, not pristine independent test evidence.\n";
}

} // namespace experiments
