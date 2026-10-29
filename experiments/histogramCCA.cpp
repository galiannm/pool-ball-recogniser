#include "histogramCCA.h"
#include "ballColourHistograms.h"
#include "ccaGridSearch.h"
#include "recognitionGridSearch.h"
#include "config.h"
#include "detection/ballDetection.h"
#include "evaluation/evaluation.h"
#include "geometry/perspectiveTransform.h"
#include "io/finalResults.h"
#include "io/output.h"
#include "visualisation/combineImages.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
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
double ratio(double a, double b) {return b ? a / b : 0;}
std::string label(int n) {return n < 0 ? "Unknown" : n == 0 ? "Cue" : std::to_string(n);}
std::string typeFor(int n) {return n == 0 ? "White" : n <= 8 ? "Solid" : "Striped";}

// Load the exact previous model file. Do not retrain or load training annotations.
BallColourModel loadFrozenHSV(const std::filesystem::path& path) {
    if (!std::filesystem::exists(path))
        throw std::runtime_error("Missing frozen histograms. Run BallExperiments --colour-histograms first: " + path.string());
    cv::FileStorage input(path.string(), cv::FileStorage::READ);
    if (!input.isOpened() || std::string(input["trainingImage"]) != "AllBalls.jpeg")
        throw std::runtime_error("Invalid AllBalls histogram model file");
    BallColourModel model{BallHistogramSpace::HSV, BallHistogramOptions{}, {}, {}};
    bool found = false;
    std::array<bool, 8> seen{};
    for (const auto& node : input["models"]) {
        if (std::string(node["space"]) != "hsv_histogram") continue;
        if (found) throw std::runtime_error("Repeated HSV model");
        found = true;
        for (const auto& entry : node["histograms"]) {
            const int colour = int(entry["colour"]);
            if (colour < 1 || colour > 8 || seen[colour - 1]) throw std::runtime_error("Invalid/repeated histogram colour");
            seen[colour - 1] = true;
            entry["histogram"] >> model.histograms[colour - 1];
            model.examples[colour - 1] = int(entry["examples"]);
            const auto& h = model.histograms[colour - 1];
            if (h.type() != CV_32F || h.dims != 3 || h.size[0] != 18 || h.size[1] != 8 || h.size[2] != 8 ||
                model.examples[colour - 1] != (colour == 8 ? 1 : 2) || !cv::checkRange(h) ||
                std::abs(cv::sum(h)[0] - 1) > 1e-5)
                throw std::runtime_error("Unexpected histogram shape, examples or normalisation");
            for (size_t i = 0; i < h.total(); ++i)
                if (h.ptr<float>()[i] < 0) throw std::runtime_error("Negative histogram mass");
        }
    }
    if (!found || std::find(seen.begin(), seen.end(), false) != seen.end())
        throw std::runtime_error("Incomplete frozen HSV model");
    return model;
}

struct Counts {
    int gt = 0, tp = 0, fp = 0, fn = 0, type = 0, number = 0, oracle = 0;
    double errorSum = 0;
    void add(const Counts& c) {
        gt += c.gt; tp += c.tp; fp += c.fp; fn += c.fn;
        type += c.type; number += c.number; oracle += c.oracle; errorSum += c.errorSum;
    }
};
void writeCounts(std::ostream& out, const Counts& c) {
    out << c.gt << ',' << c.tp + c.fp << ',' << c.tp << ',' << c.fp << ',' << c.fn << ','
        << ratio(c.tp, c.tp + c.fp) << ',' << ratio(c.tp, c.gt) << ','
        << ratio(2 * c.tp, 2 * c.tp + c.fp + c.fn) << ',' << ratio(c.errorSum, c.tp) << ','
        << c.type << ',' << c.number << ',' << c.oracle << ',' << ratio(c.type, c.tp) << ','
        << ratio(c.number, c.tp) << ',' << ratio(c.number, c.gt) << ',' << ratio(c.number, c.tp + c.fp);
}

std::string status(bool same, bool wasCorrect, bool nowCorrect, bool matched) {
    if (same) return "unchanged";
    if (!matched) return "unmatched";
    if (!wasCorrect && nowCorrect) return "corrected";
    if (wasCorrect && !nowCorrect) return "regressed";
    return "changed_still_wrong";
}

cv::Mat enlarged(const cv::Mat& image) {
    cv::Mat resized;
    cv::resize(image, resized, {240, 240}, 0, 0, cv::INTER_NEAREST);
    return resized;
}
} // namespace

static void runHistogramCCAComparison(const std::filesystem::path& directory, bool originalK2) {
    const std::vector<std::string> methods = originalK2
        ? std::vector<std::string>{"k2_hsv", "k2_original_cca_hsv"}
        : std::vector<std::string>{"k2_hsv", "k6_hsv", "k6_cca_hsv"};
    const std::vector<std::string> names = originalK2
        ? std::vector<std::string>{"K=2 + HSV histogram", "K=2 + original CCA + HSV histogram"}
        : std::vector<std::string>{"K=2 + HSV histogram", "K=6 + HSV histogram", "K=6 + solid CCA + HSV histogram"};
    const std::filesystem::path source = "output/experiments/ball_colour_histograms/training/models.yml";
    const auto model = loadFrozenHSV(source);
    const auto tuned = ccaRecognitionBase();
    CCAParameters ccaOptions;
    ccaOptions.cue = false;
    ccaOptions.stripe = false;
    ccaOptions.solidMaxBoundary = 20;
    std::filesystem::create_directories(directory);
    std::filesystem::copy_file(source, directory / "frozen_models.yml", std::filesystem::copy_options::overwrite_existing);
    auto configuration = output(directory / "configuration.txt");
    configuration << "Model source: " << source << " (exact file snapshot in frozen_models.yml). No retraining.\n"
        << "HSV colour histogram: 18x8x8; eroded-circle mask 3x3; exclude HSV V>=170,S<=165; "
        << "minimum 12 body pixels; neutral BGR chroma <15; [1,2,1]/4 smoothing; L1 normalisation; "
        << "1-Bhattacharyya similarity. Identical model, query mask and scores for all profiles.\n"
        << "K2 type: current production classifier; K=2 is explicit in the original-CCA comparison.\n";
    if (originalK2) configuration
        << "Original production CCA: light V>=170 S<=165; 8-connectivity; component cutoff 2%; ellipse boundary 5x5.\n"
        << "Cue: light>=85%, largest light>=80%, light boundary>=75%, zero eligible non-light components.\n"
        << "Stripe: largest light>=20%, light boundary>=20%, non-light>=10%.\n"
        << "Solid: non-light>=70%, light boundary<=12%, exactly one eligible non-light component.\n"
        << "Cue, stripe and solid rules enabled in their original order; otherwise baseline fallback.\n"
        << "Uses analyseConnectedComponents and classifyBallTypeWithCCA directly. No K6 or tuned CCA rules applied.\n";
    else configuration << "K6 type: frozen 2153, K=6, light V>=170 S<=125, cue>=85% light, solid>=80% non-light.\n"
        << "CCA: frozen 20242, solid only, area cutoff 2%, 8-connectivity, ellipse 5x5 boundary; "
        << "non-light>=70%, light boundary<=20%, exactly one eligible non-light component; otherwise fallback.\n";
    configuration << "Cue maps to 0, Solid to colours 1..8, Striped to colours 1..7 plus 8; stripe+black is Unknown.\n"
        << "No hue thresholds or black S/V rule are used for number recognition here. "
        << "No parameters selected or changed after evaluation. Main application unchanged.\n";
    if (!originalK2) configuration
        << "AllBalls colour training does not remove the earlier Table1-6 tuning of K6/CCA; combined scores are exploratory.\n";
    configuration.close();
    auto metrics = output(directory / "metrics.csv");
    metrics << "split,image,method,groundTruth,detections,TP,FP,FN,precision,recall,F1,meanError,"
               "correctTypes,correctNumbers,oracleNumbers,typeAccuracy,numberAccuracy,numberRecall,numberPrecision\n";
    auto predictions = output(directory / "predictions.csv");
    predictions << "image,method,candidate,trueNumber,type,number,bodyColour,bodyPixels,bestScore,margin,oracleNumber,typeCorrect,numberCorrect\n";
    auto features = output(directory / "cca_features.csv");
    features << (originalK2 ? "image,candidate,k2Type,ccaType,decision,lightPercent,nonLightPercent,lightComponents,"
                           : "image,candidate,k6Type,ccaType,decision,lightPercent,nonLightPercent,lightComponents,")
             << "largestLightPercent,nonLightComponents,largestNonLightPercent,lightBoundaryCoverage\n";
    auto changes = output(directory / "changes.csv");
    changes << "image,comparison,candidate,trueNumber,beforeType,afterType,beforeNumber,afterNumber,typeChange,numberChange\n";
    auto colourScores = output(directory / "colour_scores.csv");
    colourScores << "image,candidate,colour,name,similarity\n";
    std::vector<Counts> totals(methods.size());
    using ConfusionKey = std::tuple<std::string, std::string, std::string, std::string, std::string>;
    std::map<ConfusionKey, int> confusion;
    std::map<std::pair<std::string, std::string>, int> changeCounts;
    const std::vector<std::pair<int, int>> comparisons = originalK2
        ? std::vector<std::pair<int, int>>{{0, 1}}
        : std::vector<std::pair<int, int>>{{0, 1}, {1, 2}, {0, 2}};
    std::cout << "Loaded frozen AllBalls HSV histograms; comparing " << methods.size() << " fixed type classifiers.\n";
    for (const auto& filename : config::allImages()) {
        const std::string stem = std::filesystem::path(filename).stem().string();
        const std::string split = filename == "AllBalls.jpeg" ? "training_resubstitution" : "evaluation";
        const auto imageDirectory = directory / stem;
        std::filesystem::create_directories(imageDirectory);
        const cv::Mat image = cv::imread(config::mediaDirectory + filename);
        if (image.empty()) throw std::runtime_error("Could not load " + filename);
        const auto circles = runDetection(image, HoughParams{}).filteredCircles;
        const auto tableCentres = transformCentres(circles);
        std::vector<std::vector<std::string>> types(methods.size());
        std::vector<std::vector<int>> numbers(methods.size());
        std::vector<BallROI> rois;
        std::vector<BallColourScores> scores;
        std::vector<CCAFeatures> ccaFeatures;
        for (size_t i = 0; i < circles.size(); ++i) {
            const auto roi = extractBall(image, circles[i]);
            const auto original = segmentBallKMeans(roi, originalK2 ? 2 : config::recognitionKMeansClusters);
            const auto k2Type = classifyBallType(original);
            BallTypeResult baseType = k2Type, ccaType;
            CCAFeatures connected;
            std::string decision;
            if (originalK2) {
                connected = analyseConnectedComponents(roi, original);
                ccaType = classifyBallTypeWithCCA(original, connected);
                decision = ccaType.type == k2Type.type ? "unchanged" : "original_CCA_to_" + ccaType.type;
            } else {
                const auto clustered = segmentBallKMeans(roi, tuned.k);
                baseType = classifyExperimentalType(clustered, tuned);
                connected = measureExperimentalCCA(roi, clustered, tuned, ccaOptions);
                ccaType = classifyExperimentalCCA(baseType, connected, ccaOptions, &decision);
            }
            auto score = scoreBallColours(roi, model); // Calculated once; shared by every profile.
            const std::vector<std::string> predictedTypes = originalK2
                ? std::vector<std::string>{k2Type.type, ccaType.type}
                : std::vector<std::string>{k2Type.type, baseType.type, ccaType.type};
            for (size_t m = 0; m < methods.size(); ++m) {
                types[m].push_back(predictedTypes[m]);
                numbers[m].push_back(histogramBallNumber(predictedTypes[m], score.bestColour));
            }
            features << filename << ',' << i + 1 << ',' << baseType.type << ',' << ccaType.type << ',' << decision << ','
                << baseType.lightPercent << ',' << baseType.colouredPercent + baseType.darkPercent << ','
                << connected.lightComponents << ',' << connected.largestLightComponentPercent << ','
                << connected.nonLightComponents << ',' << connected.largestNonLightComponentPercent << ','
                << connected.lightBoundaryCoverage << '\n';
            for (int c = 1; c <= 8; ++c)
                colourScores << filename << ',' << i + 1 << ',' << c << ',' << bodyColourName(c) << ',' << score.similarity[c - 1] << '\n';
            rois.push_back(roi); scores.push_back(std::move(score)); ccaFeatures.push_back(std::move(connected));
        }
        for (size_t m = 0; m < methods.size(); ++m)
            saveFinalResults(image, directory / "final_results" / methods[m] / stem, circles, tableCentres, types[m], numbers[m]);
        // Labels are loaded only after predictions have been made and exported.
        const auto truth = loadLabelledGroundTruth(config::groundTruthPath, filename);
        if (truth.empty()) throw std::runtime_error("Missing annotations for " + filename);
        std::vector<cv::Point2f> truthCentres;
        for (const auto& ball : truth) truthCentres.push_back(ball.centre);
        const auto e = evaluateDetections(tableCentres, truthCentres, config::matchTolerance);
        for (size_t m = 0; m < methods.size(); ++m) {
            Counts c{int(truth.size()), e.TP, e.FP, e.FN, 0, 0, 0, e.meanError * e.TP};
            for (size_t i = 0; i < circles.size(); ++i) {
                const int match = e.predictionToGroundTruth[i];
                const int actual = match < 0 ? -1 : truth[match].number;
                const int oracle = actual < 0 ? -1 : histogramBallNumber(typeFor(actual), scores[i].bestColour);
                if (actual >= 0) {
                    c.type += types[m][i] == typeFor(actual);
                    c.number += numbers[m][i] == actual;
                    c.oracle += oracle == actual;
                    ++confusion[{split, methods[m], "number", std::to_string(actual), std::to_string(numbers[m][i])}];
                    ++confusion[{split, methods[m], "type", typeFor(actual), types[m][i]}];
                }
                predictions << filename << ',' << methods[m] << ',' << i + 1 << ',' << actual << ',' << types[m][i] << ','
                    << numbers[m][i] << ',' << scores[i].bestColour << ',' << scores[i].pixels << ','
                    << (scores[i].bestColour > 0 ? scores[i].similarity[scores[i].bestColour - 1] : 0) << ','
                    << scores[i].margin << ',' << oracle << ',';
                if (actual >= 0) predictions << (types[m][i] == typeFor(actual)) << ',' << (numbers[m][i] == actual);
                else predictions << ',';
                predictions << '\n';
            }
            metrics << split << ',' << filename << ',' << methods[m] << ','; writeCounts(metrics, c); metrics << '\n';
            if (split == "evaluation") totals[m].add(c);
        }
        for (size_t i = 0; i < circles.size(); ++i) {
            const int match = e.predictionToGroundTruth[i];
            const int actual = match < 0 ? -1 : truth[match].number;
            for (const auto& [before, after] : comparisons) {
                const std::string comparison = methods[before] + "_to_" + methods[after];
                const auto typeChange = status(types[before][i] == types[after][i], types[before][i] == typeFor(actual),
                    types[after][i] == typeFor(actual), actual >= 0);
                const auto numberChange = status(numbers[before][i] == numbers[after][i], numbers[before][i] == actual,
                    numbers[after][i] == actual, actual >= 0);
                changes << filename << ',' << comparison << ',' << i + 1 << ',' << actual << ',' << types[before][i] << ','
                    << types[after][i] << ',' << numbers[before][i] << ',' << numbers[after][i] << ',' << typeChange << ',' << numberChange << '\n';
                if (split == "evaluation") ++changeCounts[{comparison, numberChange}];
            }
            std::vector<ImagePanel> panels;
            for (size_t m = 0; m < methods.size(); ++m) {
                cv::Mat crop = enlarged(rois[i].image);
                const cv::Scalar colour = actual < 0 ? cv::Scalar(180, 180, 180) :
                    numbers[m][i] == actual ? cv::Scalar(50, 240, 50) : cv::Scalar(70, 70, 255);
                cv::putText(crop, types[m][i] + " " + label(numbers[m][i]), {7, 24}, cv::FONT_HERSHEY_SIMPLEX, 0.55, {0, 0, 0}, 3, cv::LINE_AA);
                cv::putText(crop, types[m][i] + " " + label(numbers[m][i]), {7, 24}, cv::FONT_HERSHEY_SIMPLEX, 0.55, colour, 1, cv::LINE_AA);
                panels.push_back({names[m], crop});
            }
            panels.push_back({originalK2 ? "K=2 light mask" : "K=6 light mask", enlarged(ccaFeatures[i].lightMask)});
            panels.push_back({originalK2 ? "K=2 non-light mask" : "K=6 non-light mask", enlarged(ccaFeatures[i].nonLightMask)});
            panels.push_back({"CCA boundary band", enlarged(ccaFeatures[i].boundaryBand)});
            const auto grid = combineImages(panels, 3);
            cv::Mat preview(grid.rows + 78, grid.cols, CV_8UC3, cv::Scalar(32, 32, 32));
            grid.copyTo(preview(cv::Rect(0, 78, grid.cols, grid.rows)));
            cv::putText(preview, filename + " candidate " + std::to_string(i + 1) + " | actual " +
                (actual < 0 ? "unmatched" : label(actual)) + " | shared colour: " + bodyColourName(scores[i].bestColour),
                {10, 21}, cv::FONT_HERSHEY_SIMPLEX, 0.5, {245, 245, 245}, 1);
            std::ostringstream evidence;
            evidence << std::fixed << std::setprecision(1) << "CCA light boundary " << ccaFeatures[i].lightBoundaryCoverage
                << "% | eligible non-light components " << ccaFeatures[i].nonLightComponents;
            cv::putText(preview, evidence.str(), {10, 45}, cv::FONT_HERSHEY_SIMPLEX, 0.43, {230, 230, 230}, 1);
            cv::putText(preview, "Green/red text = correct/wrong number; colour histogram scores are identical across profiles.",
                {10, 66}, cv::FONT_HERSHEY_SIMPLEX, 0.38, {210, 210, 210}, 1);
            saveImage(imageDirectory / ("candidate_" + std::to_string(i + 1) + ".png"), preview);
        }
        std::vector<ImagePanel> overview;
        for (size_t m = 0; m < methods.size(); ++m)
            overview.push_back({names[m], cv::imread((directory / "final_results" / methods[m] / stem / "table_numbered.png").string())});
        saveImage(imageDirectory / "table_comparison.png", combineImages(overview, 1));
        std::cout << filename << ": compared " << circles.size() << " candidates\n";
    }
    auto summary = output(directory / "summary.md");
    summary << (originalK2 ? "# Frozen HSV histograms with original K=2 CCA\n\n"
                          : "# Frozen HSV histograms with K=6 and CCA\n\n")
        << "All profiles use exactly the same saved AllBalls colour models, query masks, scores, Hough circles "
        << "and coordinates. Only type classification changes. No histograms were retrained and no parameters "
        << "were searched or selected in this run.\n\n"
        << "| Method | Correct types | Correct numbers | Number with true type supplied |\n|---|---:|---:|---:|\n";
    for (size_t m = 0; m < methods.size(); ++m) {
        metrics << "evaluation,ALL," << methods[m] << ','; writeCounts(metrics, totals[m]); metrics << '\n';
        summary << "| " << names[m] << " | " << totals[m].type << '/' << totals[m].tp << " | " << totals[m].number
            << '/' << totals[m].tp << " | " << totals[m].oracle << '/' << totals[m].tp << " |\n";
    }
    summary << "\n| Comparison (Table1–6) | Numbers corrected | Previously correct numbers lost |\n|---|---:|---:|\n";
    for (const auto& [before, after] : comparisons) {
        const std::string comparison = methods[before] + "_to_" + methods[after];
        summary << "| " << names[before] << " -> " << names[after] << " | " << changeCounts[{comparison, "corrected"}]
            << " | " << changeCounts[{comparison, "regressed"}] << " |\n";
    }
    summary << "\nCounts pool spatially matched Table1–6 detections only. AllBalls is reported separately as "
        << "training resubstitution. Detection totals are " << totals[0].tp << " TP, " << totals[0].fp << " FP, "
        << totals[0].fn << " FN and do not change. Unknown number predictions remain errors on matched balls. "
        << "Number recall includes missed annotations; number precision includes false detections in its denominator. "
        << "The true-type oracle is a diagnostic colour ceiling, not a deployed prediction.\n\n";
    if (originalK2) summary
        << "This comparison calls the existing production CCA functions with K=2 directly, enabling the "
        << "original cue, stripe and solid rules. No K=6 settings or grid-search CCA winner are applied. "
        << "Colour histograms were trained on AllBalls only. Existing CCA thresholds are held fixed, not "
        << "retuned from this run. This does not independently establish the provenance of every historical "
        << "baseline threshold, and Table1–6 have been inspected in earlier development. "
        << "The main application is unchanged.\n\n";
    else summary << "The trained colour histograms use AllBalls only. The K=6 and solid-only CCA settings came from "
        << "earlier searches using Table1–6; this combined evaluation is therefore exploratory, not independent "
        << "test accuracy. Histogram light-mask thresholds stay V>=170,S<=165 for all methods; K=6 type/CCA "
        << "uses its own V>=170,S<=125 cluster definition. Keeping these masks distinct preserves the frozen "
        << "colour experiment and isolates the type-classifier contribution. The main application is unchanged.\n\n";
    summary << "Outputs: metrics.csv, predictions.csv, changes.csv (pairwise comparisons), colour_scores.csv "
        << "(one shared set), cca_features.csv, confusion.csv, frozen_models.yml, configuration.txt, "
        << "final_results/<method>/<image>/ and per-image candidate/table comparisons.\n";
    auto matrix = output(directory / "confusion.csv");
    matrix << "split,method,task,actual,predicted,count\n";
    for (const auto& split : {"training_resubstitution", "evaluation"})
        for (const auto& method : methods) {
            for (int actual = 0; actual <= 15; ++actual)
                for (int predicted = -1; predicted <= 15; ++predicted)
                    matrix << split << ',' << method << ",number," << actual << ',' << predicted << ','
                        << confusion[{split, method, "number", std::to_string(actual), std::to_string(predicted)}] << '\n';
            for (const auto& actual : {"White", "Solid", "Striped"})
                for (const auto& predicted : {"White", "Solid", "Striped", "Unknown"})
                    matrix << split << ',' << method << ",type," << actual << ',' << predicted << ','
                        << confusion[{split, method, "type", actual, predicted}] << '\n';
        }
    std::cout << "Saved histogram + CCA experiment to " << directory << '\n';
}

void runHistogramCCAExperiment(const std::filesystem::path& directory) {
    runHistogramCCAComparison(directory, false);
}

void runOriginalK2HistogramCCAExperiment(const std::filesystem::path& directory) {
    runHistogramCCAComparison(directory, true);
}
} // namespace experiments
