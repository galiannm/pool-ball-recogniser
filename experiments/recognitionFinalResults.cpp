#include "recognitionGridSearch.h"
#include "ccaGridSearch.h"
#include "config.h"
#include "detection/ballDetection.h"
#include "evaluation/evaluation.h"
#include "geometry/perspectiveTransform.h"
#include "io/finalResults.h"
#include "io/output.h"
#include "visualisation/combineImages.h"

#include <array>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace experiments {
namespace {
struct Predictions {
    std::vector<std::string> types;
    std::vector<int> numbers;
    int correctTypes = 0, correctNumbers = 0;
};

std::ofstream output(const std::filesystem::path& path) {
    std::ofstream out(path);
    out.exceptions(std::ios::badbit | std::ios::failbit);
    return out;
}

std::string numberLabel(int n) {
    return n == 0 ? "Cue" : n < 0 ? "Unknown" : std::to_string(n);
}

std::string expectedType(int n) {
    return n == 0 ? "White" : n <= 8 ? "Solid" : "Striped";
}

std::string change(const Predictions& before, const Predictions& after, size_t i, int truth) {
    if (before.types[i] == after.types[i] && before.numbers[i] == after.numbers[i]) return "unchanged";
    if (truth < 0) return "unmatched_detection";
    const bool wasCorrect = before.numbers[i] == truth, nowCorrect = after.numbers[i] == truth;
    if (!wasCorrect && nowCorrect) return "number_corrected";
    if (wasCorrect && !nowCorrect) return "number_regressed";
    return before.numbers[i] == after.numbers[i] ? "type_changed" : "number_changed_still_wrong";
}

void recordConfiguration(const std::filesystem::path& directory, const RecognitionParameters& p) {
    auto out = output(directory / "final_configuration.txt");
    out << "Frozen recognition-search winner 2153; not a new search.\n"
        << "K=" << p.k << "; padding=0; cloth=false; K-means=BGR; RNG seed=12345\n"
        << "Light: V >= " << p.lightMinV << " and S <= " << p.lightMaxS << '\n'
        << "Cue: light >= " << p.whiteMinPercent << "%; Solid: non-light >= " << p.solidMinPercent << "%\n"
        << "Reporting dark V <= " << p.darkMaxV << "; eligible body cluster >= " << p.bodyMinPercent << "%\n"
        << "Black: S < " << p.blackMaxS << ", V <= " << p.blackMaxV << ", type Solid\n"
        << "Hue bounds: red < " << p.redUpper << " or >= " << p.redWrap
        << "; orange < " << p.orangeUpper << "; yellow < " << p.yellowUpper
        << "; green [" << p.greenLower << ',' << p.greenUpper << "); blue ["
        << p.blueLower << ',' << p.bluePurple << "); purple [" << p.bluePurple << ',' << p.purpleUpper << ")\n"
        << "Maroon: red hue and V < " << p.maroonMaxV << "; striped body number += 8\n\n"
        << "final_results/: K=6 tuned only, original strict black cutoff retained.\n"
        << "final_results_cca/: same recognition + CCA winner 20242: solid override only; "
        << "8-connectivity; component area >= 2%; boundary erosion ellipse 5x5; non-light >= 70%; "
        << "light boundary <= 20%; exactly one eligible non-light component.\n"
        << "final_results_current/: current application classifier, K=" << config::recognitionKMeansClusters
        << ", including its recent inclusive black cutoff S <= 60.\n"
        << "All profiles share identical Hough circles and transformations. No new parameters selected.\n";
}
} // namespace

void exportRecognitionFinalResults(const std::filesystem::path& directory) {
    std::filesystem::create_directories(directory / "final_comparisons");
    const auto tuned = ccaRecognitionBase(); // Frozen recognition-search winner 2153.
    CCAParameters cca;
    cca.cue = false;
    cca.stripe = false;
    cca.solidMaxBoundary = 20; // Frozen CCA winner 20242; all other defaults retained.
    recordConfiguration(directory, tuned);

    auto changes = output(directory / "final_changes.csv");
    changes << "image,candidate,trueNumber,currentType,currentNumber,k6Type,k6Number,k6Change,"
               "k6CCAType,k6CCANumber,k6CCAChange\n";
    auto counts = output(directory / "final_counts.csv");
    counts << "image,matched,FP,FN,currentCorrectTypes,currentCorrectNumbers,k6CorrectTypes,k6CorrectNumbers,"
              "k6CCACorrectTypes,k6CCACorrectNumbers\n";
    std::array<int, 3> totalTypes{}, totalNumbers{};
    int matchedTotal = 0, fpTotal = 0, fnTotal = 0;
    const std::array<std::string, 3> names = {"Current application", "K=6 tuned (2153)", "K=6 + CCA (20242)"};
    const std::array<std::string, 3> folders = {"final_results_current", "final_results", "final_results_cca"};
    for (const auto& filename : config::allImages()) {
        const cv::Mat image = cv::imread(config::mediaDirectory + filename);
        if (image.empty()) throw std::runtime_error("Could not load " + filename);
        const auto detection = runDetection(image, HoughParams{});
        const auto& circles = detection.filteredCircles;
        const auto tableCentres = transformCentres(circles);
        std::array<Predictions, 3> profiles;
        for (const auto& circle : circles) {
            const auto roi = extractBall(image, circle);
            const auto original = segmentBallKMeans(roi, config::recognitionKMeansClusters);
            const auto baselineType = classifyBallType(original);
            profiles[0].types.push_back(baselineType.type);
            profiles[0].numbers.push_back(classifyBallNumber(original, baselineType));
            const auto clusters = segmentBallKMeans(roi, tuned.k);
            const auto type = classifyExperimentalType(clusters, tuned);
            const auto adjusted = classifyExperimentalCCA(type, measureExperimentalCCA(roi, clusters, tuned, cca), cca);
            profiles[1].types.push_back(type.type);
            profiles[1].numbers.push_back(classifyExperimentalNumber(clusters, type.type, tuned));
            profiles[2].types.push_back(adjusted.type);
            profiles[2].numbers.push_back(classifyExperimentalNumber(clusters, adjusted.type, tuned));
        }
        const std::string stem = std::filesystem::path(filename).stem().string();
        for (size_t p = 0; p < profiles.size(); ++p)
            saveFinalResults(image, directory / folders[p] / stem, circles, tableCentres,
                             profiles[p].types, profiles[p].numbers);

        // Annotations are used only after inference and export, to explain changes.
        const auto truth = loadLabelledGroundTruth(config::groundTruthPath, filename);
        if (truth.empty()) throw std::runtime_error("Missing annotations for " + filename);
        std::vector<cv::Point2f> truthCentres;
        for (const auto& ball : truth) truthCentres.push_back(ball.centre);
        const auto matches = evaluateDetections(tableCentres, truthCentres, config::matchTolerance);
        const auto comparisons = directory / "final_comparisons" / stem;
        std::filesystem::create_directories(comparisons);
        for (size_t i = 0; i < circles.size(); ++i) {
            const int match = matches.predictionToGroundTruth[i];
            const int actual = match < 0 ? -1 : truth[match].number;
            for (auto& p : profiles) {
                p.correctNumbers += actual >= 0 && p.numbers[i] == actual;
                p.correctTypes += actual >= 0 && p.types[i] == expectedType(actual);
            }
            const auto k6Change = change(profiles[0], profiles[1], i, actual);
            const auto ccaChange = change(profiles[0], profiles[2], i, actual);
            changes << filename << ',' << i + 1 << ',' << actual << ',' << profiles[0].types[i] << ','
                << profiles[0].numbers[i] << ',' << profiles[1].types[i] << ',' << profiles[1].numbers[i]
                << ',' << k6Change << ',' << profiles[2].types[i] << ',' << profiles[2].numbers[i] << ',' << ccaChange << '\n';
            if (k6Change == "unchanged" && ccaChange == "unchanged") continue;
            const int half = cvCeil(circles[i][2]) + 12;
            const cv::Rect crop = cv::Rect(cvRound(circles[i][0]) - half, cvRound(circles[i][1]) - half,
                2 * half + 1, 2 * half + 1) & cv::Rect(0, 0, image.cols, image.rows);
            cv::Mat enlarged;
            cv::resize(image(crop), enlarged, {}, 4, 4, cv::INTER_NEAREST);
            std::vector<ImagePanel> panels;
            for (size_t p = 0; p < profiles.size(); ++p) {
                cv::Mat preview = enlarged.clone();
                const cv::Scalar colour = actual < 0 ? cv::Scalar(180, 180, 180) :
                    profiles[p].numbers[i] == actual ? cv::Scalar(50, 240, 50) : cv::Scalar(70, 70, 255);
                const std::string predicted = profiles[p].types[i] + " " + numberLabel(profiles[p].numbers[i]);
                cv::putText(preview, predicted, {8, 25}, cv::FONT_HERSHEY_SIMPLEX, 0.6, {0, 0, 0}, 3, cv::LINE_AA);
                cv::putText(preview, predicted, {8, 25}, cv::FONT_HERSHEY_SIMPLEX, 0.6, colour, 1, cv::LINE_AA);
                panels.push_back({names[p] + " / actual " + (actual < 0 ? "unmatched" : numberLabel(actual)), preview});
            }
            saveImage(comparisons / ("candidate_" + std::to_string(i + 1) + ".png"), combineImages(panels, 3));
        }
        std::vector<ImagePanel> overviews;
        for (size_t p = 0; p < profiles.size(); ++p) {
            overviews.push_back({names[p], cv::imread((directory / folders[p] / stem / "table_numbered.png").string())});
        }
        saveImage(comparisons / "table_comparison.png", combineImages(overviews, 1));
        counts << filename << ',' << matches.TP << ',' << matches.FP << ',' << matches.FN;
        for (size_t p = 0; p < profiles.size(); ++p) {
            counts << ',' << profiles[p].correctTypes << ',' << profiles[p].correctNumbers;
            if (filename != config::tuningImage) {
                totalTypes[p] += profiles[p].correctTypes;
                totalNumbers[p] += profiles[p].correctNumbers;
            }
        }
        counts << '\n';
        if (filename != config::tuningImage) {
            matchedTotal += matches.TP;
            fpTotal += matches.FP;
            fnTotal += matches.FN;
        }
    }
    auto summary = output(directory / "final_summary.md");
    summary << "# Final-result recognition comparisons\n\n"
        << "All profiles use the same detections and rendering as the application. "
        << "No search was rerun, and no experimental recognition rules were changed.\n\n"
        << "| Profile | Correct type (Table1–6) | Correct number (Table1–6) |\n|---|---:|---:|\n";
    for (size_t p = 0; p < names.size(); ++p)
        summary << "| " << names[p] << " | " << totalTypes[p] << '/' << matchedTotal << " | "
                << totalNumbers[p] << '/' << matchedTotal << " |\n";
    summary << "\nThere are " << fpTotal << " false detections and " << fnTotal << " missed annotations. "
        << "These are excluded from conditional recognition accuracy, not silently corrected.\n\n"
        << "- `final_results/<image>/`: tuned K=6 winner 2153, without CCA.\n"
        << "- `final_results_cca/<image>/`: K=6 plus solid-only CCA winner 20242.\n"
        << "- `final_results_current/<image>/`: snapshot of the current application.\n"
        << "- `final_comparisons/<image>/`: whole-table comparison and one labelled crop per changed candidate. "
        << "Green crop text means correct number, red incorrect, grey unmatched.\n"
        << "- `final_changes.csv`: all candidates, including corrections, regressions and unchanged predictions.\n"
        << "- `final_counts.csv`: per-image counts; AllBalls is listed separately.\n"
        << "- `final_configuration.txt`: frozen settings and the black-threshold distinction.\n\n"
        << "Each final-result folder contains the same four PNGs and balls.csv as the application. "
        << "They show predictions only; annotation labels are confined to the comparison outputs.\n\n"
        << "The current application includes S <= 60 for black. The historical K=6 experiment uses S < 60. "
        << "Consequently Table2 candidate 16 (black 8) is correct in the current application but remains "
        << "yellow 1 in these K=6 snapshots. This is shown as a regression.\n\n"
        << "The winners were selected using Table1–6 during earlier experiments. These remain exploratory "
        << "results, not independent test estimates. The main application settings are unchanged.\n";
    std::cout << "Saved final recognition comparisons to " << directory << '\n';
}
} // namespace experiments
