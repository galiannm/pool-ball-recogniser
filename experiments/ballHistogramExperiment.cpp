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
std::string typeFor(int number) {return number == 0 ? "White" : number <= 8 ? "Solid" : "Striped";}
std::string label(int number) {return number < 0 ? "Unknown" : number == 0 ? "Cue" : std::to_string(number);}
double ratio(double a, double b) {return b ? a / b : 0;}

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

cv::Mat scaled(const cv::Mat& image) {
    cv::Mat resized;
    cv::resize(image, resized, {220, 220}, 0, 0, cv::INTER_NEAREST);
    return resized;
}

cv::Mat backprojectionView(const BallColourScores& score, const BallColourModel& model) {
    if (score.bestColour < 0) return cv::Mat::zeros(score.bodyMask.size(), CV_8UC3);
    const auto& histogram = model.histograms[score.bestColour - 1];
    const float peak = *std::max_element(histogram.ptr<float>(), histogram.ptr<float>() + histogram.total());
    cv::Mat bytes, coloured;
    score.backprojections[score.bestColour - 1].convertTo(bytes, CV_8U, peak > 0 ? 255.0 / peak : 0);
    cv::applyColorMap(bytes, coloured, cv::COLORMAP_VIRIDIS);
    return coloured;
}

void saveModels(const std::filesystem::path& directory, const std::vector<BallColourModel>& models) {
    cv::FileStorage file((directory / "models.yml").string(), cv::FileStorage::WRITE);
    if (!file.isOpened()) throw std::runtime_error("Could not save colour models");
    file << "trainingImage" << "AllBalls.jpeg" << "models" << "[";
    for (const auto& model : models) {
        file << "{" << "space" << histogramSpaceName(model.space) << "histograms" << "[";
        for (int c = 0; c < 8; ++c)
            file << "{" << "colour" << c + 1 << "name" << bodyColourName(c + 1)
                 << "examples" << model.examples[c] << "histogram" << model.histograms[c] << "}";
        file << "]" << "}";
        std::vector<ImagePanel> panels;
        const int toneCount = model.space == BallHistogramSpace::HS ? 1 : model.options.toneBins;
        for (int c = 0; c < 8; ++c) {
            cv::Mat slices(model.options.hueBins, model.options.saturationBins * toneCount, CV_32F);
            const float* values = model.histograms[c].ptr<float>();
            for (int h = 0; h < model.options.hueBins; ++h)
                for (int s = 0; s < model.options.saturationBins; ++s)
                    for (int t = 0; t < toneCount; ++t)
                        slices.at<float>(h, t * model.options.saturationBins + s) =
                            values[(h * model.options.saturationBins + s) * toneCount + t];
            double peak = 0;
            cv::minMaxLoc(slices, nullptr, &peak);
            cv::Mat bytes, enlarged, coloured;
            slices.convertTo(bytes, CV_8U, peak > 0 ? 255 / peak : 0);
            cv::resize(bytes, enlarged, {384, 180}, 0, 0, cv::INTER_NEAREST);
            cv::applyColorMap(enlarged, coloured, cv::COLORMAP_VIRIDIS);
            panels.push_back({bodyColourName(c + 1), coloured});
        }
        saveImage(directory / (std::string(histogramSpaceName(model.space)) + "_histograms.png"), combineImages(panels, 2));
    }
    file << "]";
    file.release(); // Freeze model before any Table1–6 inference or annotation loading.
}
} // namespace

void runBallColourHistogramExperiment(const std::filesystem::path& directory) {
    std::filesystem::create_directories(directory / "training");
    const BallHistogramOptions options; // Fixed before evaluation; no threshold search.
    auto configuration = output(directory / "configuration.txt");
    configuration << "Training image: AllBalls.jpeg only. No evaluation-image tuning or winner selection.\n"
        << "Fixed ablations: HS (18x8), HSV (18x8x8), HLS stored H,S,L (18x8x8).\n"
        << "Body mask: erode circular ROI with 3x3 ellipse, exclude HSV V>=170 and S<=165.\n"
        << "Minimum body pixels=12. Neutral BGR chroma <15 sets H=0. No cloth filtering or padding.\n"
        << "Smooth each axis [1,2,1]/4, hue wrapping and endpoint replication on other axes.\n"
        << "Each training crop is L1 normalised; shared solid/stripe models average crops equally.\n"
        << "Ranking: largest 1-Bhattacharyya distance; no fitted rejection/margin threshold.\n"
        << "Black is colour model 8, not an HSV black threshold. Cue/type decisions remain current K=2.\n"
        << "Per-pixel body masks and K-means type masks are distinct measurements. CCA disabled.\n"
        << "Backprojections are float histogram masses, not probabilities of class membership.\n"
        << "PNG maps are divided by that class histogram's peak for visibility only; maps do not choose labels.\n";
    configuration.close();
    const cv::Mat trainingImage = cv::imread(config::mediaDirectory + "AllBalls.jpeg");
    if (trainingImage.empty()) throw std::runtime_error("Could not load AllBalls.jpeg");
    const auto trainingTruth = loadLabelledGroundTruth(config::groundTruthPath, "AllBalls.jpeg");
    const auto trainingCircles = runDetection(trainingImage, HoughParams{}).filteredCircles;
    std::vector<cv::Point2f> trainingCentres;
    for (const auto& ball : trainingTruth) trainingCentres.push_back(ball.centre);
    const auto matched = evaluateDetections(transformCentres(trainingCircles), trainingCentres, config::matchTolerance);
    std::vector<LabelledColourSample> samples;
    auto examples = output(directory / "training" / "examples.csv");
    examples << "candidate,trueNumber,bodyColour,x,y,radius,bodyPixels,used\n";
    std::array<bool, 16> seen{};
    for (size_t i = 0; i < trainingCircles.size(); ++i) {
        const int match = matched.predictionToGroundTruth[i];
        if (match < 0) continue;
        const int number = trainingTruth[match].number;
        if (number < 0 || number > 15 || seen[number]) throw std::runtime_error("Invalid or repeated AllBalls identity");
        seen[number] = true;
        const auto roi = extractBall(trainingImage, trainingCircles[i]);
        const auto mask = histogramBodyMask(roi, options);
        const int colour = number > 8 ? number - 8 : number;
        examples << i + 1 << ',' << number << ',' << colour << ',' << trainingCircles[i][0] << ','
            << trainingCircles[i][1] << ',' << trainingCircles[i][2] << ',' << cv::countNonZero(mask) << ',' << (number != 0) << '\n';
        saveImage(directory / "training" / ("ball_" + std::to_string(number) + ".png"),
            combineImages({{"Labelled AllBalls crop " + label(number), scaled(roi.image)},
                           {"Training body mask", scaled(mask)}}, 2));
        if (number != 0) samples.push_back({number, roi});
    }
    for (int number = 1; number <= 15; ++number)
        if (!seen[number]) throw std::runtime_error("Missing AllBalls training ball " + std::to_string(number));
    examples.close();
    std::vector<BallColourModel> models;
    for (const auto space : {BallHistogramSpace::HS, BallHistogramSpace::HSV, BallHistogramSpace::HLS})
        models.push_back(trainBallColourModel(samples, space, options));
    saveModels(directory / "training", models);
    std::cout << "Frozen three colour models trained from AllBalls only. Evaluating without tuning.\n";

    auto metrics = output(directory / "metrics.csv");
    metrics << "split,image,method,groundTruth,detections,TP,FP,FN,precision,recall,F1,meanError,"
               "correctTypes,correctNumbers,oracleNumbers,typeAccuracy,numberAccuracy,numberRecall,numberPrecision\n";
    auto predictions = output(directory / "predictions.csv");
    predictions << "image,method,candidate,trueNumber,trueType,type,number,bodyColour,bodyPixels,bestScore,margin,"
                   "oracleNumber,typeCorrect,numberCorrect\n";
    auto scoresCSV = output(directory / "colour_scores.csv");
    scoresCSV << "image,method,candidate,colour,name,similarity\n";
    auto changes = output(directory / "changes.csv");
    changes << "image,method,candidate,trueNumber,baselineNumber,number,change\n";
    std::map<std::string, Counts> totals;
    std::map<std::pair<std::string, std::string>, int> changeCounts;
    std::map<std::tuple<std::string, std::string, int, int>, int> confusion;
    for (const auto& filename : config::allImages()) {
        const std::string split = filename == "AllBalls.jpeg" ? "training_resubstitution" : "evaluation";
        const std::string stem = std::filesystem::path(filename).stem().string();
        const cv::Mat image = cv::imread(config::mediaDirectory + filename);
        if (image.empty()) throw std::runtime_error("Could not load " + filename);
        const auto circles = runDetection(image, HoughParams{}).filteredCircles;
        const auto tableCentres = transformCentres(circles);
        std::vector<BallROI> rois;
        std::vector<KMeansResult> clusters;
        std::vector<BallTypeResult> types;
        std::vector<std::string> typeNames;
        std::vector<std::vector<int>> numbers(4);
        std::vector<std::vector<BallColourScores>> colourScores(3);
        const std::array<std::string, 4> methods = {"current_k2", "hs_histogram", "hsv_histogram", "hls_histogram"};
        for (const auto& circle : circles) {
            auto roi = extractBall(image, circle);
            auto segmentation = segmentBallKMeans(roi, config::recognitionKMeansClusters);
            const auto type = classifyBallType(segmentation);
            typeNames.push_back(type.type);
            numbers[0].push_back(classifyBallNumber(segmentation, type));
            for (size_t m = 0; m < models.size(); ++m) {
                auto score = scoreBallColours(roi, models[m]);
                numbers[m + 1].push_back(histogramBallNumber(type.type, score.bestColour));
                colourScores[m].push_back(std::move(score));
            }
            rois.push_back(std::move(roi)); clusters.push_back(std::move(segmentation)); types.push_back(type);
        }
        for (size_t m = 0; m < methods.size(); ++m)
            saveFinalResults(image, directory / "final_results" / methods[m] / stem, circles, tableCentres, typeNames, numbers[m]);
        // Evaluation labels are loaded only after inference and prediction export.
        const auto truth = loadLabelledGroundTruth(config::groundTruthPath, filename);
        if (truth.empty()) throw std::runtime_error("Missing evaluation annotations for " + filename);
        std::vector<cv::Point2f> truthCentres;
        for (const auto& ball : truth) truthCentres.push_back(ball.centre);
        const auto e = evaluateDetections(tableCentres, truthCentres, config::matchTolerance);
        std::filesystem::create_directories(directory / stem);
        for (size_t m = 0; m < methods.size(); ++m) {
            Counts c{int(truth.size()), e.TP, e.FP, e.FN, 0, 0, 0, e.meanError * e.TP};
            for (size_t i = 0; i < circles.size(); ++i) {
                const int match = e.predictionToGroundTruth[i];
                const int actual = match < 0 ? -1 : truth[match].number;
                int oracle = -1;
                if (actual >= 0) {
                    BallTypeResult trueType; trueType.type = typeFor(actual);
                    oracle = m == 0 ? classifyBallNumber(clusters[i], trueType) :
                        histogramBallNumber(trueType.type, colourScores[m - 1][i].bestColour);
                    c.type += types[i].type == trueType.type;
                    c.number += numbers[m][i] == actual;
                    c.oracle += oracle == actual;
                    ++confusion[{split, methods[m], actual, numbers[m][i]}];
                }
                predictions << filename << ',' << methods[m] << ',' << i + 1 << ',' << actual << ','
                    << (actual < 0 ? "N/A" : typeFor(actual)) << ',' << types[i].type << ',' << numbers[m][i] << ',';
                if (m) {
                    const auto& score = colourScores[m - 1][i];
                    predictions << score.bestColour << ',' << score.pixels << ','
                        << (score.bestColour > 0 ? score.similarity[score.bestColour - 1] : 0) << ',' << score.margin;
                    for (int colour = 1; colour <= 8; ++colour)
                        scoresCSV << filename << ',' << methods[m] << ',' << i + 1 << ',' << colour << ','
                            << bodyColourName(colour) << ',' << score.similarity[colour - 1] << '\n';
                    const std::string change = numbers[m][i] == numbers[0][i] ? "unchanged" : actual < 0 ? "unmatched" :
                        numbers[m][i] == actual ? "corrected" : numbers[0][i] == actual ? "regressed" : "changed_still_wrong";
                    changes << filename << ',' << methods[m] << ',' << i + 1 << ',' << actual << ','
                            << numbers[0][i] << ',' << numbers[m][i] << ',' << change << '\n';
                    if (split == "evaluation") ++changeCounts[{methods[m], change}];
                } else predictions << ",,,";
                predictions << ',' << oracle << ',';
                if (actual >= 0) predictions << (types[i].type == typeFor(actual)) << ',' << (numbers[m][i] == actual);
                else predictions << ',';
                predictions << '\n';
            }
            metrics << split << ',' << filename << ',' << methods[m] << ',';
            writeCounts(metrics, c); metrics << '\n';
            if (split == "evaluation") totals[methods[m]].add(c);
        }
        for (size_t i = 0; i < circles.size(); ++i) {
            const int match = e.predictionToGroundTruth[i];
            const int actual = match < 0 ? -1 : truth[match].number;
            std::vector<ImagePanel> panels = {{"Current K=2: " + types[i].type + " " + label(numbers[0][i]), scaled(rois[i].image)},
                {"Body pixels (white excluded)", scaled(colourScores[0][i].bodyMask)}};
            for (size_t m = 0; m < models.size(); ++m) {
                const auto& score = colourScores[m][i];
                panels.push_back({std::string(histogramSpaceName(models[m].space)) + ": " +
                    bodyColourName(score.bestColour) + " -> " + label(numbers[m + 1][i]), scaled(backprojectionView(score, models[m]))});
            }
            const auto grid = combineImages(panels, 3);
            cv::Mat preview(grid.rows + 52, grid.cols, CV_8UC3, cv::Scalar(32, 32, 32));
            grid.copyTo(preview(cv::Rect(0, 52, grid.cols, grid.rows)));
            cv::putText(preview, filename + " candidate " + std::to_string(i + 1) + " | actual " +
                (actual < 0 ? "unmatched detection" : label(actual)), {10, 21}, cv::FONT_HERSHEY_SIMPLEX, 0.5, {255, 255, 255}, 1);
            cv::putText(preview, "Maps: per-model peak-scaled display. Rankings use histogram comparison, not map brightness.",
                {10, 43}, cv::FONT_HERSHEY_SIMPLEX, 0.38, {220, 220, 220}, 1);
            saveImage(directory / stem / ("candidate_" + std::to_string(i + 1) + ".png"), preview);
        }
        std::cout << filename << ": evaluated " << circles.size() << " candidates\n";
    }
    auto summary = output(directory / "summary.md");
    summary << "# Ball-colour histogram experiment\n\n"
        << "Models are trained exclusively on AllBalls. Three fixed representations are evaluated; "
        << "no configuration is selected or adjusted using Table1–6. AllBalls metrics are training resubstitution, "
        << "not a held-out estimate. The main application is unchanged.\n\n"
        << "| Method | Correct types | Correct numbers | Number with true type supplied |\n|---|---:|---:|---:|\n";
    for (const auto& [method, c] : totals) {
        metrics << "evaluation,ALL," << method << ','; writeCounts(metrics, c); metrics << '\n';
        summary << "| " << method << " | " << c.type << '/' << c.tp << " | " << c.number << '/' << c.tp
                << " | " << c.oracle << '/' << c.tp << " |\n";
    }
    summary << "\n| Histogram method | Corrected numbers versus current | Previously correct numbers lost |\n"
            << "|---|---:|---:|\n";
    for (const auto& model : models) {
        const std::string method = histogramSpaceName(model.space);
        summary << "| " << method << " | " << changeCounts[{method, "corrected"}] << " | "
                << changeCounts[{method, "regressed"}] << " |\n";
    }
    summary << "\nType is deliberately held at the current K=2 classifier for all methods, isolating the "
        << "colour-number stage. A cue prediction maps to 0; solids use colours 1–8; stripes use colours "
        << "1–7 plus 8. A striped prediction whose best model is black returns Unknown. Oracle numbers "
        << "supply the annotation type only for diagnosis.\n\n"
        << "Training uses the same eroded-circle, non-white pixel mask as inference. It does not use "
        << "hand-painted masks, test identities or test-derived colour thresholds. Each solid/stripe "
        << "example contributes equally to its shared colour model; black has one example. "
        << "Low-chroma hue is canonicalised to zero. Smoothing prevents hard bin boundaries from being "
        << "the only source of mismatch. There is no fitted confidence or margin rejection rule.\n\n"
        << "Scores are 1 minus Bhattacharyya distance between L1-normalised histograms. Backprojection "
        << "looks up the trained histogram mass per pixel using float images. PNG maps divide by each "
        << "model's own peak for visibility; their brightness cannot be compared as class probabilities.\n\n"
        << "Outputs: training/models.yml and labelled training masks; metrics.csv; predictions.csv; "
        << "colour_scores.csv (all eight scores); changes.csv (fixes and regressions); confusion.csv; "
        << "per-image candidate comparisons; final_results/<method>/<image>/ (the same PNGs and CSV as the application).\n\n"
        << "Detection counts and localisation are unchanged. Conditional number accuracy divides by matched "
        << "detections; number recall divides by all annotations; number precision divides by all detections. "
        << "Confusion matrices cover spatially matched detections only. Earlier project experiments inspected "
        << "Table1–6, so they are not pristine unseen images, although none of their labels fit these models. "
        << "A single training view per ball limits lighting and pose variation. Residual cloth, neighbouring "
        << "balls, white-patch masking and the unchanged type classifier remain potential failure sources.\n";
    auto matrix = output(directory / "confusion.csv");
    matrix << "split,method,actual,predicted,count\n";
    for (const auto& split : {"training_resubstitution", "evaluation"})
        for (const auto& method : {"current_k2", "hs_histogram", "hsv_histogram", "hls_histogram"})
            for (int actual = 0; actual <= 15; ++actual)
                for (int predicted = -1; predicted <= 15; ++predicted)
                    matrix << split << ',' << method << ',' << actual << ',' << predicted << ','
                           << confusion[{split, method, actual, predicted}] << '\n';
    std::cout << "Histogram experiment saved to " << directory << '\n';
}
} // namespace experiments
