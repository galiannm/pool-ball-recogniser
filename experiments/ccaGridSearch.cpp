#include "ccaGridSearch.h"
#include "recognitionGridSearch.h"
#include "config.h"
#include "detection/ballDetection.h"
#include "evaluation/evaluation.h"
#include "geometry/perspectiveTransform.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <tuple>

namespace experiments {

CCAFeatures measureExperimentalCCA(const BallROI& ball, const KMeansResult& segmentation,
                                   const RecognitionParameters& recognition,
                                   const CCAParameters& p) {
    CCAFeatures result;
    if (p.minComponentPercent < 0 || p.boundaryKernel < 3 || p.boundaryKernel % 2 != 1)
        throw std::invalid_argument("CCA needs a nonnegative area cutoff and odd kernel >= 3");
    if (ball.image.empty() || ball.mask.empty() || segmentation.regions.empty())
        return result;
    const int area = cv::countNonZero(ball.mask);
    if (!area)
        return result;
    result.lightMask = cv::Mat::zeros(ball.mask.size(), CV_8UC1);
    for (const auto& region : segmentation.regions)
        if (region.hsv[2] >= recognition.lightMinV && region.hsv[1] <= recognition.lightMaxS)
            cv::bitwise_or(result.lightMask, region.mask, result.lightMask);
    cv::bitwise_and(result.lightMask, ball.mask, result.lightMask);
    cv::bitwise_not(result.lightMask, result.nonLightMask);
    cv::bitwise_and(result.nonLightMask, ball.mask, result.nonLightMask);
    const auto components = [&](const cv::Mat& mask, int& count, double& largest) {
        cv::Mat labels, stats, centres;
        const int n = cv::connectedComponentsWithStats(mask, labels, stats, centres, 8, CV_32S);
        for (int label = 1; label < n; ++label) {
            const double percent = 100.0 * stats.at<int>(label, cv::CC_STAT_AREA) / area;
            if (percent >= p.minComponentPercent) {
                ++count;
                largest = std::max(largest, percent);
            }
        }
    };
    components(result.lightMask, result.lightComponents, result.largestLightComponentPercent);
    components(result.nonLightMask, result.nonLightComponents, result.largestNonLightComponentPercent);
    cv::Mat eroded;
    // Preserve production erosion border semantics for an exact default comparison.
    cv::erode(ball.mask, eroded, cv::getStructuringElement(
        cv::MORPH_ELLIPSE, {p.boundaryKernel, p.boundaryKernel}));
    cv::subtract(ball.mask, eroded, result.boundaryBand);
    const int boundaryArea = cv::countNonZero(result.boundaryBand);
    if (boundaryArea) {
        cv::Mat lightBoundary;
        cv::bitwise_and(result.lightMask, result.boundaryBand, lightBoundary);
        result.lightBoundaryCoverage = 100.0 * cv::countNonZero(lightBoundary) / boundaryArea;
    }
    return result;
}

BallTypeResult classifyExperimentalCCA(const BallTypeResult& baseline, const CCAFeatures& cca,
                                      const CCAParameters& p, std::string* decision) {
    BallTypeResult result = baseline;
    if (decision)
        *decision = "fallback";
    const double nonLight = baseline.colouredPercent + baseline.darkPercent;
    const auto choose = [&](const std::string& type, const std::string& rule) {
        result.type = type;
        if (decision)
            *decision = rule;
    };
    if (p.cue && baseline.lightPercent >= p.cueMinLight &&
        cca.largestLightComponentPercent >= p.cueLargestLight &&
        cca.lightBoundaryCoverage >= p.cueMinBoundary && cca.nonLightComponents == 0)
        choose("White", "cue");
    else if (p.stripe && cca.largestLightComponentPercent >= p.stripeLargestLight &&
             cca.lightBoundaryCoverage >= p.stripeMinBoundary && nonLight >= p.stripeMinNonLight)
        choose("Striped", "stripe");
    else if (p.solid && nonLight >= p.solidMinNonLight &&
             cca.lightBoundaryCoverage <= p.solidMaxBoundary &&
             cca.nonLightComponents >= 1 && cca.nonLightComponents <= p.solidMaxComponents &&
             cca.largestNonLightComponentPercent >= p.solidLargestNonLight)
        choose("Solid", "solid");
    return result;
}

namespace {

std::ofstream output(const std::filesystem::path& path) {
    std::ofstream stream(path);
    stream.exceptions(std::ios::failbit | std::ios::badbit);
    stream << std::fixed << std::setprecision(6);
    return stream;
}

std::string trueType(int n) {
    return n < 0 ? "Unknown" : n == 0 ? "White" : n <= 8 ? "Solid" : "Striped";
}

int ruleMask(const CCAParameters& p) {
    return int(p.cue) + 2 * int(p.stripe) + 4 * int(p.solid);
}

struct Candidate {
    std::string image;
    bool tuning = false;
    int id = 0, truth = -1;
    BallROI ball;
    KMeansResult original, tuned;
    BallTypeResult originalType, tunedType;
    int originalNumber = -1, tunedNumber = -1;
    CCAFeatures originalCCA;
    std::map<std::pair<double, int>, CCAFeatures> features;
};

struct Data {
    std::vector<Candidate> candidates;
    std::map<std::string, int> groundTruth;
};

bool sameFeatures(const CCAFeatures& a, const CCAFeatures& b) {
    return a.lightComponents == b.lightComponents && a.nonLightComponents == b.nonLightComponents &&
        a.largestLightComponentPercent == b.largestLightComponentPercent &&
        a.largestNonLightComponentPercent == b.largestNonLightComponentPercent &&
        a.lightBoundaryCoverage == b.lightBoundaryCoverage &&
        cv::norm(a.lightMask, b.lightMask, cv::NORM_INF) == 0 &&
        cv::norm(a.nonLightMask, b.nonLightMask, cv::NORM_INF) == 0 &&
        cv::norm(a.boundaryBand, b.boundaryBand, cv::NORM_INF) == 0;
}

Data prepare(const CCASearchGrid& grid, const std::filesystem::path& directory) {
    Data data;
    auto detectionCSV = output(directory / "detection_counts.csv");
    detectionCSV << "image,split,groundTruth,matched,falsePositives,missed\n";
    auto featureCSV = output(directory / "features.csv");
    featureCSV << "image,candidate,trueNumber,profile,minComponentPercent,boundaryKernel,"
        "lightPercent,nonLightPercent,lightComponents,largestLightPercent,"
        "nonLightComponents,largestNonLightPercent,lightBoundaryCoverage\n";
    const auto originalParameters = RecognitionParameters{};
    const auto tunedParameters = ccaRecognitionBase();
    for (const auto& name : config::allImages()) {
        const cv::Mat image = cv::imread(config::mediaDirectory + name);
        if (image.empty())
            throw std::runtime_error("Could not load " + name);
        const auto truth = loadLabelledGroundTruth(config::groundTruthPath, name);
        if (truth.empty())
            throw std::runtime_error("Missing ground truth for " + name);
        data.groundTruth[name] = static_cast<int>(truth.size());
        const auto circles = runDetection(image, HoughParams{}).filteredCircles;
        std::vector<cv::Point2f> centres;
        for (const auto& t : truth)
            centres.push_back(t.centre);
        const auto match = evaluateDetections(transformCentres(circles), centres, config::matchTolerance);
        detectionCSV << name << ',' << (name == config::tuningImage ? "reference" : "tuning") << ','
                     << truth.size() << ',' << match.TP << ',' << match.FP << ',' << match.FN << '\n';
        for (size_t i = 0; i < circles.size(); ++i) {
            Candidate c;
            c.image = name;
            c.tuning = name != config::tuningImage;
            c.id = static_cast<int>(i + 1);
            if (match.predictionToGroundTruth[i] >= 0)
                c.truth = truth[match.predictionToGroundTruth[i]].number;
            c.ball = extractBall(image, circles[i]);
            c.original = segmentBallKMeans(c.ball, 2);
            c.tuned = segmentBallKMeans(c.ball, tunedParameters.k);
            c.originalType = classifyBallType(c.original);
            c.tunedType = classifyExperimentalType(c.tuned, tunedParameters);
            c.originalNumber = classifyBallNumber(c.original, c.originalType);
            c.tunedNumber = classifyExperimentalNumber(c.tuned, c.tunedType.type, tunedParameters);
            c.originalCCA = measureExperimentalCCA(c.ball, c.original, originalParameters, {});
            const auto productionCCA = analyseConnectedComponents(c.ball, c.original);
            if (!sameFeatures(c.originalCCA, productionCCA) ||
                classifyExperimentalCCA(c.originalType, c.originalCCA, {}).type !=
                    classifyBallTypeWithCCA(c.original, productionCCA).type)
                throw std::runtime_error("CCA defaults do not reproduce production");
            const auto saveFeatures = [&](const CCAFeatures& f, const CCAParameters& p,
                                          const BallTypeResult& baseline, const std::string& profile) {
                featureCSV << name << ',' << c.id << ',' << c.truth << ',' << profile << ','
                    << p.minComponentPercent << ',' << p.boundaryKernel << ',' << baseline.lightPercent
                    << ',' << baseline.colouredPercent + baseline.darkPercent << ',' << f.lightComponents
                    << ',' << f.largestLightComponentPercent << ',' << f.nonLightComponents << ','
                    << f.largestNonLightComponentPercent << ',' << f.lightBoundaryCoverage << '\n';
            };
            saveFeatures(c.originalCCA, {}, c.originalType, "original_k2");
            for (double area : grid.componentPercent) {
                for (int kernel : grid.boundaryKernel) {
                    CCAParameters p;
                    p.minComponentPercent = area;
                    p.boundaryKernel = kernel;
                    auto f = measureExperimentalCCA(c.ball, c.tuned, tunedParameters, p);
                    saveFeatures(f, p, c.tunedType, "tuned_k6");
                    c.features.emplace(std::make_pair(area, kernel), std::move(f));
                }
            }
            data.candidates.push_back(std::move(c));
        }
        std::cout << "Cached CCA " << name << ": " << match.TP << " matched\n" << std::flush;
    }
    return data;
}

struct Score {
    int gt = 0, matched = 0, type = 0, number = 0;
    int changed = 0, typeFixed = 0, typeWorsened = 0, numberFixed = 0, numberWorsened = 0;
};

struct Trial {
    int id = 0, parent = 0;
    std::string stage;
    bool original = false;
    CCAParameters p;
    Score score;
};

const CCAFeatures& features(const Candidate& c, const Trial& t) {
    return t.original ? c.originalCCA : c.features.at({t.p.minComponentPercent, t.p.boundaryKernel});
}

BallTypeResult classify(const Candidate& c, const Trial& t, std::string* decision = nullptr) {
    return classifyExperimentalCCA(t.original ? c.originalType : c.tunedType, features(c, t), t.p, decision);
}

Score evaluate(const Data& data, const Trial& t, bool tuning, const std::string& image = "") {
    Score s;
    for (const auto& gt : data.groundTruth)
        if ((gt.first != config::tuningImage) == tuning && (image.empty() || image == gt.first))
            s.gt += gt.second;
    const auto recognition = t.original ? RecognitionParameters{} : ccaRecognitionBase();
    for (const auto& c : data.candidates) {
        if (c.tuning != tuning || c.truth < 0 || (!image.empty() && c.image != image))
            continue;
        const auto type = classify(c, t);
        const auto& clusters = t.original ? c.original : c.tuned;
        const int number = classifyExperimentalNumber(clusters, type.type, recognition);
        const bool typeCorrect = type.type == trueType(c.truth), numberCorrect = number == c.truth;
        const bool oldTypeCorrect = c.tunedType.type == trueType(c.truth), oldNumberCorrect = c.tunedNumber == c.truth;
        ++s.matched;
        s.type += typeCorrect;
        s.number += numberCorrect;
        s.changed += type.type != c.tunedType.type;
        s.typeFixed += !oldTypeCorrect && typeCorrect;
        s.typeWorsened += oldTypeCorrect && !typeCorrect;
        s.numberFixed += !oldNumberCorrect && numberCorrect;
        s.numberWorsened += oldNumberCorrect && !numberCorrect;
    }
    return s;
}

double departure(const CCAParameters& p) {
    return std::abs(p.minComponentPercent - 2) + std::abs(p.boundaryKernel - 5) / 2.0 +
        std::abs(p.cueMinLight - 85) / 5.0 + std::abs(p.cueLargestLight - 80) / 10.0 +
        std::abs(p.cueMinBoundary - 75) / 15.0 + std::abs(p.stripeLargestLight - 20) / 5.0 +
        std::abs(p.stripeMinBoundary - 20) / 10.0 + std::abs(p.stripeMinNonLight - 10) / 5.0 +
        std::abs(p.solidMinNonLight - 70) / 10.0 + std::abs(p.solidMaxBoundary - 12) / 10.0 +
        std::abs(p.solidMaxComponents - 1) + p.solidLargestNonLight / 50.0;
}

bool better(const Trial& a, const Trial& b, bool typeFirst = false) {
    const auto key = [typeFirst](const Trial& t) {
        const auto& s = t.score;
        return std::make_tuple(typeFirst ? s.type : s.number, typeFirst ? s.number : s.type,
            -s.numberWorsened, -s.typeWorsened, -s.changed,
            -int(t.p.cue) - int(t.p.stripe) - int(t.p.solid), -departure(t.p), -t.id);
    };
    return key(a) > key(b);
}

const char* parametersHeader = "minComponentPercent,boundaryKernel,cueEnabled,stripeEnabled,solidEnabled,"
    "cueMinLight,cueLargestLight,cueMinBoundary,stripeLargestLight,stripeMinBoundary,stripeMinNonLight,"
    "solidMinNonLight,solidMaxBoundary,solidMaxComponents,solidLargestNonLight";

void parameters(std::ostream& out, const CCAParameters& p) {
    out << p.minComponentPercent << ',' << p.boundaryKernel << ',' << p.cue << ',' << p.stripe << ',' << p.solid
        << ',' << p.cueMinLight << ',' << p.cueLargestLight << ',' << p.cueMinBoundary << ','
        << p.stripeLargestLight << ',' << p.stripeMinBoundary << ',' << p.stripeMinNonLight << ','
        << p.solidMinNonLight << ',' << p.solidMaxBoundary << ',' << p.solidMaxComponents << ',' << p.solidLargestNonLight;
}

const char* scoreHeader = "groundTruth,matched,typeCorrect,numberCorrect,typeAccuracy,numberAccuracy,"
    "endToEndNumberRecall,changedTypes,typeFixed,typeWorsened,numberFixed,numberWorsened";

void writeScore(std::ostream& out, const Score& s) {
    const auto ratio = [](int a, int b) { return b ? double(a) / b : 0.0; };
    out << s.gt << ',' << s.matched << ',' << s.type << ',' << s.number << ','
        << ratio(s.type, s.matched) << ',' << ratio(s.number, s.matched) << ',' << ratio(s.number, s.gt)
        << ',' << s.changed << ',' << s.typeFixed << ',' << s.typeWorsened << ',' << s.numberFixed << ',' << s.numberWorsened;
}

using Field = std::pair<int CCAParameters::*, const std::vector<int>*>;

int group(const Data& data, std::vector<Trial>& trials, int startIndex,
          const std::vector<Field>& fields, const std::string& stage) {
    const Trial start = trials.at(startIndex);
    Trial t = start;
    t.stage = stage;
    t.parent = start.id;
    int best = startIndex;
    const auto visit = [&](const auto& self, size_t depth) -> void {
        if (depth < fields.size()) {
            for (int value : *fields[depth].second) {
                t.p.*(fields[depth].first) = value;
                self(self, depth + 1);
            }
        } else {
            t.id = static_cast<int>(trials.size() + 1);
            t.score = evaluate(data, t, true);
            trials.push_back(t);
            const int index = static_cast<int>(trials.size() - 1);
            if (better(trials[index], trials[best]))
                best = index;
        }
    };
    visit(visit, 0);
    return best;
}

void visuals(const std::filesystem::path& directory, const Data& data,
             const std::vector<std::pair<std::string, Trial>>& finalists) {
    std::vector<std::pair<std::string, Trial>> unique;
    std::set<int> ids;
    for (const auto& item : finalists)
        if (ids.insert(item.second.id).second)
            unique.push_back(item);
    constexpr int width = 240, header = 55, height = 570, preview = 140;
    for (const auto& c : data.candidates) {
        const auto folder = directory / std::filesystem::path(c.image).stem();
        std::filesystem::create_directories(folder);
        cv::Mat grid(header + height, width * static_cast<int>(unique.size()), CV_8UC3, cv::Scalar(35, 35, 35));
        const std::string expected = c.truth < 0 ? "No ground-truth match"
            : "Expected: " + trueType(c.truth) + " #" + std::to_string(c.truth);
        cv::putText(grid, c.image + " / candidate " + std::to_string(c.id) + " / " + expected,
                    {10, 25}, cv::FONT_HERSHEY_SIMPLEX, 0.55, {255, 255, 255}, 1);
        for (size_t j = 0; j < unique.size(); ++j) {
            const auto& item = unique[j];
            const auto& t = item.second;
            const int left = static_cast<int>(j) * width;
            const auto text = [&](const std::string& value, int y, cv::Scalar colour = {220, 220, 220}) {
                cv::putText(grid, value, {left + 8, header + y}, cv::FONT_HERSHEY_SIMPLEX, 0.45, colour, 1);
            };
            text(item.first + " (" + std::to_string(t.id) + ")", 0);
            text(t.original ? "K=2 / original rules" : "K=6 / configuration 2153", 23);
            cv::Mat isolated = cv::Mat::zeros(c.ball.image.size(), CV_8UC3);
            c.ball.image.copyTo(isolated, c.ball.mask);
            const auto panel = [&](const cv::Mat& image, int top) {
                if (image.empty())
                    return;
                cv::Mat colour, resized;
                if (image.channels() == 1)
                    cv::cvtColor(image, colour, cv::COLOR_GRAY2BGR);
                else
                    colour = image;
                const double scale = double(preview) / std::max(colour.cols, colour.rows);
                cv::resize(colour, resized, {}, scale, scale, cv::INTER_NEAREST);
                resized.copyTo(grid(cv::Rect(left + (width - resized.cols) / 2,
                                             header + top, resized.cols, resized.rows)));
            };
            panel(isolated, 40);
            const auto& f = features(c, t);
            cv::Mat regions = cv::Mat::zeros(c.ball.image.size(), CV_8UC3);
            regions.setTo(cv::Scalar(80, 150, 220), f.nonLightMask);
            regions.setTo(cv::Scalar(245, 245, 245), f.lightMask);
            // Boundary overlay reveals the band used by the boundary-coverage feature.
            cv::Mat boundaryLight, boundaryNonLight;
            cv::bitwise_and(f.boundaryBand, f.lightMask, boundaryLight);
            cv::bitwise_and(f.boundaryBand, f.nonLightMask, boundaryNonLight);
            regions.setTo(cv::Scalar(100, 255, 100), boundaryLight);
            regions.setTo(cv::Scalar(80, 80, 255), boundaryNonLight);
            panel(regions, 220);
            text("Light/non-light + boundary", 205);
            std::string rule;
            const auto type = classify(c, t, &rule);
            const auto recognition = t.original ? RecognitionParameters{} : ccaRecognitionBase();
            const int number = classifyExperimentalNumber(t.original ? c.original : c.tuned, type.type, recognition);
            const auto status = [&](bool correct) { return c.truth < 0 ? " N/A" : correct ? " OK" : " WRONG"; };
            const auto colour = [&](bool correct) {
                return c.truth < 0 ? cv::Scalar(180, 180, 180)
                    : correct ? cv::Scalar(90, 220, 90) : cv::Scalar(80, 80, 255);
            };
            text("Type: " + type.type + status(type.type == trueType(c.truth)), 390,
                 colour(type.type == trueType(c.truth)));
            text("Number: " + std::to_string(number) + status(number == c.truth), 413, colour(number == c.truth));
            text("Rule: " + rule + " / mask " + std::to_string(ruleMask(t.p)), 436);
            text("Light components: " + std::to_string(f.lightComponents), 460);
            text("Non-light components: " + std::to_string(f.nonLightComponents), 483);
            text("Largest light: " + std::to_string(cvRound(f.largestLightComponentPercent)) + "%", 506);
            text("Light boundary: " + std::to_string(cvRound(f.lightBoundaryCoverage)) + "%", 529);
            text("Area cutoff: " + std::to_string(t.p.minComponentPercent).substr(0, 3) +
                 "% / kernel " + std::to_string(t.p.boundaryKernel), 552);
        }
        const auto path = folder / ("ball_" + std::to_string(c.id) + "_comparison.png");
        if (!cv::imwrite(path.string(), grid))
            throw std::runtime_error("Could not save " + path.string());
    }
}

void exportResults(const Data& data, const std::vector<Trial>& trials,
                   const std::vector<std::pair<std::string, Trial>>& finalists,
                   const std::filesystem::path& directory) {
    auto ranking = trials;
    std::stable_sort(ranking.begin(), ranking.end(), [](const Trial& a, const Trial& b) {
        return better(a, b);
    });
    auto leaderboard = output(directory / "leaderboard.csv");
    leaderboard << "rank,configId,parentId,stage,profile," << parametersHeader << ',' << scoreHeader << '\n';
    for (size_t i = 0; i < ranking.size(); ++i) {
        const auto& t = ranking[i];
        leaderboard << i + 1 << ',' << t.id << ',' << t.parent << ',' << t.stage << ','
                    << (t.original ? "original_k2" : "tuned_k6") << ',';
        parameters(leaderboard, t.p);
        leaderboard << ',';
        writeScore(leaderboard, t.score);
        leaderboard << '\n';
    }
    auto ablation = output(directory / "ablation.csv");
    ablation << "ruleMask,configId," << parametersHeader << ',' << scoreHeader << '\n';
    for (int mask = 0; mask < 8; ++mask) {
        const Trial* best = nullptr;
        for (const auto& t : trials)
            if (!t.original && ruleMask(t.p) == mask && (!best || better(t, *best)))
                best = &t;
        if (best) {
            ablation << mask << ',' << best->id << ',';
            parameters(ablation, best->p);
            ablation << ',';
            writeScore(ablation, best->score);
            ablation << '\n';
        }
    }
    auto configs = output(directory / "finalists.csv");
    configs << "label,configId,profile," << parametersHeader << '\n';
    auto metrics = output(directory / "metrics.csv");
    metrics << "label,configId,split,image," << scoreHeader << '\n';
    auto predictions = output(directory / "predictions.csv");
    predictions << "label,configId,image,candidate,trueNumber,trueType,type,number,baseType,baseNumber,"
        "typeCorrect,numberCorrect,typeChanged,decision,lightPercent,nonLightPercent,"
        "lightComponents,largestLightPercent,nonLightComponents,largestNonLightPercent,lightBoundaryCoverage\n";
    auto confusion = output(directory / "confusion.csv");
    confusion << "label,split,task,actual,predicted,count\n";
    for (const auto& item : finalists) {
        const auto& t = item.second;
        configs << item.first << ',' << t.id << ',' << (t.original ? "original_k2" : "tuned_k6") << ',';
        parameters(configs, t.p);
        configs << '\n';
        for (bool tuning : {true, false}) {
            const std::string split = tuning ? "tuning" : "reference";
            metrics << item.first << ',' << t.id << ',' << split << ",ALL,";
            writeScore(metrics, evaluate(data, t, tuning));
            metrics << '\n';
            for (const auto& gt : data.groundTruth) {
                if ((gt.first != config::tuningImage) != tuning)
                    continue;
                metrics << item.first << ',' << t.id << ',' << split << ',' << gt.first << ',';
                writeScore(metrics, evaluate(data, t, tuning, gt.first));
                metrics << '\n';
            }
            std::map<std::tuple<std::string, std::string, std::string>, int> counts;
            for (const std::string actual : {"White", "Solid", "Striped"})
                for (const std::string predicted : {"White", "Solid", "Striped", "Unknown"})
                    counts[{"type", actual, predicted}] = 0;
            for (int actual = 0; actual <= 15; ++actual)
                for (int predicted = -1; predicted <= 15; ++predicted)
                    counts[{"number", std::to_string(actual), std::to_string(predicted)}] = 0;
            for (const auto& c : data.candidates) {
                if (c.tuning != tuning)
                    continue;
                std::string decision;
                const auto type = classify(c, t, &decision);
                const auto recognition = t.original ? RecognitionParameters{} : ccaRecognitionBase();
                const int number = classifyExperimentalNumber(t.original ? c.original : c.tuned, type.type, recognition);
                predictions << item.first << ',' << t.id << ',' << c.image << ',' << c.id << ','
                    << c.truth << ',' << trueType(c.truth) << ',' << type.type << ',' << number << ','
                    << c.tunedType.type << ',' << c.tunedNumber << ',';
                if (c.truth >= 0) {
                    predictions << (type.type == trueType(c.truth)) << ',' << (number == c.truth);
                    ++counts[{"type", trueType(c.truth), type.type}];
                    ++counts[{"number", std::to_string(c.truth), std::to_string(number)}];
                } else {
                    predictions << ',';
                }
                const auto& f = features(c, t);
                predictions << ',' << (type.type != c.tunedType.type) << ',' << decision << ','
                    << type.lightPercent << ',' << type.colouredPercent + type.darkPercent << ','
                    << f.lightComponents << ',' << f.largestLightComponentPercent << ','
                    << f.nonLightComponents << ',' << f.largestNonLightComponentPercent << ','
                    << f.lightBoundaryCoverage << '\n';
            }
            for (const auto& count : counts)
                confusion << item.first << ',' << split << ',' << std::get<0>(count.first) << ','
                    << std::get<1>(count.first) << ',' << std::get<2>(count.first) << ',' << count.second << '\n';
        }
    }
    visuals(directory, data, finalists);
    auto summary = output(directory / "summary.md");
    auto ceilingCSV = output(directory / "number_ceiling.csv");
    ceilingCSV << "split,image,matched,oracleNumberCorrect\n";
    int tuningCeiling = 0, tuningMatched = 0;
    for (bool tuning : {true, false}) {
        int total = 0, matched = 0;
        for (const auto& gt : data.groundTruth) {
            if ((gt.first != config::tuningImage) != tuning)
                continue;
            int imageMatched = 0, correct = 0;
            for (const auto& c : data.candidates) {
                if (c.image != gt.first || c.truth < 0)
                    continue;
                ++imageMatched;
                correct += classifyExperimentalNumber(c.tuned, trueType(c.truth), ccaRecognitionBase()) == c.truth;
            }
            ceilingCSV << (tuning ? "tuning" : "reference") << ',' << gt.first << ',' << imageMatched << ',' << correct << '\n';
            total += correct;
            matched += imageMatched;
        }
        ceilingCSV << (tuning ? "tuning" : "reference") << ",ALL," << matched << ',' << total << '\n';
        if (tuning) {
            tuningCeiling = total;
            tuningMatched = matched;
        }
    }
    summary << "# CCA recognition search\n\n"
        << "Search base: frozen configuration 2153 (K=6, light V>=170/S<=125, cue>=85%, "
        << "solid>=80%, blue/purple hue boundary 125), no padding, no cloth filter.\n\n"
        << "Tuning: Table1-6. Reference only: AllBalls. These are exploratory tuning results.\n\n"
        << trials.size() << " trials including four controls; 12 feature settings, all eight rule masks, "
        << "two passes of grouped cue/stripe/solid threshold searches. Repeated configurations are retained.\n\n"
        << "Rule mask bits: cue=1, stripe=2, solid=4. Rule order: cue, stripe, solid, baseline fallback.\n\n"
        << "Ranking: correct numbers, correct types, fewer number regressions, fewer type regressions, "
        << "fewer changed types, fewer enabled rules, smaller departure from original CCA parameters, earlier ID. "
        << "The best-type ranking swaps the first two criteria. Regression counts always compare with tuned K=6.\n\n"
        << "| Configuration | Type (tuning) | Number (tuning) | Type fixed/worsened | Number fixed/worsened | Type/number (reference) |\n"
        << "|---|---:|---:|---:|---:|---:|\n";
    for (const auto& item : finalists) {
        const auto& s = item.second.score;
        const auto ref = evaluate(data, item.second, false);
        summary << '|' << item.first << " (" << item.second.id << ")|" << s.type << '/' << s.matched
            << '|' << s.number << '/' << s.matched << '|' << s.typeFixed << '/' << s.typeWorsened
            << '|' << s.numberFixed << '/' << s.numberWorsened << '|'
            << ref.type << '/' << ref.number << " (of " << ref.matched << ")|\n";
    }
    summary << "\nChanging CCA only changes type and the downstream type-dependent number decision. "
        << "It does not change cluster colours, body selection or hue thresholds. Small-component filtering "
        << "affects component statistics, not raw light percentages or boundary coverage. "
        << "K=6 features use the same light definition as configuration 2153; the original K=2 control "
        << "uses the production light definition. Default features and hybrid predictions were checked "
        << "against production on every candidate. Main application files remain unchanged.\n";
    summary << "\nWith K=6 segmentation and number rules fixed, supplying the true type correctly numbers "
        << tuningCeiling << '/' << tuningMatched << " tuning balls. This is the full-number ceiling for type-only changes "
        << "on these cached candidates, not an upper bound for a different colour/extraction pipeline.\n";
}

} // namespace

void runCCAGridSearch(const std::filesystem::path& directory) {
    std::filesystem::create_directories(directory);
    const CCASearchGrid grid;
    const auto data = prepare(grid, directory);
    std::vector<Trial> trials;
    const auto add = [&](CCAParameters p, bool original, const std::string& stage) {
        Trial t;
        t.id = static_cast<int>(trials.size() + 1);
        t.p = p;
        t.original = original;
        t.stage = stage;
        t.score = evaluate(data, t, true);
        trials.push_back(t);
        return static_cast<int>(trials.size() - 1);
    };
    CCAParameters off;
    off.cue = off.stripe = off.solid = false;
    const int original = add(off, true, "control");
    const int originalCCA = add({}, true, "control");
    const int tuned = add(off, false, "control");
    const int tunedCCA = add({}, false, "control");
    auto seeds = output(directory / "seeds.csv");
    seeds << "minComponentPercent,boundaryKernel,ruleMask,seedId,finalId\n";
    for (double area : grid.componentPercent) {
        for (int kernel : grid.boundaryKernel) {
            for (int mask = 0; mask < 8; ++mask) {
                CCAParameters p;
                p.minComponentPercent = area;
                p.boundaryKernel = kernel;
                p.cue = (mask & 1) != 0;
                p.stripe = (mask & 2) != 0;
                p.solid = (mask & 4) != 0;
                const int seed = add(p, false, "seed");
                int best = seed;
                for (int pass = 1; pass <= grid.passes; ++pass) {
                    const std::string prefix = "seed" + std::to_string(trials[seed].id)
                        + "_pass" + std::to_string(pass) + '_';
                    using P = CCAParameters;
                    if (p.cue)
                        best = group(data, trials, best, {
                            {&P::cueMinLight, &grid.cueMinLight}, {&P::cueLargestLight, &grid.cueLargestLight},
                            {&P::cueMinBoundary, &grid.cueMinBoundary}}, prefix + "cue");
                    if (p.stripe)
                        best = group(data, trials, best, {
                            {&P::stripeLargestLight, &grid.stripeLargestLight},
                            {&P::stripeMinBoundary, &grid.stripeMinBoundary},
                            {&P::stripeMinNonLight, &grid.stripeMinNonLight}}, prefix + "stripe");
                    if (p.solid)
                        best = group(data, trials, best, {
                            {&P::solidMinNonLight, &grid.solidMinNonLight},
                            {&P::solidMaxBoundary, &grid.solidMaxBoundary},
                            {&P::solidMaxComponents, &grid.solidMaxComponents},
                            {&P::solidLargestNonLight, &grid.solidLargestNonLight}}, prefix + "solid");
                }
                seeds << area << ',' << kernel << ',' << mask << ',' << trials[seed].id << ',' << trials[best].id << '\n';
            }
            std::cout << "Searched CCA area " << area << "% / kernel " << kernel
                      << " / all rule masks\n" << std::flush;
        }
    }
    int bestNumber = tuned, bestType = tuned;
    // The original K=2 profiles are controls only, not competitors in this search.
    for (int i = 0; i < static_cast<int>(trials.size()); ++i) {
        if (trials[i].original)
            continue;
        if (better(trials[i], trials[bestNumber]))
            bestNumber = i;
        if (better(trials[i], trials[bestType], true))
            bestType = i;
    }
    exportResults(data, trials, {{"original_k2", trials[original]}, {"original_k2_cca", trials[originalCCA]},
        {"tuned_k6", trials[tuned]}, {"tuned_k6_cca", trials[tunedCCA]},
        {"best_number", trials[bestNumber]}, {"best_type", trials[bestType]}}, directory);
    std::cout << "CCA search complete: " << trials.size() << " trials; best number "
        << trials[bestNumber].score.number << '/' << trials[bestNumber].score.matched
        << ", best type " << trials[bestType].score.type << '/' << trials[bestType].score.matched
        << " -> " << directory << '\n';
}

} // namespace experiments
