#include "recognitionGridSearch.h"
#include "radiusPadding.h"
#include "config.h"
#include "detection/ballDetection.h"
#include "evaluation/evaluation.h"
#include "geometry/perspectiveTransform.h"

#include <algorithm>
#include <array>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace experiments {

BallTypeResult classifyExperimentalType(const KMeansResult& segmentation,
                                       const RecognitionParameters& p) {
    BallTypeResult result;
    if (segmentation.regions.empty())
        return result;
    for (const auto& region : segmentation.regions) {
        if (region.hsv[2] >= p.lightMinV && region.hsv[1] <= p.lightMaxS)
            result.lightPercent += region.percentage;
        else if (region.hsv[2] <= p.darkMaxV)
            result.darkPercent += region.percentage;
        else
            result.colouredPercent += region.percentage;
    }
    if (result.lightPercent >= p.whiteMinPercent)
        result.type = "White";
    else if (result.colouredPercent + result.darkPercent >= p.solidMinPercent)
        result.type = "Solid";
    else if (result.lightPercent > 0 && result.colouredPercent + result.darkPercent > 0)
        result.type = "Striped";
    return result;
}

int classifyExperimentalNumber(const KMeansResult& segmentation, const std::string& type,
                               const RecognitionParameters& p) {
    if (segmentation.regions.empty())
        return -1;
    if (type == "White")
        return 0;
    if (type != "Solid" && type != "Striped")
        return -1;
    const KMeansRegion* body = nullptr;
    for (const auto& region : segmentation.regions) {
        if (region.hsv[2] >= p.lightMinV && region.hsv[1] <= p.lightMaxS)
            continue;
        if (region.percentage < p.bodyMinPercent)
            continue;
        if (!body || region.percentage > body->percentage)
            body = &region;
    }
    if (!body)
        return -1;
    const int h = body->hsv[0], s = body->hsv[1], v = body->hsv[2];
    if (s < p.blackMaxS)
        return v <= p.blackMaxV && type == "Solid" ? 8 : -1;
    int number = -1;
    if (h < p.redUpper || h >= p.redWrap)
        number = v < p.maroonMaxV ? 7 : 3;
    else if (h < p.orangeUpper)
        number = 5;
    else if (h < p.yellowUpper)
        number = 1;
    else if (h >= p.greenLower && h < p.greenUpper)
        number = 6;
    else if (h >= p.blueLower && h < p.bluePurple)
        number = 2;
    else if (h >= p.bluePurple && h < p.purpleUpper)
        number = 4;
    return number < 0 ? -1 : type == "Striped" ? number + 8 : number;
}

namespace {

std::string trueType(int number) {
    return number < 0 ? "Unknown" : number == 0 ? "White" : number <= 8 ? "Solid" : "Striped";
}

struct Candidate {
    int id = 0;
    int truth = -1;
    cv::Vec3f circle;
    std::array<BallROI, 2> balls;
    std::map<std::pair<int, bool>, KMeansResult> segmentations;
};

struct ImageData {
    std::string name;
    bool tuning = false;
    int gt = 0;
    int matched = 0;
    int falsePositives = 0;
    int missed = 0;
    std::vector<Candidate> candidates;
};

struct Score {
    int matched = 0;
    int gt = 0;
    int type = 0;
    int number = 0;
    int oracle = 0;
};

struct Trial {
    int id = 0;
    std::string stage;
    int parent = 0;
    RecognitionParameters parameters;
    Score score;
};

std::ofstream outputFile(const std::filesystem::path& path) {
    std::ofstream stream(path);
    stream.exceptions(std::ios::failbit | std::ios::badbit);
    stream << std::fixed << std::setprecision(6);
    return stream;
}

const KMeansResult& segmentation(const Candidate& c, const RecognitionParameters& p) {
    return c.segmentations.at({p.k, p.cloth});
}

// Detect once and cache only cluster features, rather than retaining thousands
// of cluster masks/segmented images that threshold evaluation never uses.
std::vector<ImageData> prepare(const RecognitionSearchGrid& grid,
                               const std::filesystem::path& directory) {
    std::vector<ImageData> data;
    auto detectionCSV = outputFile(directory / "detection_counts.csv");
    detectionCSV << "image,split,groundTruth,matched,falsePositives,missed\n";
    auto cacheCSV = outputFile(directory / "cluster_features.csv");
    cacheCSV << "image,candidate,trueNumber,k,cloth,cluster,percent,B,G,R,H,S,V\n";
    auto masksCSV = outputFile(directory / "mask_counts.csv");
    masksCSV << "image,candidate,trueNumber,cloth,pixels\n";
    for (const auto& filename : config::allImages()) {
        const auto image = cv::imread(config::mediaDirectory + filename);
        if (image.empty())
            throw std::runtime_error("Could not load " + filename);
        const auto truth = loadLabelledGroundTruth(config::groundTruthPath, filename);
        if (truth.empty())
            throw std::runtime_error("No ground truth for " + filename);
        const auto detection = runDetection(image, HoughParams{});
        std::vector<cv::Point2f> centres;
        for (const auto& ball : truth)
            centres.push_back(ball.centre);
        const auto matching = evaluateDetections(transformCentres(detection.filteredCircles),
                                                 centres, config::matchTolerance);
        ImageData entry;
        entry.name = filename;
        entry.tuning = filename != config::tuningImage;
        entry.gt = static_cast<int>(truth.size());
        entry.matched = matching.TP;
        entry.falsePositives = matching.FP;
        entry.missed = matching.FN;
        detectionCSV << filename << ',' << (entry.tuning ? "tuning" : "reference") << ','
                     << entry.gt << ',' << matching.TP << ',' << matching.FP << ',' << matching.FN << '\n';
        for (size_t i = 0; i < detection.filteredCircles.size(); ++i) {
            Candidate c;
            c.id = static_cast<int>(i + 1);
            c.circle = detection.filteredCircles[i];
            const int match = matching.predictionToGroundTruth[i];
            if (match >= 0)
                c.truth = truth[match].number;
            for (bool cloth : grid.cloth) {
                RadiusPaddingOptions options;
                options.filterCloth = cloth;
                c.balls[cloth] = extractPaddedBall(image, c.circle, options);
                masksCSV << filename << ',' << c.id << ',' << c.truth << ',' << cloth
                         << ',' << cv::countNonZero(c.balls[cloth].mask) << '\n';
                for (int k : grid.ks) {
                    auto result = segmentBallKMeans(c.balls[cloth], k);
                    // Verify the parameterised rules reproduce production before searching.
                    RecognitionParameters defaults;
                    const auto expectedType = classifyBallType(result);
                    const auto actualType = classifyExperimentalType(result, defaults);
                    if (actualType.type != expectedType.type ||
                        actualType.lightPercent != expectedType.lightPercent ||
                        actualType.darkPercent != expectedType.darkPercent ||
                        actualType.colouredPercent != expectedType.colouredPercent ||
                        classifyExperimentalNumber(result, actualType.type, defaults) !=
                            classifyBallNumber(result, expectedType))
                        throw std::runtime_error("Experimental default rules differ from production");
                    result.segmentedImage.release();
                    for (size_t r = 0; r < result.regions.size(); ++r) {
                        auto& region = result.regions[r];
                        region.mask.release();
                        cacheCSV << filename << ',' << c.id << ',' << c.truth << ',' << k << ','
                                 << cloth << ',' << r + 1 << ',' << region.percentage;
                        for (int channel = 0; channel < 3; ++channel)
                            cacheCSV << ',' << int(region.colour[channel]);
                        for (int channel = 0; channel < 3; ++channel)
                            cacheCSV << ',' << int(region.hsv[channel]);
                        cacheCSV << '\n';
                    }
                    c.segmentations.emplace(std::make_pair(k, cloth), std::move(result));
                }
            }
            entry.candidates.push_back(std::move(c));
        }
        data.push_back(std::move(entry));
        std::cout << "Cached " << filename << ": " << matching.TP << " matched, "
                  << matching.FP << " unmatched candidates\n" << std::flush;
    }
    return data;
}

Score score(const std::vector<ImageData>& data, const RecognitionParameters& p,
            bool tuning, const std::string& imageName = "") {
    Score result;
    for (const auto& image : data) {
        if (image.tuning != tuning || (!imageName.empty() && image.name != imageName))
            continue;
        result.gt += image.gt;
        for (const auto& c : image.candidates) {
            if (c.truth < 0)
                continue;
            const auto& clusters = segmentation(c, p);
            const auto type = classifyExperimentalType(clusters, p);
            const auto actual = trueType(c.truth);
            ++result.matched;
            result.type += type.type == actual;
            result.number += classifyExperimentalNumber(clusters, type.type, p) == c.truth;
            result.oracle += classifyExperimentalNumber(clusters, actual, p) == c.truth;
        }
    }
    return result;
}

bool valid(const RecognitionParameters& p) {
    return 0 < p.redUpper && p.redUpper < p.orangeUpper && p.orangeUpper < p.yellowUpper &&
           p.yellowUpper < p.greenLower && p.greenLower < p.greenUpper &&
           p.greenUpper < p.blueLower && p.blueLower < p.bluePurple &&
           p.bluePurple < p.purpleUpper && p.purpleUpper < p.redWrap && p.redWrap < 180;
}

// Resolve equal measured scores by preferring smaller departures from the
// baseline, expressed in the grid's step units. Trial ID breaks remaining ties.
double departure(const RecognitionParameters& p) {
    return (p.k - 2) + int(p.cloth) + std::abs(p.lightMinV - 170) / 20.0 +
        std::abs(p.lightMaxS - 165) / 20.0 + std::abs(p.whiteMinPercent - 85) / 5.0 +
        std::abs(p.solidMinPercent - 70) / 5.0 + std::abs(p.redUpper - 6) / 2.0 +
        std::abs(p.orangeUpper - 15) / 3.0 + std::abs(p.yellowUpper - 35) / 5.0 +
        std::abs(p.greenLower - 45) / 5.0 + std::abs(p.greenUpper - 90) / 5.0 +
        std::abs(p.blueLower - 100) / 5.0 + std::abs(p.bluePurple - 130) / 5.0 +
        std::abs(p.purpleUpper - 165) / 5.0 + std::abs(p.redWrap - 170) / 2.0 +
        std::abs(p.maroonMaxV - 150) / 20.0 + std::abs(p.blackMaxS - 60) / 20.0 +
        std::abs(p.blackMaxV - 90) / 20.0 + std::abs(p.bodyMinPercent - 2);
}

bool better(const Trial& a, const Trial& b, bool typeFirst = false) {
    const auto key = [typeFirst](const Trial& t) {
        return std::make_tuple(typeFirst ? t.score.type : t.score.number,
                               typeFirst ? t.score.number : t.score.type,
                               t.score.oracle, -departure(t.parameters), -t.id);
    };
    return key(a) > key(b);
}

const char* parameterHeader =
    "k,cloth,lightMinV,lightMaxS,darkMaxV,whiteMinPercent,solidMinPercent,"
    "redUpper,orangeUpper,yellowUpper,greenLower,greenUpper,blueLower,bluePurple,"
    "purpleUpper,redWrap,maroonMaxV,blackMaxS,blackMaxV,bodyMinPercent";

void writeParameters(std::ostream& out, const RecognitionParameters& p) {
    out << p.k << ',' << p.cloth << ',' << p.lightMinV << ',' << p.lightMaxS << ','
        << p.darkMaxV << ',' << p.whiteMinPercent << ',' << p.solidMinPercent << ','
        << p.redUpper << ',' << p.orangeUpper << ',' << p.yellowUpper << ','
        << p.greenLower << ',' << p.greenUpper << ',' << p.blueLower << ','
        << p.bluePurple << ',' << p.purpleUpper << ',' << p.redWrap << ','
        << p.maroonMaxV << ',' << p.blackMaxS << ',' << p.blackMaxV << ',' << p.bodyMinPercent;
}

double ratio(int numerator, int denominator) {
    return denominator ? double(numerator) / denominator : 0.0;
}

void writeScore(std::ostream& out, const Score& s) {
    out << s.gt << ',' << s.matched << ',' << s.type << ',' << s.number << ',' << s.oracle
        << ',' << ratio(s.type, s.matched) << ',' << ratio(s.number, s.matched)
        << ',' << ratio(s.oracle, s.matched) << ',' << ratio(s.number, s.gt);
}

const char* scoreHeader = "groundTruth,matched,typeCorrect,numberCorrect,oracleNumberCorrect,"
                          "typeAccuracy,numberAccuracy,oracleNumberAccuracy,endToEndNumberRecall";

void leaderboard(const std::filesystem::path& path, std::vector<Trial> trials) {
    std::stable_sort(trials.begin(), trials.end(), [](const Trial& a, const Trial& b) {
        return better(a, b);
    });
    auto csv = outputFile(path);
    csv << "rank,configId,stage,parentId," << parameterHeader << ',' << scoreHeader << '\n';
    for (size_t i = 0; i < trials.size(); ++i) {
        const auto& t = trials[i];
        csv << i + 1 << ',' << t.id << ',' << t.stage << ',' << t.parent << ',';
        writeParameters(csv, t.parameters);
        csv << ',';
        writeScore(csv, t.score);
        csv << '\n';
    }
}

// Grouped Cartesian searches: keep other parameters fixed, select the best
// candidate in a group, then move to the next group. Two passes revisit earlier
// decisions after later thresholds change. This is not a global exhaustive grid.
using Field = std::pair<int RecognitionParameters::*, const std::vector<int>*>;

int searchGroup(const std::vector<ImageData>& data, std::vector<Trial>& trials,
                int startingIndex, const std::vector<Field>& fields,
                const std::string& stage) {
    const Trial start = trials.at(startingIndex);
    int best = startingIndex;
    RecognitionParameters p = start.parameters;
    const auto visit = [&](const auto& self, size_t depth) -> void {
        if (depth < fields.size()) {
            for (int value : *fields[depth].second) {
                p.*(fields[depth].first) = value;
                self(self, depth + 1);
            }
        } else if (valid(p)) {
            Trial trial;
            trial.id = static_cast<int>(trials.size() + 1);
            trial.stage = stage;
            trial.parent = start.id;
            trial.parameters = p;
            trial.score = score(data, p, true);
            trials.push_back(trial);
            const int index = static_cast<int>(trials.size() - 1);
            if (better(trials[index], trials[best]))
                best = index;
        }
    };
    visit(visit, 0);
    return best;
}

void exportFinalists(const std::filesystem::path& directory,
                     const std::vector<ImageData>& data,
                     const std::vector<std::pair<std::string, Trial>>& finalists) {
    auto configs = outputFile(directory / "finalists.csv");
    configs << "label,configId," << parameterHeader << '\n';
    auto metrics = outputFile(directory / "metrics.csv");
    metrics << "label,configId,split,image," << scoreHeader << '\n';
    auto predictions = outputFile(directory / "predictions.csv");
    predictions << "label,configId,image,candidate,trueNumber,trueType,type,number,oracleNumber,"
                   "typeCorrect,numberCorrect,oracleNumberCorrect\n";
    auto confusion = outputFile(directory / "confusion.csv");
    confusion << "label,configId,split,task,actual,predicted,count\n";
    for (const auto& item : finalists) {
        const auto& label = item.first;
        const auto& trial = item.second;
        configs << label << ',' << trial.id << ',';
        writeParameters(configs, trial.parameters);
        configs << '\n';
        for (bool tuning : {true, false}) {
            const std::string split = tuning ? "tuning" : "reference";
            metrics << label << ',' << trial.id << ',' << split << ",ALL,";
            writeScore(metrics, score(data, trial.parameters, tuning));
            metrics << '\n';
            std::map<std::tuple<std::string, std::string, std::string>, int> counts;
            // Explicit zero entries make matrix dimensions unambiguous.
            for (const std::string actual : {"White", "Solid", "Striped"})
                for (const std::string predicted : {"White", "Solid", "Striped", "Unknown"})
                    counts[{"type", actual, predicted}] = 0;
            for (const auto& task : {"number", "oracleNumber"})
                for (int actual = 0; actual <= 15; ++actual)
                    for (int predicted = -1; predicted <= 15; ++predicted)
                        counts[{task, std::to_string(actual), std::to_string(predicted)}] = 0;
            for (const auto& image : data) {
                if (image.tuning != tuning)
                    continue;
                metrics << label << ',' << trial.id << ',' << split << ',' << image.name << ',';
                writeScore(metrics, score(data, trial.parameters, tuning, image.name));
                metrics << '\n';
                for (const auto& c : image.candidates) {
                    const auto& clusters = segmentation(c, trial.parameters);
                    const auto predicted = classifyExperimentalType(clusters, trial.parameters);
                    const int number = classifyExperimentalNumber(clusters, predicted.type, trial.parameters);
                    const int oracle = c.truth < 0 ? -1
                        : classifyExperimentalNumber(clusters, trueType(c.truth), trial.parameters);
                    predictions << label << ',' << trial.id << ',' << image.name << ',' << c.id
                                << ',' << c.truth << ',' << trueType(c.truth) << ',' << predicted.type
                                << ',' << number << ',' << oracle << ',';
                    if (c.truth >= 0) {
                        predictions << (predicted.type == trueType(c.truth)) << ','
                                    << (number == c.truth) << ',' << (oracle == c.truth);
                        ++counts[{"type", trueType(c.truth), predicted.type}];
                        ++counts[{"number", std::to_string(c.truth), std::to_string(number)}];
                        ++counts[{"oracleNumber", std::to_string(c.truth), std::to_string(oracle)}];
                    } else {
                        predictions << ",,";
                    }
                    predictions << '\n';
                }
            }
            for (const auto& count : counts)
                confusion << label << ',' << trial.id << ',' << split << ','
                          << std::get<0>(count.first) << ',' << std::get<1>(count.first) << ','
                          << std::get<2>(count.first) << ',' << count.second << '\n';
        }
    }

    constexpr int width = 240, height = 310, previewSize = 145, header = 55;
    std::vector<std::pair<std::string, Trial>> visualFinalists;
    std::set<int> visualIds;
    for (const auto& item : finalists)
        if (visualIds.insert(item.second.id).second)
            visualFinalists.push_back(item);
    for (const auto& image : data) {
        const auto imageDirectory = directory / std::filesystem::path(image.name).stem();
        std::filesystem::create_directories(imageDirectory);
        for (const auto& c : image.candidates) {
            cv::Mat comparison(header + height, width * static_cast<int>(visualFinalists.size()),
                               CV_8UC3, cv::Scalar(35, 35, 35));
            const std::string expected = c.truth < 0 ? "No ground-truth match"
                : "Expected: " + trueType(c.truth) + " #" + std::to_string(c.truth);
            cv::putText(comparison, image.name + " / candidate " + std::to_string(c.id)
                        + " / " + expected, {10, 26}, cv::FONT_HERSHEY_SIMPLEX,
                        0.55, {255, 255, 255}, 1);
            for (size_t j = 0; j < visualFinalists.size(); ++j) {
                const auto& item = visualFinalists[j];
                const auto& p = item.second.parameters;
                const int left = static_cast<int>(j) * width;
                const auto& ball = c.balls[p.cloth];
                cv::Mat isolated = cv::Mat::zeros(ball.image.size(), CV_8UC3), preview;
                ball.image.copyTo(isolated, ball.mask);
                const double scale = double(previewSize) / std::max(isolated.cols, isolated.rows);
                cv::resize(isolated, preview, {}, scale, scale, cv::INTER_NEAREST);
                preview.copyTo(comparison(cv::Rect(left + (width - preview.cols) / 2,
                                                   header + 60, preview.cols, preview.rows)));
                const auto& clusters = segmentation(c, p);
                const auto type = classifyExperimentalType(clusters, p);
                const int number = classifyExperimentalNumber(clusters, type.type, p);
                const auto text = [&](const std::string& value, int y, cv::Scalar colour) {
                    cv::putText(comparison, value, {left + 10, header + y},
                                cv::FONT_HERSHEY_SIMPLEX, 0.5, colour, 1);
                };
                text(item.first + " (" + std::to_string(item.second.id) + ")", 5, {255, 255, 255});
                text("K=" + std::to_string(p.k) + (p.cloth ? " / cloth on" : " / cloth off"),
                     30, {220, 220, 220});
                const auto status = [&](bool correct) {
                    return c.truth < 0 ? " N/A" : correct ? " OK" : " WRONG";
                };
                const auto colour = [&](bool correct) {
                    return c.truth < 0 ? cv::Scalar(180, 180, 180)
                        : correct ? cv::Scalar(90, 220, 90) : cv::Scalar(80, 80, 255);
                };
                text("Type: " + type.type + status(type.type == trueType(c.truth)),
                     245, colour(type.type == trueType(c.truth)));
                text("Number: " + std::to_string(number) + status(number == c.truth),
                     275, colour(number == c.truth));
            }
            const auto path = imageDirectory / ("ball_" + std::to_string(c.id) + "_comparison.png");
            if (!cv::imwrite(path.string(), comparison))
                throw std::runtime_error("Could not save " + path.string());
        }
    }
}

} // namespace

void runRecognitionGridSearch(const std::filesystem::path& directory) {
    std::filesystem::create_directories(directory);
    const RecognitionSearchGrid grid;
    const auto data = prepare(grid, directory);
    std::vector<Trial> trials;
    for (int k : grid.ks)
        for (bool cloth : grid.cloth)
            for (int v : grid.lightMinV)
                for (int s : grid.lightMaxS)
                    for (int white : grid.whiteMinPercent)
                        for (int solid : grid.solidMinPercent) {
                            RecognitionParameters p;
                            p.k = k;
                            p.cloth = cloth;
                            p.lightMinV = v;
                            p.lightMaxS = s;
                            p.whiteMinPercent = white;
                            p.solidMinPercent = solid;
                            trials.push_back({static_cast<int>(trials.size() + 1), "type_grid", 0,
                                              p, score(data, p, true)});
                        }
    const int stageOneCount = static_cast<int>(trials.size());
    if (!stageOneCount)
        throw std::runtime_error("Empty recognition grid");
    leaderboard(directory / "type_leaderboard.csv", trials);
    auto kCSV = outputFile(directory / "k_comparison.csv");
    kCSV << "comparison,configId," << parameterHeader << ',' << scoreHeader << '\n';
    for (int k : grid.ks) {
        for (bool cloth : grid.cloth) {
            int best = -1, defaults = -1;
            for (int i = 0; i < stageOneCount; ++i) {
                const auto& p = trials[i].parameters;
                if (p.k != k || p.cloth != cloth)
                    continue;
                if (best < 0 || better(trials[i], trials[best]))
                    best = i;
                if (p.lightMinV == 170 && p.lightMaxS == 165 &&
                    p.whiteMinPercent == 85 && p.solidMinPercent == 70)
                    defaults = i;
            }
            if (best < 0 || defaults < 0)
                throw std::runtime_error("Each K/filter pair must include default thresholds");
            for (const auto& choice : {std::make_pair("default_thresholds", defaults),
                                       std::make_pair("best_type_grid", best)}) {
                kCSV << choice.first << ',' << trials[choice.second].id << ',';
                writeParameters(kCSV, trials[choice.second].parameters);
                kCSV << ',';
                writeScore(kCSV, trials[choice.second].score);
                kCSV << '\n';
            }
        }
    }
    std::vector<int> ranking;
    for (int i = 0; i < stageOneCount; ++i)
        ranking.push_back(i);
    std::vector<int> seeds;
    const auto add = [&](int index) {
        if (std::find(seeds.begin(), seeds.end(), index) == seeds.end())
            seeds.push_back(index);
    };
    for (bool typeFirst : {false, true}) {
        std::stable_sort(ranking.begin(), ranking.end(), [&](int a, int b) {
            return better(trials[a], trials[b], typeFirst);
        });
        std::set<std::pair<int, bool>> selected;
        for (int index : ranking) {
            const auto& p = trials[index].parameters;
            if (selected.insert({p.k, p.cloth}).second)
                add(index);
            if (static_cast<int>(selected.size()) == grid.shortlistPerObjective)
                break;
        }
    }
    int baseline = -1, baselineCloth = -1;
    for (int i = 0; i < stageOneCount; ++i) {
        const auto& p = trials[i].parameters;
        if (p.k == 2 && p.lightMinV == 170 && p.lightMaxS == 165 &&
            p.whiteMinPercent == 85 && p.solidMinPercent == 70) {
            (p.cloth ? baselineCloth : baseline) = i;
            add(i);
        }
    }
    if (baseline < 0 || baselineCloth < 0)
        throw std::runtime_error("Grid must include both production baseline settings");

    auto shortCSV = outputFile(directory / "shortlist.csv");
    shortCSV << "seedId,finalId\n";
    for (int seed : seeds) {
        int best = seed;
        for (int pass = 1; pass <= grid.colourPasses; ++pass) {
            const std::string prefix = "seed" + std::to_string(trials[seed].id)
                + "_pass" + std::to_string(pass) + '_';
            using P = RecognitionParameters;
            best = searchGroup(data, trials, best, {
                {&P::redUpper, &grid.redUpper}, {&P::orangeUpper, &grid.orangeUpper},
                {&P::yellowUpper, &grid.yellowUpper}, {&P::redWrap, &grid.redWrap},
                {&P::maroonMaxV, &grid.maroonMaxV}}, prefix + "warm");
            best = searchGroup(data, trials, best, {
                {&P::greenLower, &grid.greenLower}, {&P::greenUpper, &grid.greenUpper},
                {&P::blueLower, &grid.blueLower}, {&P::bluePurple, &grid.bluePurple},
                {&P::purpleUpper, &grid.purpleUpper}}, prefix + "cool");
            best = searchGroup(data, trials, best, {
                {&P::blackMaxS, &grid.blackMaxS}, {&P::blackMaxV, &grid.blackMaxV},
                {&P::bodyMinPercent, &grid.bodyMinPercent}}, prefix + "neutral");
        }
        shortCSV << trials[seed].id << ',' << trials[best].id << '\n';
        std::cout << "Colour search seed " << trials[seed].id << " -> " << trials[best].id
                  << ": " << trials[best].score.number << '/' << trials[best].score.matched
                  << " numbers correct\n" << std::flush;
    }
    leaderboard(directory / "colour_leaderboard.csv",
                std::vector<Trial>(trials.begin() + stageOneCount, trials.end()));
    int bestNumber = baseline, bestType = baseline, bestStageOne = baseline;
    for (int i = 0; i < static_cast<int>(trials.size()); ++i) {
        if (better(trials[i], trials[bestNumber]))
            bestNumber = i;
        if (better(trials[i], trials[bestType], true))
            bestType = i;
        if (i < stageOneCount && better(trials[i], trials[bestStageOne]))
            bestStageOne = i;
    }
    const std::vector<std::pair<std::string, Trial>> finalists = {
        {"baseline", trials[baseline]}, {"baseline_cloth", trials[baselineCloth]},
        {"best_type_grid", trials[bestStageOne]}, {"best_number", trials[bestNumber]},
        {"best_type", trials[bestType]}};
    exportFinalists(directory, data, finalists);
    auto summary = outputFile(directory / "summary.md");
    summary << "# Recognition grid search\n\n"
        << "Tuning: Table1-6. Reference only: AllBalls. Zero radius padding; no CCA.\n\n"
        << "Stage 1 evaluated " << stageOneCount << " configurations. Stage 2 evaluated "
        << trials.size() - stageOneCount << " valid grouped combinations from " << seeds.size()
        << " seeds over " << grid.colourPasses << " passes. Repeated trials are retained.\n\n"
        << "Rank: correct numbers, correct types, oracle numbers, smaller baseline departure, lower trial ID. "
        << "Best-type rank swaps the first two criteria. AllBalls never influences ranking.\n\n"
        << "| Configuration | Type (tuning) | Number (tuning) | Oracle number (tuning) | Number (reference) |\n"
        << "|---|---:|---:|---:|---:|\n";
    for (const auto& item : finalists) {
        const auto& s = item.second.score;
        const auto reference = score(data, item.second.parameters, false);
        summary << '|' << item.first << " (" << item.second.id << ")|" << s.type << '/' << s.matched
                << '|' << s.number << '/' << s.matched << '|' << s.oracle << '/' << s.matched
                << '|' << reference.number << '/' << reference.matched << "|\n";
    }
    summary << "\nThese are exploratory tuning results, not independent test accuracy. "
        << "The supplied images share the camera setup and some unchanged balls. "
        << "Oracle-number accuracy supplies the true type, including the cue-ball shortcut; "
        << "it is a diagnostic, not a deployable classifier. Unmatched candidates have N/A correctness "
        << "and are excluded from recognition accuracy; missed balls remain in end-to-end recall's denominator.\n\n"
        << "Default experimental rules were checked against production for every cached segmentation. "
        << "Production files and parameters were not changed.\n";
    std::cout << "Search complete: " << trials.size() << " trials -> " << directory << '\n';
}

} // namespace experiments
