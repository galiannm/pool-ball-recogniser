#include "recognitionTuner.h"
#include "recognitionGridSearch.h"
#include "ccaSearchConfig.h"
#include "config.h"
#include "detection/ballDetection.h"
#include "evaluation/evaluation.h"
#include "geometry/perspectiveTransform.h"
#include "io/finalResults.h"
#include "io/output.h"
#include "visualisation/combineImages.h"

#include <algorithm>
#include <array>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>

namespace experiments {
namespace {
const std::string controlsWindow = "AllBalls tuner - sliders and previews";
constexpr const char* tuningFilename = "AllBalls.jpeg"; // No test-image selector.

struct TunerState {
    RecognitionParameters p = ccaRecognitionBase();
    bool blackInclusive = false; // Preserve winner 2153; baseline preset uses <=.
    int selected = 1; // One-based Hough candidate, not the annotated ball identity.
    std::array<int, 10> values() const {
        return {p.k, p.lightMinV, p.lightMaxS, p.whiteMinPercent, p.solidMinPercent,
                p.blackMaxS, p.blackMaxV, blackInclusive, p.bluePurple, selected};
    }
};

struct TunerData {
    cv::Mat image;
    std::vector<cv::Vec3f> circles;
    std::vector<cv::Point2f> tableCentres;
    std::vector<BallROI> balls;
    std::vector<int> actual;
    int groundTruthCount = 0;
    std::map<int, std::vector<KMeansResult>> cache;
};

struct TunerFrame {
    cv::Mat overview, detail;
    std::vector<std::string> types;
    std::vector<int> numbers;
    int matched = 0, correctTypes = 0, correctNumbers = 0;
};

std::string label(int n) { return n == 0 ? "Cue" : n < 0 ? "Unknown" : std::to_string(n); }
std::string actualType(int n) { return n == 0 ? "White" : n <= 8 ? "Solid" : "Striped"; }

TunerData loadTunerData() {
    TunerData data;
    data.image = cv::imread(config::mediaDirectory + tuningFilename);
    if (data.image.empty()) throw std::runtime_error("Could not load AllBalls.jpeg");
    data.circles = runDetection(data.image, HoughParams{}).filteredCircles;
    if (data.circles.empty()) throw std::runtime_error("No AllBalls candidates to tune");
    data.tableCentres = transformCentres(data.circles);
    for (const auto& circle : data.circles) data.balls.push_back(extractBall(data.image, circle));
    const auto truth = loadLabelledGroundTruth(config::groundTruthPath, tuningFilename);
    if (truth.empty()) throw std::runtime_error("Missing AllBalls annotations");
    std::vector<cv::Point2f> truthCentres;
    for (const auto& ball : truth) truthCentres.push_back(ball.centre);
    const auto matches = evaluateDetections(data.tableCentres, truthCentres, config::matchTolerance);
    for (const int match : matches.predictionToGroundTruth)
        data.actual.push_back(match < 0 ? -1 : truth[match].number);
    data.groundTruthCount = static_cast<int>(truth.size());
    return data;
}

const std::vector<KMeansResult>& clustersFor(TunerData& data, int k) {
    auto found = data.cache.find(k);
    if (found == data.cache.end()) {
        std::vector<KMeansResult> clusters;
        for (const auto& ball : data.balls) clusters.push_back(segmentBallKMeans(ball, k));
        found = data.cache.emplace(k, std::move(clusters)).first;
    }
    return found->second;
}

void text(cv::Mat& image, const std::string& value, cv::Point at,
          cv::Scalar colour = {245, 245, 245}, double scale = 0.5) {
    cv::putText(image, value, at, cv::FONT_HERSHEY_SIMPLEX, scale, {0, 0, 0}, 3, cv::LINE_AA);
    cv::putText(image, value, at, cv::FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv::LINE_AA);
}

TunerFrame renderTuner(TunerData& data, const TunerState& state) {
    const auto& p = state.p;
    if (p.k < 2 || p.k > 8 || state.selected < 1 || size_t(state.selected) > data.balls.size())
        throw std::invalid_argument("Invalid tuner K or selected candidate");
    const auto& clusters = clustersFor(data, p.k);
    RecognitionParameters numberParameters = p;
    // The experimental number rule uses strict <; +1 expresses <= for integer S.
    if (state.blackInclusive) ++numberParameters.blackMaxS;
    TunerFrame frame;
    frame.overview = cv::Mat(data.image.rows + 80, data.image.cols, CV_8UC3, cv::Scalar(32, 32, 32));
    data.image.copyTo(frame.overview(cv::Rect(0, 80, data.image.cols, data.image.rows)));
    for (size_t i = 0; i < clusters.size(); ++i) {
        const auto type = classifyExperimentalType(clusters[i], p);
        const int number = classifyExperimentalNumber(clusters[i], type.type, numberParameters);
        frame.types.push_back(type.type);
        frame.numbers.push_back(number);
        const int actual = data.actual[i];
        frame.matched += actual >= 0;
        frame.correctTypes += actual >= 0 && type.type == actualType(actual);
        frame.correctNumbers += actual >= 0 && number == actual;
        const cv::Scalar colour = actual < 0 ? cv::Scalar(180, 180, 180) :
            actual == number ? cv::Scalar(40, 240, 40) : cv::Scalar(60, 60, 255);
        const cv::Point centre(cvRound(data.circles[i][0]), cvRound(data.circles[i][1]) + 80);
        cv::circle(frame.overview, centre, cvRound(data.circles[i][2]), colour, 2, cv::LINE_AA);
        if (int(i + 1) == state.selected)
            cv::circle(frame.overview, centre, cvRound(data.circles[i][2]) + 5, {255, 255, 0}, 2, cv::LINE_AA);
        text(frame.overview, "#" + std::to_string(i + 1) + ":" + label(number),
             centre + cv::Point(-20, -cvRound(data.circles[i][2]) - 7), colour);
    }
    text(frame.overview, "AllBalls ONLY | K=" + std::to_string(p.k) + " | types " +
        std::to_string(frame.correctTypes) + "/" + std::to_string(frame.matched) + " | numbers " +
        std::to_string(frame.correctNumbers) + "/" + std::to_string(frame.matched) + " | detections " +
        std::to_string(frame.matched) + "/" + std::to_string(data.groundTruthCount), {12, 25});
    text(frame.overview, "S: save | B: current K=2 preset | T: tuned K=6 preset | Q/Esc: quit | green correct, red wrong, cyan selected",
         {12, 53});

    const size_t selected = state.selected - 1;
    const auto& segmentation = clusters[selected];
    const auto& ball = data.balls[selected];
    cv::Mat light = cv::Mat::zeros(ball.mask.size(), CV_8UC1);
    cv::Mat black = light.clone(), body = light.clone();
    const KMeansRegion* chosen = nullptr;
    for (const auto& region : segmentation.regions) {
        const bool isLight = region.hsv[2] >= p.lightMinV && region.hsv[1] <= p.lightMaxS;
        if (isLight) light.setTo(255, region.mask);
        if (region.hsv[1] < numberParameters.blackMaxS && region.hsv[2] <= p.blackMaxV)
            black.setTo(255, region.mask);
        if (!isLight && region.percentage >= p.bodyMinPercent &&
            (!chosen || region.percentage > chosen->percentage)) chosen = &region;
    }
    if (chosen) chosen->mask.copyTo(body);
    auto enlarge = [](const cv::Mat& image) {
        cv::Mat result;
        cv::resize(image, result, {240, 240}, 0, 0, cv::INTER_NEAREST);
        return result;
    };
    const cv::Mat panels = combineImages({{"Original crop", enlarge(ball.image)},
        {"BGR K-means regions", enlarge(segmentation.segmentedImage)}, {"Light clusters (HSV V/S)", enlarge(light)},
        {"Black-like clusters (S/V)", enlarge(black)}, {"Selected body cluster", enlarge(body)},
        {"Extraction mask", enlarge(ball.mask)}}, 3);
    frame.detail = cv::Mat(panels.rows + 100, panels.cols, CV_8UC3, cv::Scalar(32, 32, 32));
    panels.copyTo(frame.detail(cv::Rect(0, 100, panels.cols, panels.rows)));
    const auto type = classifyExperimentalType(segmentation, p);
    text(frame.detail, "Candidate " + std::to_string(state.selected) + " | actual " + label(data.actual[selected]) +
         " | predicted " + frame.types[selected] + " " + label(frame.numbers[selected]), {10, 22});
    std::ostringstream percentages;
    percentages << std::fixed << std::setprecision(1) << "Light " << type.lightPercent
                << "% | non-light " << type.colouredPercent + type.darkPercent << "%";
    text(frame.detail, percentages.str(), {10, 45});
    const std::string hsv = chosen ? std::to_string(chosen->hsv[0]) + "," + std::to_string(chosen->hsv[1]) +
        "," + std::to_string(chosen->hsv[2]) : "none";
    text(frame.detail, "Body HSV: " + hsv + " | black S " + (state.blackInclusive ? "<= " : "< ") +
         std::to_string(p.blackMaxS) + ", V <= " + std::to_string(p.blackMaxV), {10, 68});
    text(frame.detail, "Masks classify whole clusters; black-like pixels alone do not determine the final number.", {10, 90}, {210, 210, 210}, 0.4);
    return frame;
}

std::filesystem::path saveTunerSnapshot(const std::filesystem::path& directory, TunerData& data,
                                      const TunerState& state, const TunerFrame& frame) {
    std::filesystem::create_directories(directory);
    std::filesystem::path snapshot;
    for (int id = 1;; ++id) {
        std::ostringstream name;
        name << "snapshot_" << std::setw(4) << std::setfill('0') << id;
        snapshot = directory / name.str();
        if (std::filesystem::create_directory(snapshot)) break;
    }
    cv::FileStorage settings((snapshot / "settings.yml").string(), cv::FileStorage::WRITE);
    if (!settings.isOpened()) throw std::runtime_error("Could not save tuner settings");
    const auto& p = state.p;
    settings << "image" << tuningFilename << "colourSpace" << "BGR clustering with HSV rules"
        << "k" << p.k << "cloth" << int(p.cloth) << "lightMinV" << p.lightMinV << "lightMaxS" << p.lightMaxS
        << "darkMaxV" << p.darkMaxV << "whiteMinPercent" << p.whiteMinPercent << "solidMinPercent" << p.solidMinPercent
        << "redUpper" << p.redUpper << "orangeUpper" << p.orangeUpper << "yellowUpper" << p.yellowUpper
        << "greenLower" << p.greenLower << "greenUpper" << p.greenUpper << "blueLower" << p.blueLower
        << "bluePurple" << p.bluePurple << "purpleUpper" << p.purpleUpper << "redWrap" << p.redWrap
        << "maroonMaxV" << p.maroonMaxV << "blackMaxS" << p.blackMaxS << "blackMaxV" << p.blackMaxV
        << "blackSaturationInclusive" << int(state.blackInclusive) << "bodyMinPercent" << p.bodyMinPercent
        << "selectedCandidate" << state.selected << "CCA" << 0 << "padding" << 0 << "kmeansSeed" << 12345;
    settings.release();
    saveImage(snapshot / "overview.png", frame.overview);
    saveImage(snapshot / "selected_ball.png", frame.detail);
    saveFinalResults(data.image, snapshot / "final_results" / "AllBalls", data.circles, data.tableCentres, frame.types, frame.numbers);
    std::ofstream csv(snapshot / "predictions.csv"), features(snapshot / "clusters.csv");
    csv.exceptions(std::ios::failbit | std::ios::badbit);
    features.exceptions(std::ios::failbit | std::ios::badbit);
    csv << "candidate,trueNumber,type,number,typeCorrect,numberCorrect\n";
    features << "candidate,cluster,percent,B,G,R,H,S,V\n" << std::fixed << std::setprecision(6);
    const auto& clusters = clustersFor(data, p.k);
    for (size_t i = 0; i < frame.numbers.size(); ++i) {
        csv << i + 1 << ',' << data.actual[i] << ',' << frame.types[i] << ',' << frame.numbers[i] << ',';
        if (data.actual[i] >= 0)
            csv << (frame.types[i] == actualType(data.actual[i])) << ',' << (frame.numbers[i] == data.actual[i]);
        else csv << ',';
        csv << '\n';
        for (size_t j = 0; j < clusters[i].regions.size(); ++j) {
            const auto& r = clusters[i].regions[j];
            features << i + 1 << ',' << j + 1 << ',' << r.percentage;
            for (const auto value : {r.colour[0], r.colour[1], r.colour[2], r.hsv[0], r.hsv[1], r.hsv[2]})
                features << ',' << int(value);
            features << '\n';
        }
    }
    finishOutputFile(csv);
    finishOutputFile(features);
    return snapshot;
}

const std::array<const char*, 10> sliderNames = {"K", "Light V min", "Light S max", "Cue light %",
    "Solid nonlight %", "Black S max", "Black V max", "Black S inclusive", "Blue-purple H", "Candidate"};

void setControls(const TunerState& state) {
    const auto values = state.values();
    for (size_t i = 0; i < values.size(); ++i) cv::setTrackbarPos(sliderNames[i], controlsWindow, values[i]);
}

TunerState readControls() {
    std::array<int, 10> v;
    for (size_t i = 0; i < v.size(); ++i) v[i] = cv::getTrackbarPos(sliderNames[i], controlsWindow);
    TunerState state;
    state.p.k = v[0]; state.p.lightMinV = v[1]; state.p.lightMaxS = v[2]; state.p.whiteMinPercent = v[3];
    state.p.solidMinPercent = v[4]; state.p.blackMaxS = v[5]; state.p.blackMaxV = v[6];
    state.blackInclusive = v[7]; state.p.bluePurple = v[8]; state.selected = v[9];
    return state;
}

cv::Mat tunerDashboard(const TunerData& data, const TunerFrame& frame) {
    cv::Mat dashboard(360, 1120, CV_8UC3, cv::Scalar(32, 32, 32));
    text(dashboard, "AllBalls only | B baseline | T tuned K=6 | S save | Q/Esc quit | sliders below previews",
         {10, 22}, {245, 245, 245}, 0.48);
    // Enlarge the occupied part of AllBalls rather than its mostly empty cloth.
    cv::Rect occupied;
    for (const auto& circle : data.circles) {
        const int half = cvCeil(circle[2]) + 24;
        occupied |= cv::Rect(cvRound(circle[0]) - half, cvRound(circle[1]) + 80 - half,
                            2 * half + 1, 2 * half + 1);
    }
    occupied &= cv::Rect(0, 80, frame.overview.cols, frame.overview.rows - 80);
    auto fit = [&](const cv::Mat& image, const cv::Rect& target) {
        const double scale = std::min(double(target.width) / image.cols, double(target.height) / image.rows);
        cv::Mat resized;
        cv::resize(image, resized, {std::max(1, cvRound(image.cols * scale)),
                                   std::max(1, cvRound(image.rows * scale))}, 0, 0, cv::INTER_AREA);
        resized.copyTo(dashboard(cv::Rect(target.x + (target.width - resized.cols) / 2,
            target.y + (target.height - resized.rows) / 2, resized.cols, resized.rows)));
    };
    fit(frame.overview(occupied), {8, 45, 650, 280});
    fit(frame.detail, {675, 35, 435, 315});
    text(dashboard, "Types " + std::to_string(frame.correctTypes) + "/" + std::to_string(frame.matched) +
         " | numbers " + std::to_string(frame.correctNumbers) + "/" + std::to_string(frame.matched) +
         " | cyan = selected candidate", {10, 347}, {230, 230, 230}, 0.46);
    return dashboard;
}
} // namespace

void runRecognitionTuner(const std::filesystem::path& directory) {
    auto data = loadTunerData();
    TunerState state;
    cv::namedWindow(controlsWindow, cv::WINDOW_NORMAL);
    auto frame = renderTuner(data, state);
    // Materialise the image before sizing: Cocoa overrides sizes on first imshow.
    cv::imshow(controlsWindow, tunerDashboard(data, frame));
    const std::array<int, 10> maxima = {8, 255, 255, 100, 100, 255, 255, 1, 164, int(data.balls.size())};
    for (size_t i = 0; i < sliderNames.size(); ++i)
        cv::createTrackbar(sliderNames[i], controlsWindow, nullptr, maxima[i]);
    cv::setTrackbarMin("K", controlsWindow, 2);
    cv::setTrackbarMin("Blue-purple H", controlsWindow, 101);
    cv::setTrackbarMin("Candidate", controlsWindow, 1);
    setControls(state);
    // Cocoa adds slider height to this image height (ten sliders, about 250 px).
    cv::resizeWindow(controlsWindow, 1120, 330);
    cv::moveWindow(controlsWindow, 20, 20);
    std::cout << "AllBalls tuner: S saves a new snapshot; B baseline; T tuned; Q/Esc quits.\n";
    while (true) {
        const int key = cv::waitKey(30);
        if (key == 27 || key == 'q' || key == 'Q') break;
        if (cv::getWindowProperty(controlsWindow, cv::WND_PROP_VISIBLE) < 1) break;
        if (key == 'b' || key == 'B') {
            TunerState baseline;
            baseline.p = RecognitionParameters{};
            baseline.blackInclusive = true;
            baseline.selected = state.selected;
            setControls(baseline);
        } else if (key == 't' || key == 'T') {
            TunerState tuned;
            tuned.selected = state.selected;
            setControls(tuned);
        }
        const auto updated = readControls();
        if (updated.values() != state.values()) {
            state = updated;
            frame = renderTuner(data, state);
            cv::imshow(controlsWindow, tunerDashboard(data, frame));
        }
        if (key == 's' || key == 'S') {
            const auto saved = saveTunerSnapshot(directory, data, state, frame);
            std::cout << "Saved " << saved << '\n';
        }
    }
    cv::destroyAllWindows();
}
} // namespace experiments
