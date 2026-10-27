#include "radiusPadding.h"

#include "config.h"
#include "detection/ballDetection.h"
#include "evaluation/evaluation.h"
#include "geometry/perspectiveTransform.h"
#include "recognition/ballTypeClassifier.h"
#include "recognition/kMeansSegmentation.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace experiments {
namespace {

// Learn the local green cloth colour from a ring outside the padded circle.
// Only remove matching components that reach the circular mask's boundary.
void removeCloth(const cv::Mat& image, cv::Point centre, int radius,
                 const RadiusPaddingOptions& options,
                 BallROI& ball) {
    const int outerRadius = radius + options.ringWidth;
    cv::Rect sampleRect(centre.x - outerRadius, centre.y - outerRadius,
                        2 * outerRadius + 1, 2 * outerRadius + 1);
    sampleRect &= cv::Rect(0, 0, image.cols, image.rows);
    cv::Mat hsv;
    cv::cvtColor(image(sampleRect), hsv, cv::COLOR_BGR2HSV);
    int histogram[180] = {};
    std::vector<cv::Vec3b> samples;
    for (int y = 0; y < hsv.rows; ++y) {
        for (int x = 0; x < hsv.cols; ++x) {
            const double dx = x + sampleRect.x - centre.x;
            const double dy = y + sampleRect.y - centre.y;
            const double distanceSquared = dx * dx + dy * dy;
            if (distanceSquared <= double(radius) * radius ||
                distanceSquared > double(outerRadius) * outerRadius)
                continue;
            const auto pixel = hsv.at<cv::Vec3b>(y, x);
            // Restrict learning to green cloth, not neighbouring ball colours.
            if (pixel[0] >= 35 && pixel[0] <= 95 && pixel[1] >= 50 && pixel[2] >= 30) {
                samples.push_back(pixel);
                ++histogram[pixel[0]];
            }
        }
    }
    if (samples.size() < 20)
        return; // Too little green evidence: leave the padded mask intact.

    const int hue = int(std::max_element(histogram, histogram + 180) - histogram);
    double saturation = 0, value = 0;
    int count = 0;
    for (const auto& pixel : samples) {
        if (std::abs(int(pixel[0]) - hue) <= options.hueTolerance) {
            saturation += pixel[1];
            value += pixel[2];
            ++count;
        }
    }
    saturation /= count;
    value /= count;

    cv::Mat localHSV;
    cv::cvtColor(ball.image, localHSV, cv::COLOR_BGR2HSV);
    cv::Mat candidates = cv::Mat::zeros(ball.mask.size(), CV_8UC1);
    for (int y = 0; y < localHSV.rows; ++y) {
        for (int x = 0; x < localHSV.cols; ++x) {
            const auto pixel = localHSV.at<cv::Vec3b>(y, x);
            if (ball.mask.at<uchar>(y, x) && pixel[1] >= 50 && pixel[2] >= 30 &&
                std::abs(int(pixel[0]) - hue) <= options.hueTolerance &&
                std::abs(pixel[1] - saturation) <= options.saturationTolerance &&
                std::abs(pixel[2] - value) <= options.valueTolerance)
                candidates.at<uchar>(y, x) = 255;
        }
    }

    // Include clipped image edges in the boundary, too.
    cv::Mat interior, boundary;
    cv::erode(ball.mask, interior,
              cv::getStructuringElement(cv::MORPH_ELLIPSE, {3, 3}),
              {-1, -1}, 1, cv::BORDER_CONSTANT, cv::Scalar(0));
    cv::subtract(ball.mask, interior, boundary);
    cv::Mat labels;
    const int components = cv::connectedComponents(candidates, labels, 8, CV_32S);
    std::vector<bool> remove(components, false);
    for (int y = 0; y < labels.rows; ++y)
        for (int x = 0; x < labels.cols; ++x)
            if (boundary.at<uchar>(y, x) && labels.at<int>(y, x) > 0)
                remove[labels.at<int>(y, x)] = true;

    for (int y = 0; y < labels.rows; ++y)
        for (int x = 0; x < labels.cols; ++x)
            if (remove[labels.at<int>(y, x)])
                ball.mask.at<uchar>(y, x) = 0;
}

void save(const std::filesystem::path& path, const cv::Mat& image) {
    if (!cv::imwrite(path.string(), image))
        throw std::runtime_error("Could not save " + path.string());
}

} // namespace

BallROI extractPaddedBall(const cv::Mat& image, const cv::Vec3f& detected,
                          const RadiusPaddingOptions& options) {
    if (options.padding < 0 || options.ringWidth < 1 || options.hueTolerance < 0 ||
        options.saturationTolerance < 0 || options.valueTolerance < 0)
        throw std::invalid_argument("Padding/tolerances must be nonnegative and ring width positive");
    if (image.empty())
        return {};

    // Reuse the original extraction so padding=0 is exactly the existing baseline.
    const int radius = std::max(1, cvRound(detected[2])) + options.padding;
    cv::Vec3f padded = detected;
    padded[2] = static_cast<float>(radius);
    BallROI ball = extractBall(image, padded);
    if (ball.image.empty() || !options.filterCloth)
        return ball;

    const cv::Point centre(cvRound(detected[0]), cvRound(detected[1]));
    removeCloth(image, centre, radius, options, ball);
    return ball;
}

void runRadiusPaddingExperiment(const std::filesystem::path& imagePath,
                                const std::filesystem::path& outputDirectory) {
    const cv::Mat image = cv::imread(imagePath.string());
    if (image.empty())
        throw std::runtime_error("Could not read " + imagePath.string());
    const auto detection = runDetection(image, HoughParams{});
    const auto directory = outputDirectory / imagePath.stem();
    std::filesystem::create_directories(directory);
    std::ofstream csv(directory / "results.csv");
    if (!csv)
        throw std::runtime_error("Could not create results.csv");
    csv << "ball,x,y,houghRadius,padding,clothFilter,paddedPixels,keptPixels,removedPixels,"
           "type,number,lightPercent,colouredPercent,darkPercent,"
           "trueNumber,trueType,typeCorrect,numberCorrect\n";

    const auto truth = loadLabelledGroundTruth(config::groundTruthPath,
                                               imagePath.filename().string());
    std::vector<cv::Point2f> truthCentres;
    for (const auto& ball : truth)
        truthCentres.push_back(ball.centre);
    const auto matching = evaluateDetections(transformCentres(detection.filteredCircles),
                                             truthCentres, config::matchTolerance);
    constexpr int tileWidth = 220, tileHeight = 255, previewSize = 150;
    constexpr int headerHeight = 60;
    for (size_t i = 0; i < detection.filteredCircles.size(); ++i) {
        const auto& circle = detection.filteredCircles[i];
        const int truthIndex = matching.predictionToGroundTruth[i];
        const bool matched = truthIndex >= 0;
        const int trueNumber = matched ? truth[truthIndex].number : -1;
        const std::string trueType = !matched ? "Unknown"
            : trueNumber == 0 ? "White" : trueNumber <= 8 ? "Solid" : "Striped";
        cv::Mat grid(headerHeight + 2 * tileHeight, 5 * tileWidth,
                     CV_8UC3, cv::Scalar(35, 35, 35));
        const std::string title = imagePath.filename().string() + " / candidate "
            + std::to_string(i + 1) + (matched
                ? " / expected: " + trueType + " #" + std::to_string(trueNumber)
                : " / no ground-truth match (N/A)");
        cv::putText(grid, title, {10, 24}, cv::FONT_HERSHEY_SIMPLEX,
                    0.55, {255, 255, 255}, 1);
        cv::putText(grid, "Top: padding only | Bottom: cloth filtering | Green: correct, red: wrong, grey: N/A",
                    {10, 48}, cv::FONT_HERSHEY_SIMPLEX, 0.5, {220, 220, 220}, 1);
        for (int padding = 0; padding <= 4; ++padding) {
            RadiusPaddingOptions options;
            options.padding = padding;
            const auto unfiltered = extractPaddedBall(image, circle, options);
            const int paddedPixels = cv::countNonZero(unfiltered.mask);
            for (int mode = 0; mode < 2; ++mode) {
                options.filterCloth = mode == 1;
                const auto ball = mode ? extractPaddedBall(image, circle, options) : unfiltered;
                const int keptPixels = cv::countNonZero(ball.mask);
                const auto segmentation = segmentBallKMeans(ball, config::recognitionKMeansClusters);
                const auto prediction = classifyBallType(segmentation);
                const int number = classifyBallNumber(segmentation, prediction);
                csv << i + 1 << ',' << circle[0] << ',' << circle[1] << ',' << circle[2]
                    << ',' << padding << ',' << mode << ',' << paddedPixels << ',' << keptPixels
                    << ',' << paddedPixels - keptPixels << ',' << prediction.type << ',' << number
                    << ',' << prediction.lightPercent << ',' << prediction.colouredPercent
                    << ',' << prediction.darkPercent << ',';
                if (matched)
                    csv << trueNumber << ',' << trueType << ','
                        << (prediction.type == trueType ? 1 : 0) << ','
                        << (number == trueNumber ? 1 : 0);
                else
                    csv << ",,,";
                csv << '\n';

                cv::Mat isolated = cv::Mat::zeros(ball.image.size(), ball.image.type());
                ball.image.copyTo(isolated, ball.mask);
                const int left = padding * tileWidth;
                const int top = headerHeight + mode * tileHeight;
                cv::putText(grid, "+" + std::to_string(padding) + " px", {left + 10, top + 22},
                            cv::FONT_HERSHEY_SIMPLEX, 0.55, {255, 255, 255}, 1);
                cv::Mat preview;
                // Use one scale across all variants so padding remains visible.
                const double scale = double(previewSize) /
                    (2 * (std::max(1, cvRound(circle[2])) + 4) + 1);
                cv::resize(isolated, preview, {}, scale, scale, cv::INTER_NEAREST);
                preview.copyTo(grid(cv::Rect(left + (tileWidth - preview.cols) / 2,
                                            top + 32 + (previewSize - preview.rows) / 2,
                                            preview.cols, preview.rows)));
                const auto indicator = [&](const std::string& label, bool correct, int y) {
                    const cv::Scalar colour = !matched ? cv::Scalar(180, 180, 180)
                        : correct ? cv::Scalar(90, 220, 90) : cv::Scalar(80, 80, 255);
                    cv::putText(grid, label + (!matched ? " N/A" : correct ? " OK" : " WRONG"),
                                {left + 10, top + y}, cv::FONT_HERSHEY_SIMPLEX, 0.5, colour, 1);
                };
                indicator("Type: " + prediction.type, prediction.type == trueType, 210);
                indicator("Number: " + (number < 0 ? std::string("Unknown") : std::to_string(number)),
                          number == trueNumber, 235);
            }
        }
        save(directory / ("ball_" + std::to_string(i + 1) + "_comparison.png"), grid);
    }

    csv.close();
    if (!csv)
        throw std::runtime_error("Could not finish results.csv");
    std::cout << imagePath << ": " << detection.filteredCircles.size()
              << " balls, 10 variants each -> " << directory << '\n';
}

} // namespace experiments
