#include <opencv2/opencv.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace pool_submission {

// ============== 
// CONFIGURATION
// ============== 
struct HoughParams {
    double dp = 1.5;
    double minDist = 18.0;
    double param1 = 75.0;
    double param2 = 35.0;
    int minRadius = 15;
    int maxRadius = 30;
};


inline const std::string configMediaDirectory = "Media/";
inline const std::string configGroundTruthPath = "data/ground_truth/BallLocationsNew.csv";
inline const std::string configOutputDirectory = "output/";
inline const std::string configTuningImage = "AllBalls.jpeg";

inline const std::string configApplicationOutputDirectory = configOutputDirectory + "final-results/";

inline constexpr int configTableWidth = 1434;
inline constexpr int configTableHeight = 683;
inline constexpr double configMatchTolerance = 15.0;
inline constexpr float configPocketRadius = 25.0f;
inline constexpr int configRecognitionKMeansClusters = 2;

inline const std::vector<cv::Point> configTableCorners = {{337, 35}, {1570, 21}, {1888, 451}, {38, 472}};

inline const std::vector<std::string> configValidationImages = {"Table1.jpeg", "Table2.jpeg", "Table3.jpeg", "Table4.jpeg", "Table5.jpeg", "Table6.jpeg"};

inline std::vector<std::string> configAllImages() {
    auto images = configValidationImages;
    images.insert(images.begin(), configTuningImage);
    return images;
}

struct DetectionResult {
    cv::Mat maskedImage;
    std::vector<cv::Vec3f> rawCircles;
    std::vector<cv::Vec3f> filteredCircles;
};

struct BallROI {
    cv::Mat image;
    cv::Mat mask;
};

struct KMeansRegion {
    cv::Mat mask;

    double percentage = 0;

    // Average colour of cluster (OpenCV BGR)
    cv::Vec3b colour = cv::Vec3b(0, 0, 0);
    cv::Vec3b hsv = cv::Vec3b(0, 0, 0);
};

struct KMeansResult {
    int k = 0;
    cv::Mat segmentedImage;
    std::vector<KMeansRegion> regions;

    // Hue comparaisons between the two largest clusters
    double hueDifference = -1;
    double saturationDifference = 0;
    double brightnessDifference = 0;
    double bgrDistance = 0;
};

struct BallTypeResult {
    std::string type = "Unknown";

    double lightPercent = 0;
    double colouredPercent = 0;
    double darkPercent = 0;
};

struct EvaluationResult {
    int TP = 0;
    int FP = 0;
    int FN = 0;

    double precision = 0;
    double recall = 0;
    double f1 = 0;
    double meanError = 0;

    // Index of matched ground-truth ball for each prediction.
    // -1 means the prediction is a false positive.
    std::vector<int> predictionToGroundTruth;

    // Index of matched prediction for each ground-truth ball.
    // -1 means the ground-truth ball was missed.
    std::vector<int> groundTruthToPrediction;
};

struct LabelledBall {
    int number; // 1-15, 0 = cue ball
    cv::Point2f centre;
};

// ===============================
// MAIN PIPELINE HELPER FUNCTIONS
// ===============================
std::vector<cv::Point> getTableCorners() {
    return configTableCorners;
}

/*
Calculates the perspective transformation that maps the four image corners onto the rectangular table.
*/
cv::Mat getTableTransformation() {
    // Original image corners
    std::vector<cv::Point2f> source;

    for (const cv::Point& corner : getTableCorners()) {
        source.emplace_back(corner.x, corner.y);
    }

    // Transformed table coordinates
    std::vector<cv::Point2f> destination = {cv::Point2f(0, 0), cv::Point2f(configTableWidth - 1, 0), cv::Point2f(configTableWidth - 1, configTableHeight - 1), cv::Point2f(0, configTableHeight - 1)};

    return cv::getPerspectiveTransform(source, destination);
}

/*
Converts detected circle centres from original image coordinates into table coordinates.
*/
std::vector<cv::Point2f> transformCentres(const std::vector<cv::Vec3f>& circles) {
    cv::Mat matrix = getTableTransformation();

    std::vector<cv::Point2f> centres;

    for (const cv::Vec3f& c : circles) {
        centres.push_back(cv::Point2f(c[0], c[1]));
    }

    std::vector<cv::Point2f> transformed;

    if (!centres.empty()) {
        cv::perspectiveTransform(centres, transformed, matrix);
    }

    return transformed;
}

/*
Removes detected circles whose transformed centres are closer than pocketRadius to any of the six pocket locations.
*/
std::vector<cv::Vec3f> filterPocketDetections(const std::vector<cv::Vec3f>& circles) {

    // Table dimensions after perspective transformation
    const float width = configTableWidth;
    const float height = configTableHeight;

    // Approximate pocket locations in transformed coordinates
    std::vector<cv::Point2f> pockets = {{0, 0}, {width / 2, 0}, {width - 1, 0}, {0, height - 1}, {width / 2, height - 1}, {width - 1, height - 1}};

    const float pocketRadius = configPocketRadius;

    std::vector<cv::Point2f> transformedCentres = transformCentres(circles);

    std::vector<cv::Vec3f> filtered;

    for (size_t i = 0; i < circles.size(); ++i) {
        bool nearPocket = false;

        for (const cv::Point2f& pocket : pockets) {
            if (cv::norm(transformedCentres[i] - pocket) < pocketRadius) {
                nearPocket = true;
                break;
            }
        }

        if (!nearPocket) {
            filtered.push_back(circles[i]);
        }
    }

    return filtered;
}

/*
Finds circle candidates using Hough detection after converting the image to greyscale and blurring it.
*/
std::vector<cv::Vec3f> detectBalls(const cv::Mat& image, const HoughParams& params) {
    cv::Mat grey_image;
    cv::cvtColor(image, grey_image, cv::COLOR_BGR2GRAY);

    cv::Mat blurred_image;
    cv::GaussianBlur(grey_image, blurred_image, cv::Size(5, 5), 1.5);

    std::vector<cv::Vec3f> circles;

    cv::HoughCircles(blurred_image, circles, cv::HOUGH_GRADIENT, params.dp, params.minDist, params.param1, params.param2, params.minRadius, params.maxRadius);

    return circles;
}

/*
Keeps the pixels inside the table polygon and makes the rest of the image black.
*/
cv::Mat maskTable(const cv::Mat& image) {
    if (image.empty())
        return {};

    cv::Mat mask = cv::Mat::zeros(image.size(), CV_8UC1);
    cv::fillConvexPoly(mask, getTableCorners(), cv::Scalar(255));

    cv::Mat maskedImage;
    cv::bitwise_and(image, image, maskedImage, mask);

    return maskedImage;
}

/*
Runs table masking, circle detection and pocket checking while retaining the intermediate results.
*/
DetectionResult runDetection(const cv::Mat& image, const HoughParams& params) {
    DetectionResult result;

    if (image.empty())
        return result;

    result.maskedImage = maskTable(image);

    result.rawCircles = detectBalls(result.maskedImage, params);
    result.filteredCircles = filterPocketDetections(result.rawCircles);

    return result;
}

/*
Crops the image around a detected circle and creates a mask identifying the pixels inside that circle.
*/
BallROI extractBall(const cv::Mat& image, const cv::Vec3f& detectedCircle) {
    BallROI result;

    if (image.empty())
        return result;

    // Extract circle information from Hough detection
    int x = cvRound(detectedCircle[0]);
    int y = cvRound(detectedCircle[1]);
    int radius = std::max(1, cvRound(detectedCircle[2]));

    // Define square region around the ball
    cv::Rect roi(x - radius, y - radius, 2 * radius + 1, 2 * radius + 1);

    // Ensure ROI stays inside the image
    roi &= cv::Rect(0, 0, image.cols, image.rows);

    if (roi.empty())
        return result;

    // Extract the original colour pixels
    result.image = image(roi).clone();
    result.mask = cv::Mat::zeros(result.image.size(), CV_8UC1);

    cv::Point localCentre(x - roi.x, y - roi.y);

    cv::circle(result.mask, localCentre, radius, cv::Scalar(255), cv::FILLED);

    return result;
}

/*
Groups the masked ball pixels into K colour clusters and records their colours, masks and proportions.
*/
KMeansResult segmentBallKMeans(const BallROI& ball, int k) {
    KMeansResult result;
    result.k = k;

    if (ball.image.empty() || ball.mask.empty() || k < 2)
        return result;

    // Collect ball pixels
    std::vector<cv::Point> pixels;

    for (int y = 0; y < ball.image.rows; ++y) {
        for (int x = 0; x < ball.image.cols; ++x) {

            if (ball.mask.at<uchar>(y, x) != 0)
                pixels.emplace_back(x, y);
        }
    }

    int total = static_cast<int>(pixels.size());

    if (total < k)
        return result;

    // Prepare BGR samples for K-means
    cv::Mat samples(total, 3, CV_32F);

    for (int i = 0; i < total; ++i) {

        cv::Vec3b colour = ball.image.at<cv::Vec3b>(pixels[i]);

        samples.at<float>(i, 0) = colour[0];
        samples.at<float>(i, 1) = colour[1];
        samples.at<float>(i, 2) = colour[2];
    }

    // K-means clustering
    cv::Mat labels;
    cv::Mat centres;

    cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 50, 0.2);

    // Fixed seed for reproducibility
    uint64 previousState = cv::theRNG().state;
    cv::theRNG().state = 12345;
    cv::kmeans(samples, k, labels, criteria, 3, cv::KMEANS_PP_CENTERS, centres);
    cv::theRNG().state = previousState;

    // Create cluster masks
    std::vector<cv::Mat> masks(k);

    for (int j = 0; j < k; ++j) {
        masks[j] = cv::Mat::zeros(ball.image.size(), CV_8UC1);
    }

    result.segmentedImage = cv::Mat::zeros(ball.image.size(), CV_8UC3);

    // Reconstruct segmented image
    for (int i = 0; i < total; ++i) {

        int cluster = labels.at<int>(i, 0);
        cv::Point p = pixels[i];
        cv::Vec3b colour(cv::saturate_cast<uchar>(centres.at<float>(cluster, 0)), cv::saturate_cast<uchar>(centres.at<float>(cluster, 1)), cv::saturate_cast<uchar>(centres.at<float>(cluster, 2)));
        result.segmentedImage.at<cv::Vec3b>(p) = colour;
        masks[cluster].at<uchar>(p) = 255;
    }

    // Extract cluster information
    for (int j = 0; j < k; ++j) {

        int area = cv::countNonZero(masks[j]);

        if (area == 0)
            continue;

        KMeansRegion region;

        region.mask = masks[j];
        region.percentage = 100.0 * area / total;
        region.colour = cv::Vec3b(cv::saturate_cast<uchar>(centres.at<float>(j, 0)), cv::saturate_cast<uchar>(centres.at<float>(j, 1)), cv::saturate_cast<uchar>(centres.at<float>(j, 2)));

        cv::Mat bgr(1, 1, CV_8UC3);
        bgr.at<cv::Vec3b>(0, 0) = region.colour;

        cv::Mat hsv;
        cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);

        region.hsv = hsv.at<cv::Vec3b>(0, 0);
        result.regions.push_back(region);
    }

    // Sort largest regions first
    std::sort(result.regions.begin(), result.regions.end(), [](const KMeansRegion& a, const KMeansRegion& b) {
        return a.percentage > b.percentage;
    });

    // compare cluster colours
    if (result.k == 2 && result.regions.size() == 2) {
        cv::Vec3b colour1 = result.regions[0].colour;
        cv::Vec3b colour2 = result.regions[1].colour;

        cv::Mat bgr(1, 2, CV_8UC3);
        bgr.at<cv::Vec3b>(0, 0) = colour1;
        bgr.at<cv::Vec3b>(0, 1) = colour2;

        cv::Mat hsv;
        cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);

        cv::Vec3b hsv1 = hsv.at<cv::Vec3b>(0, 0);
        cv::Vec3b hsv2 = hsv.at<cv::Vec3b>(0, 1);

        int H1 = hsv1[0];
        int S1 = hsv1[1];
        int V1 = hsv1[2];

        int H2 = hsv2[0];
        int S2 = hsv2[1];
        int V2 = hsv2[2];

        result.saturationDifference = std::abs(S1 - S2);
        result.brightnessDifference = std::abs(V1 - V2);

        if (S1 >= 60 && S2 >= 60 && V1 >= 70 && V2 >= 70) {
            int difference = std::abs(H1 - H2);
            result.hueDifference = std::min(difference, 180 - difference);
        }

        cv::Vec3d difference(double(colour1[0] - colour2[0]), double(colour1[1] - colour2[1]), double(colour1[2] - colour2[2]));
        result.bgrDistance = cv::norm(difference);
    }

    return result;
}

// ======================================
// K-MEANS WHITE-COVERAGE CLASSIFICATION HERE NEXT
// ======================================

/*
Estimates whether a ball is white, solid or striped from the proportions of light, dark and coloured clusters.
*/
BallTypeResult classifyBallType(const KMeansResult& segmentation) {
    BallTypeResult result;

    if (segmentation.regions.empty())
        return result;

    // Fixed thresholds used by the report configuration
    const int LIGHT_MIN_V = 170;
    const int LIGHT_MAX_S = 165;
    const int DARK_MAX_V = 90;

    // Initial ball-type thresholds
    const double WHITE_BALL_MIN = 85.0;
    const double SOLID_MIN = 70.0;

    for (const KMeansRegion& region : segmentation.regions) {

        // Convert representative BGR colour to HSV
        cv::Mat bgr(1, 1, CV_8UC3);
        bgr.at<cv::Vec3b>(0, 0) = region.colour;

        cv::Mat hsv;
        cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);

        cv::Vec3b pixel = hsv.at<cv::Vec3b>(0, 0);

        int S = pixel[1];
        int V = pixel[2];

        if (V >= LIGHT_MIN_V && S <= LIGHT_MAX_S) {
            result.lightPercent += region.percentage;
        } else if (V <= DARK_MAX_V) {
            result.darkPercent += region.percentage;
        } else {
            result.colouredPercent += region.percentage;
        }
    }

    // Classify using cluster proportions
    if (result.lightPercent >= WHITE_BALL_MIN) {
        result.type = "White";
    } else if (result.colouredPercent + result.darkPercent >= SOLID_MIN) {
        result.type = "Solid";
    } else if (result.lightPercent > 0 && result.colouredPercent + result.darkPercent > 0) {
        result.type = "Striped";
    }

    return result;
}

// ==================================================
// COLOUR HISTOGRAMS AND UNIQUE IDENTITY ASSIGNMENT
// ==================================================

enum class BallHistogramSpace { HS, HSV, HLS };

struct BallHistogramOptions {
    int hueBins = 18, saturationBins = 8, toneBins = 8;
    int lightMinV = 170, lightMaxS = 165;
    int erosionKernel = 3, minimumBodyPixels = 12, neutralChroma = 15;
};

struct LabelledColourSample {
    int number; // Training labels only: 1–15; cue (0) is excluded.
    BallROI ball;
};

struct BallColourModel {
    BallHistogramSpace space;
    BallHistogramOptions options;
    std::array<cv::Mat, 8> histograms; // Colours 1–8; 9–15 share colours 1–7.
    std::array<int, 8> examples{};
};

struct BallColourScores {
    cv::Mat bodyMask;
    std::array<double, 8> similarity{};
    int pixels = 0, bestColour = -1;
    double margin = 0;
};

void validate(const BallHistogramOptions& p) {
    if (p.hueBins < 2 || p.saturationBins < 2 || p.toneBins < 2 || p.erosionKernel < 1 || p.erosionKernel % 2 == 0 || p.minimumBodyPixels < 1 || p.lightMinV < 0 || p.lightMinV > 255 || p.lightMaxS < 0 || p.lightMaxS > 255 ||
        p.neutralChroma < 0 || p.neutralChroma > 255)
        throw std::invalid_argument("Invalid ball histogram settings");
}

/*
Converts a ball crop into the selected histogram colour space and sets the hue of nearly neutral pixels to zero.
*/
cv::Mat featureImage(const BallROI& ball, BallHistogramSpace space, const BallHistogramOptions& p) {
    cv::Mat converted, features;
    if (space == BallHistogramSpace::HLS) {
        cv::cvtColor(ball.image, converted, cv::COLOR_BGR2HLS);
        features.create(converted.size(), converted.type());
        const int mapping[] = {0, 0, 2, 1, 1, 2}; // Store H,S,L, so tone is always channel 2.
        cv::mixChannels(&converted, 1, &features, 1, mapping, 3);
    } else
        cv::cvtColor(ball.image, features, cv::COLOR_BGR2HSV);
    for (int y = 0; y < features.rows; ++y)
        for (int x = 0; x < features.cols; ++x) {
            const auto bgr = ball.image.at<cv::Vec3b>(y, x);
            const int chroma = int(std::max({bgr[0], bgr[1], bgr[2]})) - std::min({bgr[0], bgr[1], bgr[2]});
            if (chroma < p.neutralChroma)
                features.at<cv::Vec3b>(y, x)[0] = 0;
        }
    return features;
}

/*
Smooths neighbouring histogram bins, wrapping hue at its endpoints, and normalises the total bin weight to one.
*/
cv::Mat smoothHistogram(const cv::Mat& histogram, const BallHistogramOptions& p, bool tone) {
    cv::Mat current = histogram.clone();
    const int hCount = p.hueBins, sCount = p.saturationBins, tCount = tone ? p.toneBins : 1;
    auto at = [=](int h, int s, int t) {
        return (h * sCount + s) * tCount + t;
    };

    // Separable [1,2,1]/4 smoothing. Hue wraps; saturation and tone clamp at endpoints.
    for (int axis = 0; axis < (tone ? 3 : 2); ++axis) {
        cv::Mat next = cv::Mat::zeros(current.dims, current.size, CV_32F);
        const float* source = current.ptr<float>();
        float* target = next.ptr<float>();
        for (int h = 0; h < hCount; ++h) {
            for (int s = 0; s < sCount; ++s) {
                for (int t = 0; t < tCount; ++t) {
                    for (int offset = -1; offset <= 1; ++offset) {
                        const int hh = axis == 0 ? (h + offset + hCount) % hCount : h;
                        const int ss = axis == 1 ? std::clamp(s + offset, 0, sCount - 1) : s;
                        const int tt = axis == 2 ? std::clamp(t + offset, 0, tCount - 1) : t;
                        target[at(h, s, t)] += source[at(hh, ss, tt)] * (offset == 0 ? 0.5f : 0.25f);
                    }
    
                }   
            }
        }
        current = next;
    }
    cv::normalize(current, current, 1, 0, cv::NORM_L1);
    return current;
}

/*
Builds a smoothed, normalised colour histogram from the pixels selected by the supplied mask.
*/
cv::Mat makeHistogram(const cv::Mat& features, const cv::Mat& mask, BallHistogramSpace space, const BallHistogramOptions& p) {
    const int channels[] = {0, 1, 2}, bins[] = {p.hueBins, p.saturationBins, p.toneBins};
    const float hue[] = {0, 180}, other[] = {0, 256};
    const float* ranges[] = {hue, other, other};
    cv::Mat histogram;
    cv::calcHist(&features, 1, channels, mask, histogram, space == BallHistogramSpace::HS ? 2 : 3, bins, ranges);
    return smoothHistogram(histogram, p, space != BallHistogramSpace::HS);
}

const char* bodyColourName(int colour) {
    static const char* names[] = {"unknown", "yellow", "blue", "red", "purple", "orange", "green", "maroon", "black"};
    return names[colour >= 1 && colour <= 8 ? colour : 0];
}

/*
Selects ball body pixels by shrinking the circular mask slightly and removing bright, low-saturation pixels.
*/
cv::Mat histogramBodyMask(const BallROI& ball, const BallHistogramOptions& p = {}) {
    validate(p);
    if (ball.image.empty() || ball.image.type() != CV_8UC3 || ball.mask.type() != CV_8UC1 || ball.image.size() != ball.mask.size())
        throw std::invalid_argument("Invalid ball ROI");
    cv::Mat mask, hsv, light;
    cv::erode(ball.mask, mask, cv::getStructuringElement(cv::MORPH_ELLIPSE, {p.erosionKernel, p.erosionKernel}), {-1, -1}, 1, cv::BORDER_CONSTANT, 0);
    cv::cvtColor(ball.image, hsv, cv::COLOR_BGR2HSV);
    cv::inRange(hsv, cv::Scalar(0, 0, p.lightMinV), cv::Scalar(179, p.lightMaxS, 255), light);
    mask.setTo(0, light);
    return mask;
}

/*
Builds eight reference colour histograms from labelled training balls, pooling solid and striped balls of the same colour.
*/
BallColourModel trainBallColourModel(const std::vector<LabelledColourSample>& samples, BallHistogramSpace space, const BallHistogramOptions& options = {}) {
    validate(options);
    BallColourModel model{space, options, {}, {}};
    for (const auto& sample : samples) {
        if (sample.number < 0 || sample.number > 15)
            throw std::invalid_argument("Invalid training identity");

        if (sample.number == 0)
            continue;

        const auto mask = histogramBodyMask(sample.ball, options);

        if (cv::countNonZero(mask) < options.minimumBodyPixels)
            throw std::runtime_error("Insufficient body pixels for training ball " + std::to_string(sample.number));

        const int colour = (sample.number > 8 ? sample.number - 8 : sample.number) - 1;

        const auto histogram = makeHistogram(featureImage(sample.ball, space, options), mask, space, options);

        if (model.histograms[colour].empty())
            model.histograms[colour] = histogram.clone();
        else
            model.histograms[colour] += histogram;

        ++model.examples[colour];
    }
    for (size_t i = 0; i < model.histograms.size(); ++i) {
        if (!model.examples[i])
            throw std::runtime_error("No training example for " + std::string(bodyColourName(i + 1)));
        cv::normalize(model.histograms[i], model.histograms[i], 1, 0, cv::NORM_L1);
    }
    return model;
}

/*
Compares the ball body's HSV histogram with all eight reference colour histograms.
*/
BallColourScores scoreBallColours(const BallROI& ball, const BallColourModel& model) {
    BallColourScores result;
    result.bodyMask = histogramBodyMask(ball, model.options);
    result.pixels = cv::countNonZero(result.bodyMask);

    if (result.pixels < model.options.minimumBodyPixels)
        return result;

    const auto features = featureImage(ball, model.space, model.options);
    const auto query = makeHistogram(features, result.bodyMask, model.space, model.options);

    for (size_t i = 0; i < model.histograms.size(); ++i) {
        const auto& histogram = model.histograms[i];
        if (histogram.empty() || histogram.type() != CV_32F || histogram.total() != query.total())
            throw std::invalid_argument("Invalid trained colour histogram");

        // Flat views make comparison independent of OpenCV's handling of 3D Mats.
        const cv::Mat a(1, int(query.total()), CV_32F, const_cast<float*>(query.ptr<float>()));
        const cv::Mat b(1, int(histogram.total()), CV_32F, const_cast<float*>(histogram.ptr<float>()));

        result.similarity[i] = std::clamp(1 - cv::compareHist(a, b, cv::HISTCMP_BHATTACHARYYA), 0.0, 1.0);
    }
    const auto best = std::max_element(result.similarity.begin(), result.similarity.end());
    result.bestColour = int(best - result.similarity.begin()) + 1;

    double second = 0;
    for (size_t i = 0; i < result.similarity.size(); ++i)
        if (int(i + 1) != result.bestColour)
            second = std::max(second, result.similarity[i]);

    result.margin = *best - second;
    return result;
}

constexpr double typeWidth = 0.10;

/*
Uses the Hungarian method to choose one distinct column per row while maximising the total assignment weight.
*/
std::vector<int> maximumAssignment(const std::vector<std::vector<double>>& weights) {
    const int n = int(weights.size());

    if (!n)
        return {};

    const int m = int(weights[0].size());

    if (m < n)
        throw std::invalid_argument("Assignment requires at least as many columns as rows");
    for (const auto& row : weights) {
        if (int(row.size()) != m)
            throw std::invalid_argument("Ragged assignment matrix");
        for (double w : row)
            if (!std::isfinite(w))
                throw std::invalid_argument("Non-finite assignment weight");
    }

    std::vector<double> u(n + 1), v(m + 1);
    std::vector<int> p(m + 1), way(m + 1);

    for (int i = 1; i <= n; ++i) {
        p[0] = i;
        int j0 = 0;

        std::vector<double> minimum(m + 1, std::numeric_limits<double>::infinity());
        std::vector<bool> used(m + 1, false);

        do {
            used[j0] = true;
            const int i0 = p[j0];
            int j1 = 0;
            double delta = std::numeric_limits<double>::infinity();

            for (int j = 1; j <= m; ++j) {
                if (!used[j]) {
                    const double reduced = -weights[i0 - 1][j - 1] - u[i0] - v[j];
                    if (reduced < minimum[j]) {
                        minimum[j] = reduced;
                        way[j] = j0;
                    }
                    if (minimum[j] < delta) {
                        delta = minimum[j];
                        j1 = j;
                    }
                }
            }

            for (int j = 0; j <= m; ++j) {
                if (used[j]) {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else
                    minimum[j] -= delta;
            }

            j0 = j1;

        } while (p[j0] != 0);

        do {
            const int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0);
    }

    std::vector<int> assigned(n, -1);
    for (int j = 1; j <= m; ++j)
        if (p[j])
            assigned[p[j] - 1] = j - 1;
    return assigned;
}

struct Candidate {
    cv::Vec3f circle;
    cv::Point2f tableCentre;
    BallROI roi;
    BallTypeResult baselineType;
    BallColourScores colours;
    std::array<double, 3> typeSupport{};
    std::array<double, 16> identityScore{};
    std::array<bool, 16> eligible{};
    int localBest = -1;
    double bestScore = 0;
};

/*
Checks whether a candidate has a usable body colour score or was classified as white.
*/
bool hasUsableEvidence(const Candidate& c) {
    return c.colours.bestColour >= 1 || c.baselineType.type == "White";
}

/*
Combines white coverage and histogram colour similarities into scores and eligibility flags for all sixteen identities.
*/
void identityEvidence(Candidate& c) {
    const double light = std::clamp(c.baselineType.lightPercent / 100, 0.0, 1.0);
    const auto compatibility = [](double distance) {
        return std::exp(-0.5 * std::pow(distance / typeWidth, 2));
    };

    // Distances to the original K2 light intervals: cue >=.85, solid <=.30, stripe [.30,.85].
    c.typeSupport = {compatibility(std::max(0.0, 0.85 - light)), compatibility(std::max(0.0, light - 0.30)), compatibility(std::max({0.0, 0.30 - light, light - 0.85}))};
    c.identityScore[0] = light * c.typeSupport[0];
    for (int n = 1; n <= 15; ++n) {
        const int colour = n <= 8 ? n : n - 8;
        c.identityScore[n] = c.colours.bestColour < 0 ? 0 : c.colours.similarity[colour - 1] * c.typeSupport[n <= 8 ? 1 : 2];
    }

    const auto best = std::max_element(c.identityScore.begin(), c.identityScore.end());
    c.localBest = int(best - c.identityScore.begin());
    c.bestScore = *best;
    for (int n = 0; n <= 15; ++n)
        c.eligible[n] = hasUsableEvidence(c) && c.identityScore[n] > 1e-9;
}

/*
Assigns identities to candidates with the highest total score while allowing each identity once and any number of Unknowns.
*/
std::vector<int> assignUnique(const std::vector<Candidate>& candidates) {
    const double abstention = 0;

    std::vector<std::vector<double>> weights(candidates.size(), std::vector<double>(16 + candidates.size(), abstention + 1e-9));
    for (size_t i = 0; i < candidates.size(); ++i)
        for (int n = 0; n <= 15; ++n)
            weights[i][n] = candidates[i].eligible[n] ? candidates[i].identityScore[n] : -1e6;

    auto result = maximumAssignment(weights);
    for (auto& n : result)
        if (n >= 16)
            n = -1; // Separate dummy columns allow any number of Unknowns.
    return result;
}

/*
Recomputes recognition evidence using an expanded crop while preserving the original detected circle and centre.
*/
Candidate rawWhitePadding(const cv::Mat& image, const Candidate& original, const BallColourModel& model, int padding) {
    Candidate grown = original;
    auto cropCircle = original.circle;
    cropCircle[2] += padding;

    grown.roi = extractBall(image, cropCircle);
    grown.baselineType = classifyBallType(segmentBallKMeans(grown.roi, 2));
    grown.colours = scoreBallColours(grown.roi, model);

    identityEvidence(grown);
    return grown; // Stored circle/centre remain the original Hough geometry.
}

/*
When at least two candidates are classified as white, selects the one with the largest original detected radius.
*/
int selectAmbiguousCue(const std::vector<Candidate>& candidates) {
    int count = 0, selected = -1;
    double largest = -1;
    for (size_t i = 0; i < candidates.size(); ++i) {
        const auto& c = candidates[i];
        if (c.baselineType.type != "White")
            continue;
        ++count;

        const double size = c.circle[2];
        if (size > largest) {
            largest = size;
            selected = int(i);
        }
    }
    return count >= 2 ? selected : -1;
}

/*
Restricts the selected candidate to the cue identity and prevents every other candidate from receiving that identity.
*/
void reserveCueIdentity(std::vector<Candidate>& candidates, int selected) {
    if (selected < 0)
        return;
    for (size_t i = 0; i < candidates.size(); ++i) {
        auto& c = candidates[i];
        if (int(i) == selected) {
            c.eligible.fill(false);
            c.eligible[0] = true;
        } else {
            c.eligible[0] = false;
            c.identityScore[0] = 0;
        }
    }
}

// ================================ 
// TRAINING AND PREDICTION HELPERS
// ================================ 

/*
Training labels/crop geometry were recorded from the labelled AllBalls image.
*/ 
BallColourModel trainSubmissionModel(const cv::Mat& training) {
    struct Example {
        int number;
        cv::Vec3f circle;
    };
    const std::vector<Example> examples = {{13, {1382.25f, 194.25f, 22.799999f}}, {5, {1365.75f, 146.25f, 21.75f}},     {8, {1532.25f, 143.25f, 21.6f}},      {1, {1142.25f, 147.75f, 20.549999f}}, {12, {1319.25f, 194.25f, 22.35f}},
                                           {15, {1508.25f, 197.25f, 21.299999f}}, {11, {1262.25f, 194.25f, 21.f}},      {9, {1148.25f, 192.75f, 23.549999f}}, {4, {1304.25f, 147.75f, 22.200001f}}, {6, {1418.25f, 149.25f, 22.5f}},
                                           {10, {1208.25f, 195.75f, 18.9f}},      {3, {1248.75f, 146.25f, 19.200001f}}, {7, {1472.25f, 147.75f, 22.049999f}}, {2, {1194.75f, 147.75f, 20.4f}},      {14, {1442.25f, 197.25f, 20.4f}}};
    std::vector<LabelledColourSample> samples;
    for (const auto& e : examples)
        samples.push_back({e.number, extractBall(training, e.circle)});
    return trainBallColourModel(samples, BallHistogramSpace::HSV);
}

std::ofstream checkedCSV(const std::filesystem::path& path) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream stream(path);
    stream.exceptions(std::ios::failbit | std::ios::badbit);
    stream << std::fixed << std::setprecision(9);
    return stream;
}

std::string submissionType(int n) {
    return n < 0 ? "Unknown" : n == 0 ? "White" : n <= 8 ? "Solid" : "Striped";
}

double fraction(double a, double b) {
    return b ? a / b : 0;
}

struct Prediction {
    std::string method;
    std::vector<cv::Vec3f> circles;
    std::vector<cv::Point2f> centres;
    std::vector<int> numbers;
};

std::vector<Candidate> measureCandidates(
    const cv::Mat& image,
    const std::vector<cv::Vec3f>& circles,
    const BallColourModel& model
) {
    const auto centres = transformCentres(circles);
    std::vector<Candidate> candidates;

    for (size_t i = 0; i < circles.size(); ++i) {
        Candidate c;
        c.circle = circles[i];
        c.tableCentre = centres[i];

        c.roi = extractBall(image, c.circle);
        c.baselineType = classifyBallType(segmentBallKMeans(c.roi, 2));
        c.colours = scoreBallColours(c.roi, model);

        identityEvidence(c);
        candidates.push_back(c);
    }
    return candidates;
}

std::vector<Candidate> paddedCandidates(
    const cv::Mat& image,
    const std::vector<Candidate>& original,
    const BallColourModel& model
) {
    auto padded = original;
    for (size_t i = 0; i < padded.size(); ++i) {
        if (original[i].baselineType.type == "White") {
            padded[i] = rawWhitePadding(image, original[i], model, 1);
        }
    }
    return padded;
}

Prediction uniquePrediction(const std::string& method, const std::vector<Candidate>& evidence, int reserved = -1) {
    auto candidates = evidence;
    reserveCueIdentity(candidates, reserved);
    Prediction p;
    p.method = method;
    p.numbers = assignUnique(candidates);
    for (const auto& c : candidates) {
        p.circles.push_back(c.circle);
        p.centres.push_back(c.tableCentre);
    }
    if (reserved >= 0 && p.numbers.at(reserved) != 0)
        throw std::runtime_error("Cue reservation failed");
    return p;
}

// ================================ 
// EVALUATION AGAINST GROUND TRUTH
// ================================

/*
Reads the annotated ball positions for one image from a CSV file and optionally records their ball IDs.
*/
std::vector<cv::Point2f> loadGroundTruth(const std::string& csvPath, const std::string& imageName, std::vector<int>* ballIds = nullptr) {
    std::vector<cv::Point2f> points;
    if (ballIds)
        ballIds->clear();
    std::ifstream file(csvPath);

    if (!file.is_open()) {
        std::cerr << "Could not open: " << csvPath << std::endl;
        return points;
    }

    std::string line;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string filename;

        std::getline(ss, filename, ',');

        if (filename != imageName)
            continue;

        std::vector<std::string> values;
        std::string value;

        while (std::getline(ss, value, ',')) {
            values.push_back(value);
        }

        // Each ball has a Row and Column
        for (size_t i = 0; i + 1 < values.size(); i += 2) {
            if (values[i].empty() || values[i + 1].empty())
                continue;

            float row = std::stof(values[i]);
            float col = std::stof(values[i + 1]);

            points.push_back(cv::Point2f(col, row));
            // Preserve the CSV ball number even when earlier pairs are blank.
            if (ballIds)
                ballIds->push_back(static_cast<int>(i / 2 + 1));
        }

        break;
    }

    return points;
}

/*
Matches detections to annotations within the distance tolerance and calculates detection counts, precision, recall, F1 and mean position error.
*/
EvaluationResult evaluateDetections(const std::vector<cv::Point2f>& predictions, const std::vector<cv::Point2f>& groundTruth, double tolerance) {
    // List valid matches, trying nearby ground-truth balls first.
    std::vector<std::vector<std::pair<double, size_t>>> candidates(predictions.size());

    for (size_t p = 0; p < predictions.size(); ++p) {
        for (size_t g = 0; g < groundTruth.size(); ++g) {
            double distance = std::hypot(double(predictions[p].x) - groundTruth[g].x, double(predictions[p].y) - groundTruth[g].y);
            if (distance <= tolerance)
                candidates[p].emplace_back(distance, g);
        }
        std::sort(candidates[p].begin(), candidates[p].end());
    }

    // Reassign earlier matches when needed to maximize the number of
    // one-to-one matches. Each ground-truth ball counts at most once.
    std::vector<int> matchedPrediction(groundTruth.size(), -1);
    std::function<bool(size_t, std::vector<bool>&)> tryMatch = [&](size_t p, std::vector<bool>& visited) -> bool {
        for (const auto& candidate : candidates[p]) {
            size_t g = candidate.second;
            if (visited[g])
                continue;
            visited[g] = true;

            if (matchedPrediction[g] == -1 || tryMatch(static_cast<size_t>(matchedPrediction[g]), visited)) {
                matchedPrediction[g] = static_cast<int>(p);
                return true;
            }
        }
        return false;
    };

    EvaluationResult result;
    for (size_t p = 0; p < predictions.size(); ++p) {
        std::vector<bool> visited(groundTruth.size(), false);
        if (tryMatch(p, visited))
            ++result.TP;
    }

    // Store the final one-to-one matching.
    result.predictionToGroundTruth.assign(predictions.size(), -1);
    result.groundTruthToPrediction.assign(groundTruth.size(), -1);

    for (size_t g = 0; g < groundTruth.size(); ++g) {
        int p = matchedPrediction[g];

        if (p == -1)
            continue;

        result.predictionToGroundTruth[p] = static_cast<int>(g);
        result.groundTruthToPrediction[g] = p;
    }

    double totalError = 0;
    for (size_t g = 0; g < groundTruth.size(); ++g) {
        if (matchedPrediction[g] == -1)
            continue;
        const cv::Point2f& prediction = predictions[matchedPrediction[g]];
        totalError += std::hypot(double(prediction.x) - groundTruth[g].x, double(prediction.y) - groundTruth[g].y);
    }

    result.FP = static_cast<int>(predictions.size()) - result.TP;
    result.FN = static_cast<int>(groundTruth.size()) - result.TP;

    if (!predictions.empty())
        result.precision = double(result.TP) / predictions.size();
    if (!groundTruth.empty())
        result.recall = double(result.TP) / groundTruth.size();
    if (result.precision + result.recall > 0)
        result.f1 = 2 * result.precision * result.recall / (result.precision + result.recall);
    if (result.TP > 0)
        result.meanError = totalError / result.TP;

    return result;
}

/*
Reads the annotated positions and identities for one image, representing the cue ball with identity zero.
*/
std::vector<LabelledBall> loadLabelledGroundTruth(const std::string& csvPath, const std::string& imageName) {
    std::vector<LabelledBall> balls;
    std::ifstream file(csvPath);

    if (!file.is_open())
        return balls;

    std::string line;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string filename;

        std::getline(ss, filename, ',');

        if (filename != imageName)
            continue;

        std::vector<std::string> values;
        std::string value;

        while (std::getline(ss, value, ','))
            values.push_back(value);

        // 15 numbered balls + white cue ball
        for (int i = 0; i < 16; ++i) {
            int index = 2 * i;

            if (index + 1 >= static_cast<int>(values.size()))
                break;

            if (values[index].empty() || values[index + 1].empty())
                continue;

            float row = std::stof(values[index]);
            float col = std::stof(values[index + 1]);

            LabelledBall ball;
            ball.number = (i == 15) ? 0 : i + 1;
            ball.centre = cv::Point2f(col, row);

            balls.push_back(ball);
        }

        break;
    }

    return balls;
}

struct Counts {
    int gt = 0, tp = 0, fp = 0, fn = 0, correct = 0, types = 0, cues = 0, cueTotal = 0, unknown = 0;
    double error = 0;
    void add(const Counts& c) {
        gt += c.gt;
        tp += c.tp;
        fp += c.fp;
        fn += c.fn;
        correct += c.correct;
        types += c.types;
        cues += c.cues;
        cueTotal += c.cueTotal;
        unknown += c.unknown;
        error += c.error;
    }
};

void writeCounts(std::ostream& out, const Counts& c) {
    out << c.gt << ',' << c.tp + c.fp << ',' << c.tp << ',' << c.fp << ',' << c.fn << ',' << fraction(c.tp, c.tp + c.fp) << ',' << fraction(c.tp, c.gt) << ',' << fraction(2 * c.tp, 2 * c.tp + c.fp + c.fn) << ',' << fraction(c.error, c.tp)
        << ',' << c.correct << ',' << fraction(c.correct, c.tp) << ',' << fraction(c.correct, c.gt) << ',' << c.types << ',' << c.cues << ',' << c.cueTotal << ',' << c.unknown << '\n';
}

// ================================
// VISUALISATION AND FILE OUTPUT
// ================================
/*
Creates an overhead view of the image by applying the table perspective transformation.
*/
cv::Mat transformImage(const cv::Mat& image) {
    cv::Mat matrix = getTableTransformation();
    cv::Mat transformed;
    cv::warpPerspective(image, transformed, matrix, cv::Size(configTableWidth, configTableHeight));
    return transformed;
}

// =============================== 
// VISUAL RESULTS AND FILE EXPORT
// =============================== 

/*
Creates any missing parent directories and saves an image, reporting an error if saving fails.
*/
void saveImage(const std::filesystem::path& path, const cv::Mat& image) {
    if (!path.parent_path().empty())
        std::filesystem::create_directories(path.parent_path());

    if (!cv::imwrite(path.string(), image))
        throw std::runtime_error("Could not save image: " + path.string());
}

/*
Flushes and closes an output file, reporting an error if writing did not finish successfully.
*/
void finishOutputFile(std::ofstream& file) {
    file.flush();
    file.close();

    if (!file)
        throw std::runtime_error("Could not finish writing output CSV");
}

constexpr double ballDiameterMm = 43.0;

std::string finalBallLabel(int number) {
    if (number == 0)
        return "Cue";

    if (number >= 1 && number <= 15)
        return std::to_string(number);

    return "Unknown";
}

/*
Draws a readable predicted ball label near a centre while keeping the text inside the image.
*/
void drawFinalLabel(cv::Mat& image, cv::Point centre, int number, cv::Point offset = {}) {
    const std::string label = finalBallLabel(number);
    const double scale = number < 0 ? 0.36 : 0.45;
    int baseline = 0;

    const cv::Size text = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, scale, 1, &baseline);
    cv::Point anchor(centre.x - text.width / 2 + offset.x, centre.y + text.height / 2 + offset.y);
    
    anchor.x = std::clamp(anchor.x, 2, std::max(2, image.cols - text.width - 2));
    anchor.y = std::clamp(anchor.y, text.height + 2, std::max(text.height + 2, image.rows - baseline - 2));
    
    cv::putText(image, label, anchor, cv::FONT_HERSHEY_SIMPLEX, scale, cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
    cv::putText(image, label, anchor, cv::FONT_HERSHEY_SIMPLEX, scale, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
}

/*
Returns a drawing colour for each predicted ball identity.
*/
cv::Scalar diagramBallColour(int number) {
    switch (number >= 9 && number <= 15 ? number - 8 : number) {
    case 0:
        return {245, 245, 245};
    case 1:
        return {20, 215, 245};
    case 2:
        return {200, 70, 30};
    case 3:
        return {45, 45, 220};
    case 4:
        return {145, 55, 125};
    case 5:
        return {20, 130, 245};
    case 6:
        return {60, 145, 35};
    case 7:
        return {50, 40, 115};
    case 8:
        return {25, 25, 25};
    default:
        return {145, 145, 145};
    }
}

/*
Creates an overhead table diagram showing the predicted ball positions, colours and identity labels.
*/
cv::Mat drawTableDiagram(const std::vector<cv::Point2f>& centres, const std::vector<int>& numbers) {
    cv::Mat diagram(configTableHeight, configTableWidth, CV_8UC3, cv::Scalar(65, 115, 45));
    cv::rectangle(diagram, {0, 0}, {diagram.cols - 1, diagram.rows - 1}, cv::Scalar(160, 190, 160), 1);
    // Nominal physical size; never reuse a source-image Hough radius here.
    
    const double radius = ballDiameterMm / 2;
    constexpr int shift = 8;
    constexpr int fixedScale = 1 << shift;

    for (size_t i = 0; i < centres.size(); ++i) {
        const cv::Point fixedCentre(cvRound(centres[i].x * fixedScale), cvRound(centres[i].y * fixedScale));
        const int fixedRadius = cvRound(radius * fixedScale);
        const cv::Scalar colour = diagramBallColour(numbers[i]);

        cv::circle(diagram, fixedCentre, fixedRadius, colour, cv::FILLED, cv::LINE_AA, shift);
        if (numbers[i] >= 9 && numbers[i] <= 15) {

            // Clip a coloured middle band to a white disk, preserving its round outline.
            const int half = cvCeil(radius) + 2;
            const cv::Point origin(cvFloor(centres[i].x) - half, cvFloor(centres[i].y) - half);
            const cv::Rect patch = cv::Rect(origin.x, origin.y, 2 * half + 1, 2 * half + 1) & cv::Rect(0, 0, diagram.cols, diagram.rows);
            
            if (!patch.empty()) {
                cv::Mat mask = cv::Mat::zeros(patch.size(), CV_8UC1);
                const cv::Point localFixed = fixedCentre - cv::Point(patch.x * fixedScale, patch.y * fixedScale);
                cv::circle(mask, localFixed, fixedRadius, cv::Scalar(255), cv::FILLED, cv::LINE_AA, shift);
                cv::Mat ball(patch.size(), CV_8UC3, cv::Scalar(245, 245, 245));
                const int y = cvRound(centres[i].y) - patch.y;
                cv::rectangle(ball, {0, y - 10}, {patch.width - 1, y + 10}, colour, cv::FILLED);
                ball.copyTo(diagram(patch), mask);
            }
        }
        cv::circle(diagram, fixedCentre, fixedRadius, cv::Scalar(230, 230, 230), 1, cv::LINE_AA, shift);
    }
    // Draw labels last so neighbouring ball symbols cannot paint over them.
    for (size_t i = 0; i < centres.size(); ++i)
        drawFinalLabel(diagram, {cvRound(centres[i].x), cvRound(centres[i].y)}, numbers[i]);
    return diagram;
}

/*
Saves the numbered original image, overhead views, table diagram and a CSV of predicted ball identities and coordinates.
*/
void saveFinalResults(const cv::Mat& image, const std::filesystem::path& directory, const std::vector<cv::Vec3f>& circles, const std::vector<cv::Point2f>& tableCentres, const std::vector<std::string>& types, const std::vector<int>& numbers) {
    if (circles.size() != tableCentres.size() || circles.size() != types.size() || circles.size() != numbers.size())
        throw std::runtime_error("Final result vectors have inconsistent candidate counts");

    std::filesystem::create_directories(directory);
    cv::Mat original = image.clone();
    const cv::Mat rectified = transformImage(image);
    cv::Mat overhead = rectified.clone();

    for (size_t i = 0; i < circles.size(); ++i) {
        const cv::Point sourceCentre(cvRound(circles[i][0]), cvRound(circles[i][1]));
        cv::circle(original, sourceCentre, cvRound(circles[i][2]), cv::Scalar(0, 220, 255), 2, cv::LINE_AA);
        drawFinalLabel(original, sourceCentre, numbers[i]);

        const cv::Point tableCentre(cvRound(tableCentres[i].x), cvRound(tableCentres[i].y));
        cv::drawMarker(overhead, tableCentre, cv::Scalar(0, 220, 255), cv::MARKER_CROSS, 9, 1, cv::LINE_AA);
        drawFinalLabel(overhead, tableCentre, numbers[i], {0, -12});
    }
    saveImage(directory / "original_numbered.png", original);
    saveImage(directory / "table_rectified.png", rectified);
    saveImage(directory / "table_numbered.png", overhead);
    saveImage(directory / "table_diagram.png", drawTableDiagram(tableCentres, numbers));

    std::ofstream csv(directory / "balls.csv");
    if (!csv.is_open())
        throw std::runtime_error("Could not create final results CSV: " + directory.string());

    csv << "candidate,predictedType,predictedNumber,originalX,originalY,originalRadiusPixels,tableX,tableY\n" << std::fixed << std::setprecision(6);
    for (size_t i = 0; i < circles.size(); ++i)
        csv << i + 1 << ',' << types[i] << ',' << numbers[i] << ',' << circles[i][0] << ',' << circles[i][1] << ',' << circles[i][2] << ',' << tableCentres[i].x << ',' << tableCentres[i].y << '\n';
    
    finishOutputFile(csv);
    std::cout << "Final images and table coordinates saved to: " << directory << std::endl;
}

/*
Returns a copy of the image with each detected circle boundary and centre marked.
*/
cv::Mat drawDetectedBalls(const cv::Mat& image, const std::vector<cv::Vec3f>& circles) {
    cv::Mat result = image.clone();

    for (const cv::Vec3f& detected : circles) {
        cv::Point center(cvRound(detected[0]), cvRound(detected[1]));

        int radius = cvRound(detected[2]);

        // Draw circle boundary
        cv::circle(result, center, radius, cv::Scalar(0, 0, 255), 2);

        // Draw centre
        cv::circle(result, center, 2, cv::Scalar(255, 0, 0), -1);
    }

    return result;
}

// ==============
// MAIN PIPELINE
// ==============
struct MainPipelineResult {
    DetectionResult detection;
    std::vector<Candidate> original, padded;
    int reservedCue = -1;
    Prediction prediction;
};

MainPipelineResult runMainPipeline(const cv::Mat& image, const BallColourModel& model) {
    MainPipelineResult result;
    // 1. Locate balls and reject circles near pockets.
    result.detection = runDetection(image, HoughParams{});

    // 2. Measure K=2 white coverage and HSV histogram colour evidence.
    result.original = measureCandidates(image, result.detection.filteredCircles, model);

    // 3. Reanalyse originally White candidates using a one-pixel larger crop.
    result.padded = paddedCandidates(image, result.original, model);

    // 4. With multiple original White candidates, reserve the largest raw circle.
    result.reservedCue = selectAmbiguousCue(result.original);

    // 5. Assign unique identities using the padded evidence and cue reservation.
    result.prediction = uniquePrediction("largest_raw_radius", result.padded, result.reservedCue);
    
    return result;
}

// ===================
// BATCH COORDINATION 
// ===================
void runSubmission() {
    const auto directory = std::filesystem::path(configApplicationOutputDirectory);
    std::filesystem::create_directories(directory);
    const auto training = cv::imread(configMediaDirectory + configTuningImage);
    if (training.empty())
        throw std::runtime_error("Could not load Media/AllBalls.jpeg; run from the project root");
    
    // Build reference colour histograms from labelled AllBalls crops only.
    const auto model = trainSubmissionModel(training);
    
    {
        cv::FileStorage saved((directory / "trained_model.yml").string(), cv::FileStorage::WRITE);
        if (!saved.isOpened())
            throw std::runtime_error("Could not save trained model");
        saved << "trainingImage" << "AllBalls.jpeg" << "neutralChroma" << 15 << "minimumBodyPixels" << 12;
        for (int c = 0; c < 8; ++c)
            saved << ("colour" + std::to_string(c + 1)) << model.histograms[c];
    }

    auto metrics = checkedCSV(directory / "metrics.csv");
    metrics << "split,image,method,groundTruth,detections,TP,FP,FN,precision,recall,F1,meanError,"
               "correctNumbers,numberAccuracy,endToEndAccuracy,correctTypes,correctCues,cueTotal,unknownPredictions\n";
    
    auto predictions = checkedCSV(directory / "predictions.csv");
    predictions << "image,method,candidate,predictedNumber,tableX,tableY\n";
    
    auto matches = checkedCSV(directory / "evaluation_matches.csv");
    matches << "image,method,candidate,trueNumber,predictedNumber,distance,status\n";
    
    auto evidence = checkedCSV(directory / "recognition_evidence.csv");
    evidence << "image,candidate,originalType,originalLightPercent,analysisLightPercent,bodyPixels,padding,reservedCue\n";
    
    std::map<std::string, Counts> totals;
    std::map<std::tuple<std::string, int, int>, int> confusion;
    
    for (const auto& filename : configAllImages()) {
        const auto image = cv::imread(configMediaDirectory + filename);
        if (image.empty())
            throw std::runtime_error("Could not load " + filename);
        
        const std::string stem = std::filesystem::path(filename).stem().string();
        const auto main = runMainPipeline(image, model);
        const auto& detection = main.detection;
        const auto& circles = detection.filteredCircles;
        const auto& original = main.original;
        const auto& padded = main.padded;
        const int raw = main.reservedCue;

        std::vector<Prediction> profiles{main.prediction};

        // All inference and exports are completed before reading this image's labels.
        for (const auto& p : profiles) {
            const auto destination = directory / "mainFinalResults" / stem;
            std::vector<std::string> types;
            for (int n : p.numbers)
                types.push_back(submissionType(n));
            saveFinalResults(image, destination, p.circles, p.centres, types, p.numbers);
            for (size_t i = 0; i < p.circles.size(); ++i)
                predictions << filename << ',' << p.method << ',' << i + 1 << ',' << p.numbers[i] << ',' << p.centres[i].x << ',' << p.centres[i].y << '\n';
        }

        const auto stageDir = directory / "mainDetectionResults" / stem;
        saveImage(stageDir / "masked.png", detection.maskedImage);
        cv::Mat gray, blur;
        cv::cvtColor(detection.maskedImage, gray, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(gray, blur, {5, 5}, 1.5);

        saveImage(stageDir / "blurred.png", blur);
        saveImage(stageDir / "raw_circles.png", drawDetectedBalls(image, detection.rawCircles));
        saveImage(stageDir / "filtered_circles.png", drawDetectedBalls(image, circles));

        const auto typeDir = directory / "mainTypeRecognitionResults" / stem;
        const auto colourDir = directory / "mainColourRecognitionResults" / stem;
        
        for (size_t i = 0; i < circles.size(); ++i) {
            const auto& sample = padded[i];
            const auto segmentation = segmentBallKMeans(sample.roi, 2);
            if (!segmentation.segmentedImage.empty())
                saveImage(typeDir / ("ball_" + std::to_string(i + 1) + "_kmeans.png"), segmentation.segmentedImage);
            if (!sample.colours.bodyMask.empty()) {
                cv::Mat body;
                sample.roi.image.copyTo(body, sample.colours.bodyMask);
                saveImage(colourDir / ("ball_" + std::to_string(i + 1) + "_body.png"), body);
            }
        }

        for (size_t i = 0; i < circles.size(); ++i) {
            evidence << filename << ',' << i + 1 << ',' << original[i].baselineType.type << ',' << original[i].baselineType.lightPercent << ',' << padded[i].baselineType.lightPercent << ',' << padded[i].colours.pixels << ','
                     << (original[i].baselineType.type == "White") << ',' << (int(i) == raw) << '\n';
        }

        const auto truth = loadLabelledGroundTruth(configGroundTruthPath, filename);
        if (truth.empty()) {
            std::cerr << "No annotations for " << filename << "; predictions saved, evaluation skipped\n";
            continue;
        }

        std::vector<cv::Point2f> trueCentres;
        for (const auto& t : truth)
            trueCentres.push_back(t.centre);

        for (const auto& p : profiles) {
            const auto e = evaluateDetections(p.centres, trueCentres, configMatchTolerance);
            Counts count;
            count.gt = int(truth.size());
            count.tp = e.TP;
            count.fp = e.FP;
            count.fn = e.FN;
            count.error = e.meanError * e.TP;

            for (size_t i = 0; i < p.circles.size(); ++i) {
                const int j = e.predictionToGroundTruth[i], n = p.numbers[i];
                count.unknown += n < 0;

                if (j >= 0) {
                    const int actual = truth[j].number;
                    count.correct += n == actual;
                    count.types += submissionType(n) == submissionType(actual);
                    count.cues += actual == 0 && n == 0;
                    count.cueTotal += actual == 0;
                    if (filename != configTuningImage)
                        ++confusion[{p.method, actual, n}];
                    matches << filename << ',' << p.method << ',' << i + 1 << ',' << actual << ',' << n << ',' << cv::norm(p.centres[i] - truth[j].centre) << ',' << (n == actual ? "correct" : "wrong_identity") << '\n';
                } else
                    matches << filename << ',' << p.method << ',' << i + 1 << ",," << n << ",,false_positive\n";
            }

            for (size_t j = 0; j < truth.size(); ++j)
                if (e.groundTruthToPrediction[j] < 0)
                    matches << filename << ',' << p.method << ",," << truth[j].number << ",,,missed\n";
            
            const bool evaluation = filename != configTuningImage;
            metrics << (evaluation ? "development" : "training_resubstitution") << ',' << filename << ',' << p.method << ',';
            writeCounts(metrics, count);

            if (evaluation)
                totals[p.method].add(count);

            if (p.method == "largest_raw_radius")
                std::cout << filename << ": TP=" << e.TP << " FP=" << e.FP << " FN=" << e.FN << " identities=" << count.correct << '/' << e.TP << '\n';
        }
    }
    for (const auto& [method, count] : totals) {
        metrics << "development,ALL," << method << ',';
        writeCounts(metrics, count);
    }

    auto matrix = checkedCSV(directory / "mainEvaluationResults" / "identity_confusion.csv");
    matrix << "method,trueNumber,predictedNumber,count\n";
    for (const auto& [key, n] : confusion)
        matrix << std::get<0>(key) << ',' << std::get<1>(key) << ',' << std::get<2>(key) << ',' << n << '\n';
    
    const auto selected = totals.find("largest_raw_radius");
    if (selected != totals.end()) {
        const auto& c = selected->second;
        std::cout << "Table1--6: " << c.tp << " TP, " << c.fp << " FP, " << c.fn << " FN; " << c.correct << '/' << c.tp << " correct identities, " << 100 * fraction(c.correct, c.gt) << "% end-to-end\n";
    }
}

// =========================================== 
// HOUGH PARAMETER SEARCH
// Runs only with POOL_BALL_MODE=grid-search
// =========================================== 
const std::string IMAGE_PATH = configMediaDirectory + configTuningImage;
const std::string CSV_PATH = configGroundTruthPath;
const double MATCH_TOLERANCE = configMatchTolerance;

/*
Searches 480 Hough parameter combinations on AllBalls and exports the best detection settings.
*/
void runGridSearch() {
    cv::Mat image = cv::imread(IMAGE_PATH);

    if (image.empty()) {
        std::cerr << "Could not open AllBalls.jpeg" << std::endl;
        return;
    }

    cv::Mat maskedImage = maskTable(image);

    std::vector<cv::Point2f> groundTruth = loadGroundTruth(CSV_PATH, configTuningImage);

    if (groundTruth.size() != 16) {
        std::cerr << "Invalid ground truth: " << groundTruth.size() << " balls" << std::endl;
        return;
    }

    // DEFINE PARAMETER SEARCH SPACE
    std::vector<double> dpValues = {1.0, 1.5, 2.0};
    std::vector<double> minDistValues = {15, 18, 20, 22, 25};
    std::vector<double> param1Values = {75, 100, 125, 150};
    std::vector<double> param2Values = {15, 20, 25, 30, 35, 40, 45, 50};
    // 480 configurations

    std::filesystem::create_directories(configOutputDirectory);
    std::ofstream file(configOutputDirectory + "grid_search.csv");

    if (!file.is_open()) {
        std::cerr << "Could not create output/grid_search.csv" << std::endl;
        return;
    }

    file << "dp,minDist,param1,param2,"
         << "minRadius,maxRadius,detections,"
         << "TP,FP,FN,precision,recall,F1,meanError\n";

    double bestTP = -1;
    int bestFP = std::numeric_limits<int>::max();
    double bestError = std::numeric_limits<double>::infinity();

    HoughParams bestParams;
    EvaluationResult bestResult;

    int count = 0;
    const size_t totalConfigurations = dpValues.size() * minDistValues.size() * param1Values.size() * param2Values.size();

    // TEST EVERY PARAMETER COMBINATION
    for (double dp : dpValues) {
        for (double minDist : minDistValues) {
            for (double p1 : param1Values) {
                for (double p2 : param2Values) {
                    HoughParams params;

                    params.dp = dp;
                    params.minDist = minDist;
                    params.param1 = p1;
                    params.param2 = p2;

                    // Fixed during this first search
                    params.minRadius = 15;
                    params.maxRadius = 30;

                    // Detect balls
                    std::vector<cv::Vec3f> circles = detectBalls(maskedImage, params);

                    // Transform detected centres
                    std::vector<cv::Point2f> predictions = transformCentres(circles);

                    // Evaluate
                    EvaluationResult result = evaluateDetections(predictions, groundTruth, MATCH_TOLERANCE);

                    // Save results
                    file << dp << "," << minDist << "," << p1 << "," << p2 << "," << params.minRadius << "," << params.maxRadius << "," << circles.size() << "," << result.TP << "," << result.FP << "," << result.FN << "," << result.precision
                         << "," << result.recall << "," << result.f1 << "," << result.meanError << "\n";

                    count++;

                    // Prioritise recall, then precision
                    // then centre localisation accuracy.
                    double candidateError = result.TP > 0 ? result.meanError : std::numeric_limits<double>::infinity();

                    bool better = (
                        // 1. Detect more actual balls
                        result.TP > bestTP ||

                        // 2. Same number of balls: fewer false positives
                        (result.TP == bestTP && result.FP < bestFP) ||

                        // 3. Same TP and FP: better localisation
                        (result.TP == bestTP && result.FP == bestFP && candidateError < bestError));

                    if (better) {
                        bestTP = result.TP;
                        bestFP = result.FP;
                        bestError = candidateError;

                        bestParams = params;
                        bestResult = result;
                    }

                    std::cout << "\rTested " << count << "/" << totalConfigurations << " configurations" << std::flush;
                }
            }
        }
    }

    finishOutputFile(file);

    // DISPLAY BEST CONFIGURATION
    std::cout << std::fixed << std::setprecision(4);

    std::cout << "\n\n===== GRID SEARCH COMPLETE =====\n";
    std::cout << "Configurations tested: " << count << std::endl;

    std::cout << "\nBEST PARAMETERS:\n";
    std::cout << "dp: " << bestParams.dp << std::endl;
    std::cout << "minDist: " << bestParams.minDist << std::endl;
    std::cout << "param1: " << bestParams.param1 << std::endl;
    std::cout << "param2: " << bestParams.param2 << std::endl;
    std::cout << "minRadius: " << bestParams.minRadius << std::endl;
    std::cout << "maxRadius: " << bestParams.maxRadius << std::endl;

    std::cout << "\nBEST PERFORMANCE:\n";
    std::cout << "TP: " << bestResult.TP << std::endl;
    std::cout << "FP: " << bestResult.FP << std::endl;
    std::cout << "FN: " << bestResult.FN << std::endl;
    std::cout << "Precision: " << bestResult.precision << std::endl;
    std::cout << "Recall: " << bestResult.recall << std::endl;
    std::cout << "F1: " << bestResult.f1 << std::endl;

    if (bestResult.TP > 0)
        std::cout << "Mean error: " << bestResult.meanError << " pixels" << std::endl;
    else
        std::cout << "Mean error: N/A" << std::endl;

    // SAVE VISUALISATION OF BEST CONFIGURATION
    std::vector<cv::Vec3f> bestCircles = detectBalls(maskedImage, bestParams);
    cv::Mat bestImage = drawDetectedBalls(maskedImage, bestCircles);

    saveImage(configOutputDirectory + "best_detection.jpeg", bestImage);

    std::cout << "\nSaved: output/grid_search.csv" << std::endl;
    std::cout << "Saved: output/best_detection.jpeg" << std::endl;
}

// ============
// EXPERIMENTS
// ============
struct HistoricalDetectionVariant {
    std::vector<cv::Vec3f> circles;
    cv::Mat foreground, distance, markerView;
};

// Foreground is defined by how different each pixel is from the median-ish
// green felt colour estimated from the centre region (illustrative heuristic).
// It deliberately avoids consulting any ground-truth ball locations.
cv::Mat historicalForeground(const cv::Mat& image) {
    cv::Mat hsv, foreground, cleaned;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
    // Green felt has hue around 35-90 in OpenCV's 0-179 hue representation.
    // Thresholds are exploratory and NOT the historical experiment settings.
    cv::Mat green;
    cv::inRange(hsv, cv::Scalar(30, 35, 20), cv::Scalar(100, 255, 255), green);
    cv::bitwise_not(green, foreground);
    cv::Mat polygon=cv::Mat::zeros(image.size(),CV_8U);
    cv::fillConvexPoly(polygon,getTableCorners(),cv::Scalar(255));
    cv::bitwise_and(foreground,polygon,foreground);
    cv::morphologyEx(foreground, cleaned, cv::MORPH_OPEN,
                     cv::getStructuringElement(cv::MORPH_ELLIPSE, {3,3}));
    return cleaned;
}

// EXPERIMENT: Watershed augmentation. Preserve all selected Hough candidates;
// investigate whether distance peaks in segmented clusters can add circles.
// Geometric validation mirrors the IDEA in the notes (area/circularity/aspect
// and proximity), but this is not guaranteed identical to the earlier branch.
HistoricalDetectionVariant experimentWatershed(const cv::Mat& image,
                                                const std::vector<cv::Vec3f>& base) {
    HistoricalDetectionVariant result;
    result.circles = base;
    result.foreground = historicalForeground(image);
    cv::distanceTransform(result.foreground, result.distance, cv::DIST_L2, 3);
    // Original research branch restricts watershed to crowded groups of
    // nearby Hough candidates. Build padded windows around such pairs.
    cv::Mat crowded = cv::Mat::zeros(image.size(),CV_8U);
    for (size_t i=0;i<base.size();++i)
        for (size_t j=i+1;j<base.size();++j) {
            const cv::Point2f a(base[i][0],base[i][1]), b(base[j][0],base[j][1]);
            if (cv::norm(a-b)>1.4*(base[i][2]+base[j][2])) continue;
            for (size_t k : {i,j}) {
                const int r=cvRound(base[k][2]*3.0);
                cv::circle(crowded,{cvRound(base[k][0]),cvRound(base[k][1])},
                           r,cv::Scalar(255),cv::FILLED);
            }
        }
    cv::bitwise_and(result.foreground,crowded,result.foreground);
    cv::Mat foreground8=result.foreground.clone();
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(foreground8, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    cv::Mat markers(image.size(), CV_32S, cv::Scalar(1));
    markers.setTo(0, result.foreground);
    int label=2;
    for (const auto& contour : contours) {
        if (cv::contourArea(contour) < 100) continue;
        cv::drawContours(markers, std::vector<std::vector<cv::Point>>{contour}, 0,
                         cv::Scalar(label++), cv::FILLED);
    }
    // Watershed partitions the foreground basins; only approved circular basins
    // not overlapping existing detections become additional candidates.
    if (label > 2) cv::watershed(image, markers);
    for (int k=2; k<label; ++k) {
        cv::Mat basin = markers == k;
        std::vector<std::vector<cv::Point>> regions;
        cv::findContours(basin, regions, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        for (const auto& contour : regions) {
            const double area=cv::contourArea(contour);
            const double nominal=CV_PI*22.0*22.0;
            if (area < 0.25*nominal || area > 1.8*nominal) continue;
            const double perimeter=cv::arcLength(contour,true);
            if (perimeter <= 0 || 4*CV_PI*area/(perimeter*perimeter) < 0.35) continue;
            const cv::Rect rect=cv::boundingRect(contour);
            if (rect.width==0 || rect.height==0 ||
                double(std::max(rect.width,rect.height))/std::min(rect.width,rect.height)>2.2) continue;
            cv::Point2f centre;
            float radius=0;
            cv::minEnclosingCircle(contour,centre,radius);
            if (radius<15 || radius>30) continue;
            cv::Mat disc=cv::Mat::zeros(image.size(),CV_8U);
            cv::circle(disc,centre,cvRound(radius),cv::Scalar(255),cv::FILLED);
            cv::Mat support;
            cv::bitwise_and(disc,result.foreground,support);
            if (cv::countNonZero(support)<0.35*cv::countNonZero(disc)) continue;
            bool duplicate=false;
            for (const auto& c:result.circles)
                if (cv::norm(centre-cv::Point2f(c[0],c[1]))<20) duplicate=true;
            if (duplicate) continue;
            const cv::Vec3f proposal(centre.x,centre.y,radius);
            if (!filterPocketDetections({proposal}).empty()) result.circles.push_back(proposal);
        }
    }
    cv::normalize(result.distance,result.distance,0,255,cv::NORM_MINMAX,CV_8U);
    cv::applyColorMap(result.distance,result.distance,cv::COLORMAP_TURBO);
    markers.convertTo(result.markerView,CV_8U,13);
    return result;
}

// EXPERIMENT: Original cluster-local Hough followed by watershed validation.
// Kept self-contained for the single-file submission. The algorithm and default
// settings follow experiments/segmentationGuidedDetection.{h,cpp}; no labels
// enter region selection, local circle proposals, or watershed validation.
// Baseline circles outside routed clusters are retained; local proposals replace
// the routed interior, with the original empty-search fallback.
namespace guided_hough_watershed {

struct SegmentationHoughOptions {
    int clothTileSize = 128;
    int minimumClothSamples = 30;
    double clothMinimumDistance = 16;
    double clothMaximumDistance = 45;
    double clothMADMultiplier = 2.5;
    int tableInsetKernel = 41; // Suppress rail/shadow connections before component labelling.
    int closingKernel = 3;
    int minimumComponentArea = 80;
    double clusterAreaRatio = 1.6;
    double clusterAspectRatio = 1.45;
    double elongatedMinimumAreaRatio = 0.8;
    double cropPaddingRadii = 1.25;
    double routingOverlapRadii = 0.5;
    double localAccumulatorThreshold = 22;
    double localMinCentreDistance = 12;
    double localMinRadiusRatio = 0.7;
    double localMaxRadiusRatio = 1.6;
    int absoluteMinLocalRadius = 12;
    int absoluteMaxLocalRadius = 32;
    int angularSamples = 72;
    double edgeTolerance = 2.5;
    double minimumRadialAlignment = 0.6;
    double minimumEdgeSupport = 0.30;
    double minimumArcDegrees = 75;
    double minimumForegroundSupport = 0.35;
    double duplicateDistanceRadii = 0.8;
    double watershedMinimumAreaRatio = 0.15;
    double watershedMaximumAreaRatio = 2.0;
};

struct GuidedProposal {
    cv::Vec3f circle;
    int region = 0;
    double edgeSupport = 0, arcDegrees = 0, foregroundSupport = 0, quality = 0;
    bool accepted = false, watershedAccepted = false;
    int guidedIndex = -1;
    double watershedAreaRatio = 0;
    std::string reason, watershedReason;
};

struct GuidedCluster {
    cv::Rect core, crop;
    double expectedRadius = 0;
    int minRadius = 0, maxRadius = 0;
    bool usedEmptySearchFallback = false;
    cv::Mat markers, labels;
    std::vector<int> seedIndices;
};

struct SegmentationGuidedResult {
    cv::Mat foreground, routingMask;
    std::vector<GuidedCluster> clusters;
    std::vector<GuidedProposal> proposals;
    std::vector<cv::Vec3f> baseline, standardOutside, guided, watershed;
};


namespace {

double median(std::vector<double> values) {
    if (values.empty())
        return 0;
    const size_t middle = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + middle, values.end());
    return values[middle];
}

cv::Point centre(const cv::Vec3f& c) {
    return {cvRound(c[0]), cvRound(c[1])};
}

// A perspective-aware size guide using only known geometry and configured
// Hough size bounds. No circles or annotations are needed to locate clusters.
double expectedRadius(double y) {
    const auto corners = getTableCorners();
    int top = corners.front().y, bottom = top;
    for (const auto& p : corners) {
        top = std::min(top, p.y);
        bottom = std::max(bottom, p.y);
    }
    const HoughParams hough;
    const double fraction = std::clamp((y - top) / std::max(1, bottom - top), 0.0, 1.0);
    return hough.minRadius + fraction * (hough.maxRadius - hough.minRadius);
}

struct ClothModel {
    cv::Vec3d colour;
    double threshold = 0;
    bool valid = false;
};

ClothModel clothModel(const cv::Mat& hsv, const cv::Mat& lab, const cv::Mat& table, const cv::Rect& crop, const SegmentationHoughOptions& p) {
    std::vector<cv::Vec3b> samples;
    std::vector<double> channels[3];
    for (int y = crop.y; y < crop.br().y; y += 2)
        for (int x = crop.x; x < crop.br().x; x += 2) {
            const auto h = hsv.at<cv::Vec3b>(y, x);
            if (table.at<uchar>(y, x) && h[0] >= 35 && h[0] <= 95 && h[1] >= 35 && h[2] >= 30) {
                const auto pixel = lab.at<cv::Vec3b>(y, x);
                samples.push_back(pixel);
                for (int c = 0; c < 3; ++c)
                    channels[c].push_back(pixel[c]);
            }
        }
    ClothModel model;
    if (static_cast<int>(samples.size()) < p.minimumClothSamples)
        return model;
    model.valid = true;
    model.colour = {median(channels[0]), median(channels[1]), median(channels[2])};
    std::vector<double> distances, deviations;
    for (const auto& pixel : samples)
        distances.push_back(cv::norm(cv::Vec3d(pixel) - model.colour));
    const double middle = median(distances);
    for (double d : distances)
        deviations.push_back(std::abs(d - middle));
    model.threshold = std::clamp(middle + p.clothMADMultiplier * median(deviations), p.clothMinimumDistance, p.clothMaximumDistance);
    return model;
}

cv::Mat foreground(const cv::Mat& image, const cv::Mat& table, const SegmentationHoughOptions& p) {
    cv::Mat inset, hsv, lab;
    cv::erode(table, inset, cv::getStructuringElement(cv::MORPH_ELLIPSE, {p.tableInsetKernel, p.tableInsetKernel}));
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
    cv::cvtColor(image, lab, cv::COLOR_BGR2Lab);
    const auto fallback = clothModel(hsv, lab, inset, {0, 0, image.cols, image.rows}, p);
    cv::Mat mask = cv::Mat::zeros(image.size(), CV_8UC1);
    if (!fallback.valid)
        return mask; // No usable cloth model: the algorithm falls back to standard Hough everywhere.
    for (int top = 0; top < image.rows; top += p.clothTileSize)
        for (int left = 0; left < image.cols; left += p.clothTileSize) {
            const cv::Rect tile(left, top, std::min(p.clothTileSize, image.cols - left), std::min(p.clothTileSize, image.rows - top));
            auto local = clothModel(hsv, lab, inset, tile, p);
            if (!local.valid)
                local = fallback;
            for (int y = tile.y; y < tile.br().y; ++y)
                for (int x = tile.x; x < tile.br().x; ++x)
                    if (inset.at<uchar>(y, x) && cv::norm(cv::Vec3d(lab.at<cv::Vec3b>(y, x)) - local.colour) > local.threshold)
                        mask.at<uchar>(y, x) = 255;
        }
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_ELLIPSE, {p.closingKernel, p.closingKernel}));
    cv::Mat holes;
    cv::copyMakeBorder(mask, holes, 1, 1, 1, 1, cv::BORDER_CONSTANT, cv::Scalar(0));
    cv::floodFill(holes, {0, 0}, cv::Scalar(255));
    cv::bitwise_not(holes, holes);
    cv::bitwise_or(mask, holes(cv::Rect(1, 1, image.cols, image.rows)), mask);
    cv::bitwise_and(mask, inset, mask);
    cv::Mat labels, stats, centres, clean = cv::Mat::zeros(mask.size(), CV_8UC1);
    const int count = cv::connectedComponentsWithStats(mask, labels, stats, centres, 8, CV_32S);
    for (int label = 1; label < count; ++label)
        if (stats.at<int>(label, cv::CC_STAT_AREA) >= p.minimumComponentArea)
            clean.setTo(255, labels == label);
    return clean;
}

std::vector<GuidedCluster> clusterWindows(const cv::Mat& mask, const cv::Mat& table, const SegmentationHoughOptions& p) {
    cv::Mat labels, stats, centres, inner, border;
    const int count = cv::connectedComponentsWithStats(mask, labels, stats, centres, 8, CV_32S);
    cv::erode(table, inner, cv::getStructuringElement(cv::MORPH_ELLIPSE, {41, 41}));
    cv::subtract(table, inner, border);
    std::vector<GuidedCluster> result;
    const double tableArea = cv::countNonZero(table);
    for (int label = 1; label < count; ++label) {
        const int area = stats.at<int>(label, cv::CC_STAT_AREA);
        const cv::Rect bounds(stats.at<int>(label, cv::CC_STAT_LEFT), stats.at<int>(label, cv::CC_STAT_TOP), stats.at<int>(label, cv::CC_STAT_WIDTH), stats.at<int>(label, cv::CC_STAT_HEIGHT));
        const double radius = expectedRadius(centres.at<double>(label, 1));
        const double ratio = area / (CV_PI * radius * radius);
        const double aspect = double(std::max(bounds.width, bounds.height)) / std::min(bounds.width, bounds.height);
        if (ratio < p.clusterAreaRatio && !(aspect >= p.clusterAspectRatio && ratio >= p.elongatedMinimumAreaRatio))
            continue;
        cv::Mat atBorder;
        cv::bitwise_and(labels == label, border, atBorder);
        if (area > 0.10 * tableArea || (aspect > 4 && double(cv::countNonZero(atBorder)) / area > 0.6))
            continue; // Reject very large background failures and long rail-like strips.
        GuidedCluster region;
        region.core = bounds;
        region.expectedRadius = radius;
        const int padding = cvRound(p.cropPaddingRadii * radius);
        region.crop = cv::Rect(bounds.x - padding, bounds.y - padding, bounds.width + 2 * padding, bounds.height + 2 * padding) & cv::Rect(0, 0, mask.cols, mask.rows);
        result.push_back(region);
    }
    bool merged = true;
    while (merged) {
        merged = false;
        for (size_t i = 0; i < result.size() && !merged; ++i)
            for (size_t j = i + 1; j < result.size(); ++j)
                if (!(result[i].crop & result[j].crop).empty()) {
                    result[i].crop |= result[j].crop;
                    result[i].core |= result[j].core;
                    result[i].expectedRadius = expectedRadius(result[i].core.y + result[i].core.height / 2.0);
                    result.erase(result.begin() + j);
                    merged = true;
                    break;
                }
    }
    return result;
}

void edgeEvidence(GuidedProposal& proposal, const GuidedCluster& region, const cv::Mat& edges, const cv::Mat& gx, const cv::Mat& gy, const cv::Mat& foregroundMask, const SegmentationHoughOptions& p) {
    const auto& c = proposal.circle;
    const double cx = c[0] - region.crop.x, cy = c[1] - region.crop.y, radius = c[2];
    std::vector<bool> supported(p.angularSamples, false);
    const int tolerance = static_cast<int>(std::ceil(p.edgeTolerance));
    for (int i = 0; i < p.angularSamples; ++i) {
        const double angle = 2 * CV_PI * i / p.angularSamples;
        const double dx = std::cos(angle), dy = std::sin(angle);
        const double x = cx + radius * dx, y = cy + radius * dy;
        for (int oy = -tolerance; oy <= tolerance && !supported[i]; ++oy)
            for (int ox = -tolerance; ox <= tolerance; ++ox) {
                const int ex = cvRound(x) + ox, ey = cvRound(y) + oy;
                if (ex < 0 || ey < 0 || ex >= edges.cols || ey >= edges.rows || !edges.at<uchar>(ey, ex) || std::hypot(ex - x, ey - y) > p.edgeTolerance)
                    continue;
                const double sx = gx.at<float>(ey, ex), sy = gy.at<float>(ey, ex);
                const double strength = std::hypot(sx, sy);
                if (strength > 1 && std::abs(sx * dx + sy * dy) / strength >= p.minimumRadialAlignment) {
                    supported[i] = true;
                    break;
                }
            }
    }
    proposal.edgeSupport = double(std::count(supported.begin(), supported.end(), true)) / p.angularSamples;
    auto closed = supported;
    for (int i = 0; i < p.angularSamples; ++i)
        if (!supported[i] && supported[(i + p.angularSamples - 1) % p.angularSamples] && supported[(i + 1) % p.angularSamples])
            closed[i] = true; // Bridge one missing sample, without changing raw support fraction.
    int longest = 0, run = 0;
    for (int i = 0; i < 2 * p.angularSamples; ++i) {
        run = closed[i % p.angularSamples] ? std::min(p.angularSamples, run + 1) : 0;
        longest = std::max(longest, run);
    }
    proposal.arcDegrees = 360.0 * longest / p.angularSamples;
    cv::Mat disk = cv::Mat::zeros(region.crop.size(), CV_8UC1), support;
    cv::circle(disk, {cvRound(cx), cvRound(cy)}, cvRound(radius), cv::Scalar(255), cv::FILLED);
    cv::bitwise_and(disk, foregroundMask(region.crop), support);
    proposal.foregroundSupport = double(cv::countNonZero(support)) / std::max(1, cv::countNonZero(disk));
    proposal.quality = 0.7 * proposal.edgeSupport + 0.3 * proposal.arcDegrees / 360;
}

bool duplicate(const cv::Vec3f& a, const std::vector<cv::Vec3f>& selected, const SegmentationHoughOptions& p) {
    for (const auto& b : selected)
        if (cv::norm(cv::Point2f(a[0], a[1]) - cv::Point2f(b[0], b[1])) < p.duplicateDistanceRadii * std::min(a[2], b[2]))
            return true;
    return false;
}

void watershedCheck(const cv::Mat& image, const cv::Mat& table, SegmentationGuidedResult& result, const SegmentationHoughOptions& p) {
    std::vector<bool> keep(result.guided.size(), true);
    for (size_t r = 0; r < result.clusters.size(); ++r) {
        auto& region = result.clusters[r];
        cv::Mat foreground = result.foreground(region.crop).clone();
        for (size_t i = 0; i < result.guided.size(); ++i)
            if (region.crop.contains(centre(result.guided[i]))) {
                region.seedIndices.push_back(static_cast<int>(i));
                cv::circle(foreground, centre(result.guided[i]) - region.crop.tl(), cvRound(0.5 * result.guided[i][2]), cv::Scalar(255), cv::FILLED);
            }
        if (region.seedIndices.empty())
            continue;
        cv::Mat expanded;
        cv::dilate(foreground, expanded, cv::getStructuringElement(cv::MORPH_ELLIPSE, {7, 7}));
        region.markers = cv::Mat::zeros(foreground.size(), CV_32SC1);
        region.markers.setTo(1, expanded == 0);
        region.markers.setTo(1, table(region.crop) == 0);
        int label = 2;
        for (int i : region.seedIndices)
            cv::circle(region.markers, centre(result.guided[i]) - region.crop.tl(), 3, cv::Scalar(label++), cv::FILLED);
        region.labels = region.markers.clone();
        cv::Mat smooth;
        cv::GaussianBlur(image(region.crop), smooth, {3, 3}, 0.8);
        cv::watershed(smooth, region.labels);
        for (auto& candidate : result.proposals) {
            if (!candidate.accepted || candidate.region != static_cast<int>(r + 1))
                continue;
            const auto index = std::find(region.seedIndices.begin(), region.seedIndices.end(), candidate.guidedIndex);
            if (index == region.seedIndices.end())
                throw std::runtime_error("Missing watershed marker for accepted proposal");
            cv::Mat basin, supported;
            cv::compare(region.labels, static_cast<int>(index - region.seedIndices.begin()) + 2, basin, cv::CMP_EQ);
            cv::bitwise_and(basin, foreground, supported);
            candidate.watershedAreaRatio = cv::countNonZero(supported) / (CV_PI * candidate.circle[2] * candidate.circle[2]);
            candidate.watershedAccepted = candidate.watershedAreaRatio >= p.watershedMinimumAreaRatio && candidate.watershedAreaRatio <= p.watershedMaximumAreaRatio;
            candidate.watershedReason = candidate.watershedAccepted ? "accepted" : "basin_area";
            keep[candidate.guidedIndex] = candidate.watershedAccepted;
        }
    }
    for (size_t i = 0; i < result.guided.size(); ++i)
        if (keep[i])
            result.watershed.push_back(result.guided[i]);
}

} // namespace

SegmentationGuidedResult detectSegmentationGuided(const cv::Mat& image, const SegmentationHoughOptions& p) {
    SegmentationGuidedResult result;
    if (image.empty())
        return result;
    if (image.type() != CV_8UC3 || p.clothTileSize < 1 || p.angularSamples < 12 || p.tableInsetKernel < 1 || p.tableInsetKernel % 2 != 1 || p.closingKernel < 1 || p.closingKernel % 2 != 1 ||
        p.clothMinimumDistance > p.clothMaximumDistance || p.minimumClothSamples < 1)
        throw std::invalid_argument("Invalid segmentation-guided detection image/options");
    cv::Mat table = cv::Mat::zeros(image.size(), CV_8UC1);
    cv::fillConvexPoly(table, getTableCorners(), cv::Scalar(255));
    // These operations deliberately precede all Hough calls.
    result.foreground = foreground(image, table, p);
    result.clusters = clusterWindows(result.foreground, table, p);
    result.routingMask = cv::Mat::zeros(image.size(), CV_8UC1);
    for (const auto& region : result.clusters) {
        const int margin = cvRound(p.routingOverlapRadii * region.expectedRadius);
        const cv::Rect interior(region.core.x + margin, region.core.y + margin, std::max(0, region.core.width - 2 * margin), std::max(0, region.core.height - 2 * margin));
        if (!interior.empty())
            result.routingMask(interior).setTo(255);
    }
    const auto standard = runDetection(image, HoughParams{});
    result.baseline = standard.filteredCircles;
    for (const auto& c : result.baseline)
        if (!result.routingMask.at<uchar>(centre(c)))
            result.standardOutside.push_back(c);
    result.guided = result.standardOutside;
    for (size_t r = 0; r < result.clusters.size(); ++r) {
        auto& region = result.clusters[r];
        HoughParams local;
        local.param2 = p.localAccumulatorThreshold;
        local.minDist = p.localMinCentreDistance;
        local.minRadius = region.minRadius = std::max(p.absoluteMinLocalRadius, static_cast<int>(std::floor(p.localMinRadiusRatio * region.expectedRadius)));
        local.maxRadius = region.maxRadius = std::min(p.absoluteMaxLocalRadius, static_cast<int>(std::ceil(p.localMaxRadiusRatio * region.expectedRadius)));
        cv::Mat grey, smooth, edges, gx, gy;
        cv::cvtColor(image(region.crop), grey, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(grey, smooth, {5, 5}, 1.5);
        cv::Canny(smooth, edges, local.param1 / 2, local.param1);
        cv::Sobel(smooth, gx, CV_32F, 1, 0, 3);
        cv::Sobel(smooth, gy, CV_32F, 0, 1, 3);
        const auto circles = detectBalls(image(region.crop), local);
        for (const auto& c : circles) {
            GuidedProposal candidate;
            candidate.region = static_cast<int>(r + 1);
            candidate.circle = {c[0] + region.crop.x, c[1] + region.crop.y, c[2]};
            edgeEvidence(candidate, region, edges, gx, gy, result.foreground, p);
            if (!table.at<uchar>(centre(candidate.circle)))
                candidate.reason = "outside_table";
            else if (candidate.edgeSupport < p.minimumEdgeSupport)
                candidate.reason = "edge_support";
            else if (candidate.arcDegrees < p.minimumArcDegrees)
                candidate.reason = "visible_arc";
            else if (candidate.foregroundSupport < p.minimumForegroundSupport)
                candidate.reason = "foreground_support";
            else if (filterPocketDetections({candidate.circle}).empty())
                candidate.reason = "pocket_filter";
            result.proposals.push_back(candidate);
        }
    }
    std::vector<size_t> ranking(result.proposals.size());
    std::iota(ranking.begin(), ranking.end(), 0);
    std::stable_sort(ranking.begin(), ranking.end(), [&](size_t a, size_t b) {
        return result.proposals[a].quality > result.proposals[b].quality;
    });
    for (size_t i : ranking) {
        auto& candidate = result.proposals[i];
        if (!candidate.reason.empty())
            continue;
        if (duplicate(candidate.circle, result.guided, p))
            candidate.reason = "duplicate";
        else {
            candidate.reason = "accepted";
            candidate.accepted = true;
            candidate.guidedIndex = static_cast<int>(result.guided.size());
            result.guided.push_back(candidate.circle);
        }
    }
    // Fall back only when a cluster's local search accepts nothing, without looking at labels.
    for (size_t r = 0; r < result.clusters.size(); ++r) {
        auto& region = result.clusters[r];
        const bool found = std::any_of(result.proposals.begin(), result.proposals.end(), [&](const auto& c) {
            return c.accepted && c.region == static_cast<int>(r + 1);
        });
        if (!found) {
            region.usedEmptySearchFallback = true;
            for (const auto& c : result.baseline)
                if (region.core.contains(centre(c)) && !duplicate(c, result.guided, p))
                    result.guided.push_back(c);
        }
    }
    watershedCheck(image, table, result, p);
    return result;
}


} // namespace guided_hough_watershed

struct HistoricalType {
    std::string name="Unknown";
    double light=0,dark=0,colour=0;
    cv::Mat colourMask;
};

// EXPERIMENT: original per-pixel HLS thresholding (before K-means). All counts
// are restricted to the circular ROI. This evaluates the direct threshold idea.
HistoricalType experimentManualHLS(const BallROI& ball) {
    HistoricalType out;
    cv::Mat hls;
    cv::cvtColor(ball.image,hls,cv::COLOR_BGR2HLS);
    out.colourMask=cv::Mat::zeros(ball.image.size(),CV_8U);
    const double total=cv::countNonZero(ball.mask);
    if (!total) return out;
    for(int y=0;y<hls.rows;++y) for(int x=0;x<hls.cols;++x) {
        if (!ball.mask.at<uchar>(y,x)) continue;
        const auto pixel=hls.at<cv::Vec3b>(y,x);
        const int L=pixel[1],S=pixel[2];
        if (L>=175 && S<=85) out.light++;
        else if (L<=65) out.dark++;
        else if (S>=85) {out.colour++;out.colourMask.at<uchar>(y,x)=255;}
    }
    out.light=100*out.light/total;
    out.dark=100*out.dark/total;
    out.colour=100*out.colour/total;
    if(out.light>=85) out.name="White";
    else if(out.colour+out.dark>=70) out.name="Solid";
    else if(out.light>0 && out.colour+out.dark>0) out.name="Striped";
    return out;
}

// EXPERIMENT: K=2..6 type classification. The final method uses K=2. Higher K
// values are exploratory and must not be claimed to improve recognition without
// evaluating their output against ground truth.
HistoricalType experimentKMeansType(const BallROI& ball,int k,cv::Mat* segmented=nullptr) {
    HistoricalType result;
    auto seg=segmentBallKMeans(ball,k);
    if(segmented) *segmented=seg.segmentedImage;
    auto classification=classifyBallType(seg);
    result.name=classification.type;
    result.light=classification.lightPercent;
    result.dark=classification.darkPercent;
    result.colour=classification.colouredPercent;
    result.colourMask=cv::Mat::zeros(ball.image.size(),CV_8U);
    for (const auto& region:seg.regions) {
        const auto h=region.hsv;
        if (!(h[2]>=170 && h[1]<=165) && !(h[2]<=90))
            cv::bitwise_or(result.colourMask,region.mask,result.colourMask);
    }
    return result;
}

// EXPERIMENT: Connected-component spatial features from K-means light mask.
// This reconstructs the reported component count, largest region and boundary
// coverage features, without pretending the old rule thresholds were recovered.
struct HistoricalCCA {
    int lightCount=0,nonLightCount=0;
    double largestLightPct=0,largestNonLightPct=0,boundaryLightPct=0;
    cv::Mat lightMask,nonLightMask;
};
HistoricalCCA experimentCCA(const BallROI& ball) {
    HistoricalCCA out;
    const auto seg=segmentBallKMeans(ball,2);
    out.lightMask=cv::Mat::zeros(ball.image.size(),CV_8U);
    for(const auto& r:seg.regions)
        if(r.hsv[2]>=170 && r.hsv[1]<=165)
            cv::bitwise_or(out.lightMask,r.mask,out.lightMask);
    cv::subtract(ball.mask,out.lightMask,out.nonLightMask);
    const double area=std::max(1,cv::countNonZero(ball.mask));
    auto measure=[&](const cv::Mat& mask,int& count,double& largest){
        cv::Mat labels, stats,centroids;
        const int n=cv::connectedComponentsWithStats(mask,labels,stats,centroids,8);
        count=0;
        int big=0;
        for(int i=1;i<n;++i) {
            const int pixels=stats.at<int>(i,cv::CC_STAT_AREA);
            if (100.0*pixels/area >= 2.0) {++count;big=std::max(big,pixels);}
        }
        largest=100.0*big/area;
    };
    measure(out.lightMask,out.lightCount,out.largestLightPct);
    measure(out.nonLightMask,out.nonLightCount,out.largestNonLightPct);
    cv::Mat eroded,boundary,coverage;
    cv::erode(ball.mask,eroded,cv::getStructuringElement(cv::MORPH_ELLIPSE,{5,5}));
    cv::subtract(ball.mask,eroded,boundary);
    cv::bitwise_and(boundary,out.lightMask,coverage);
    out.boundaryLightPct=100.0*cv::countNonZero(coverage)/std::max(1,cv::countNonZero(boundary));
    return out;
}

// EXPERIMENT: CCA feature rules derived from the ORIGINAL branch's
// ccaSearchConfig.h / ccaGridSearch.cpp. This compact comparison keeps the
// selected K=2 baseline (the historical CCA search used a tuned K=6 base).
// Thus it is source-informed, but NOT an exact reproduction of that grid search.
std::string experimentCCAType(const BallROI& ball,const HistoricalCCA& spatial) {
    const auto base=classifyBallType(segmentBallKMeans(ball,2));
    const double nonLight=base.colouredPercent+base.darkPercent;
    if (base.lightPercent>=85 && spatial.largestLightPct>=80 &&
        spatial.boundaryLightPct>=75 && spatial.nonLightCount==0) return "White";
    if (spatial.largestLightPct>=20 && spatial.boundaryLightPct>=20 &&
        nonLight>=10) return "Striped";
    if (nonLight>=70 && spatial.boundaryLightPct<=12 &&
        spatial.nonLightCount==1) return "Solid";
    return base.type;
}

// EXPERIMENT: classify a ball using only a K-means cluster representative HSV
// hue (plus V/S for black). Kept as a baseline against histogram matching.
int experimentKMeansHueIdentity(const BallROI& ball) {
    const auto seg=segmentBallKMeans(ball,2);
    const auto type=classifyBallType(seg).type;
    if(type=="White") return 0;
    int colour=-1;
    for(const auto& r:seg.regions) {
        const int h=r.hsv[0],s=r.hsv[1],v=r.hsv[2];
        if(v>=170 && s<=165) continue;
        if(s<=60 && v<=90) {colour=8;break;}
        if(h<6 || h>=170) colour=v<150?7:3;
        else if(h<15) colour=5;
        else if(h<35) colour=1;
        else if(h>=45 && h<90) colour=6;
        else if(h>=100 && h<130) colour=2;
        else if(h>=130 && h<165) colour=4;
        if(colour>=0) break;
    }
    if(colour<0 || type=="Unknown") return -1;
    return type=="Striped" ? colour+8 : colour;
}

// EXPERIMENT: HSV histogram BACKPROJECTION. Unlike histogram comparison, this
// maps each pixel to its likelihood under a selected reference histogram.
// The reconstruction is a visual diagnostic only, not the final colour score.
cv::Mat experimentHistogramBackprojection(const BallROI& ball,
                                           const BallColourModel& model,int colour) {
    if(colour<1 || colour>8) return {};
    const auto features=featureImage(ball,model.space,model.options);
    const int channels[]={0,1,2};
    const float hue[]={0,180}, other[]={0,256};
    const float* ranges[]={hue,other,other};
    cv::Mat hist=model.histograms[colour-1].clone();
    if(hist.empty()) return {};
    // Histogram backprojection accepts multi-dimensional floating histograms.
    // Feature images are 8-bit and histogram bins/ranges match training.
    cv::Mat projection;
    cv::calcBackProject(&features,1,channels,hist,projection,ranges,255.0);
    projection.setTo(0,ball.mask==0);
    cv::Mat view;
    cv::normalize(projection,view,0,255,cv::NORM_MINMAX,CV_8U);
    return view;
}

// EXPERIMENT: expected apparent ball radius from table homography. Correctly
// transforms points, not original-image radii. For an approximate comparison,
// a 21.5 mm table-plane radius is projected back to the photograph locally.
// This tests the physically motivated size rule rather than the raw-largest rule.
int experimentExpectedRadiusCue(const std::vector<Candidate>& candidates) {
    int count=0,selected=-1;
    double best=-std::numeric_limits<double>::infinity();
    const cv::Mat inverse=getTableTransformation().inv();
    for(size_t i=0;i<candidates.size();++i) {
        const auto& c=candidates[i];
        if(c.baselineType.type!="White") continue;
        ++count;
        const cv::Point2f p=c.tableCentre;
        std::vector<cv::Point2f> table={p,{p.x+21.5f,p.y},{p.x,p.y+21.5f}};
        std::vector<cv::Point2f> source;
        cv::perspectiveTransform(table,source,inverse);
        const double expected=(cv::norm(source[0]-source[1])+cv::norm(source[0]-source[2]))/2.0;
        if(expected<1) continue;
        const double score=-std::abs(c.circle[2]-expected);
        if(score>best) {best=score;selected=int(i);}
    }
    return count>=2?selected:-1;
}

// Same metrics and spatial matching for every reconstructed experiment; a type
// or colour-only experiment cannot be confused with full 16-class accuracy.
void runHistoricalExperiments() {
    const auto root=std::filesystem::path(configApplicationOutputDirectory);
    const auto training=cv::imread(configMediaDirectory+configTuningImage);
    if(training.empty()) throw std::runtime_error("Missing AllBalls training image");
    const auto model=trainSubmissionModel(training);
    auto recognition=checkedCSV(root/"historical_recognition_experiments.csv");
    recognition<<"experiment,image,matchedDetections,correctTypes,correctIdentities,notes\n";
    auto detection=checkedCSV(root/"historical_detection_experiments.csv");
    detection<<"experiment,image,TP,FP,FN,F1,meanError\n";
    auto ccaCSV=checkedCSV(root/"ExperimentCCAResults"/"spatial_features.csv");
    ccaCSV<<"image,candidate,lightComponents,nonLightComponents,largestLightPct,largestNonLightPct,boundaryLightPct\n";
    for(const auto& filename:configValidationImages) {
        cv::Mat image=cv::imread(configMediaDirectory+filename);
        if(image.empty()) throw std::runtime_error("Missing image: "+filename);
        const auto stem=std::filesystem::path(filename).stem().string();
        const auto base=runDetection(image,HoughParams{});
        const auto candidates=measureCandidates(image,base.filteredCircles,model);
        const auto truth=loadLabelledGroundTruth(configGroundTruthPath,filename);
        std::vector<cv::Point2f> trueCentres;
        for(const auto& b:truth) trueCentres.push_back(b.centre);
        auto evaluateD=[&](const std::string& name,const std::vector<cv::Vec3f>& circles) {
            const auto projected=transformCentres(circles);
            const auto e=evaluateDetections(projected,trueCentres,configMatchTolerance);
            detection<<name<<','<<filename<<','<<e.TP<<','<<e.FP<<','<<e.FN<<','<<e.f1<<','<<e.meanError<<'\n';
            saveImage(root/("Experiment"+name+"Results")/stem/"circles.png",drawDetectedBalls(image,circles));
        };
        // Detection reconstruction visual diagnostics and metrics.
        const auto water=experimentWatershed(image,base.filteredCircles);
        evaluateD("Watershed",water.circles);
        saveImage(root/"ExperimentWatershedResults"/stem/"foreground.png",water.foreground);
        saveImage(root/"ExperimentWatershedResults"/stem/"distance.png",water.distance);
        saveImage(root/"ExperimentWatershedResults"/stem/"basins.png",water.markerView);
        // Original combined experiment; export both stages to isolate what
        // watershed rejects, rather than conflating their detection totals.
        const auto guided=guided_hough_watershed::detectSegmentationGuided(
            image, guided_hough_watershed::SegmentationHoughOptions{});
        evaluateD("GuidedHoughPreWatershed",guided.guided);
        evaluateD("GuidedHoughWatershed",guided.watershed);
        const auto combinedDir=root/"ExperimentGuidedHoughWatershedResults"/stem;
        saveImage(combinedDir/"foreground.png",guided.foreground);
        saveImage(combinedDir/"routing_mask.png",guided.routingMask);
        cv::Mat clusters=image.clone();
        cv::Mat basins=cv::Mat::zeros(image.size(),CV_8UC3);
        for (const auto& region:guided.clusters) {
            cv::rectangle(clusters,region.core,cv::Scalar(0,220,255),2);
            cv::rectangle(clusters,region.crop,cv::Scalar(255,120,0),1);
            if (!region.labels.empty()) {
                cv::Mat labels8, coloured;
                region.labels.convertTo(labels8,CV_8U,13);
                cv::applyColorMap(labels8,coloured,cv::COLORMAP_TURBO);
                coloured.setTo(cv::Scalar(0,0,0),region.labels<=1);
                coloured.copyTo(basins(region.crop));
            }
        }
        saveImage(combinedDir/"clusters.png",clusters);
        saveImage(combinedDir/"basins.png",basins);
        auto proposals=checkedCSV(combinedDir/"proposals.csv");
        proposals<<"region,x,y,radius,edgeSupport,arcDegrees,foregroundSupport,acceptedBeforeWatershed,watershedAreaRatio,acceptedAfterWatershed,reason,watershedReason\n";
        for (const auto& proposal:guided.proposals)
            proposals<<proposal.region<<','<<proposal.circle[0]<<','<<proposal.circle[1]<<','<<proposal.circle[2]<<','
                     <<proposal.edgeSupport<<','<<proposal.arcDegrees<<','<<proposal.foregroundSupport<<','
                     <<proposal.accepted<<','<<proposal.watershedAreaRatio<<','<<proposal.watershedAccepted<<','
                     <<proposal.reason<<','<<proposal.watershedReason<<'\n';
        // Recognition experiments use the same detections for fair comparison.
        const auto matching=evaluateDetections(transformCentres(base.filteredCircles),trueCentres,configMatchTolerance);
        std::map<std::string,std::vector<int>> ids;
        std::map<std::string,std::vector<std::string>> types;
        for(int k=2;k<=6;++k) types["KMeansK"+std::to_string(k)]={};
        for(size_t i=0;i<candidates.size();++i) {
            const auto& roi=candidates[i].roi;
            const auto hls=experimentManualHLS(roi);
            types["ManualHLS"].push_back(hls.name);
            saveImage(root/"ExperimentManualHLSResults"/stem/("ball_"+std::to_string(i+1)+"_colourMask.png"),hls.colourMask);
            for(int k=2;k<=6;++k) {
                cv::Mat segmented;
                const auto result=experimentKMeansType(roi,k,&segmented);
                const std::string name="KMeansK"+std::to_string(k);
                types[name].push_back(result.name);
                if(!segmented.empty())
                    saveImage(root/("Experiment"+name+"Results")/stem/("ball_"+std::to_string(i+1)+"_segmented.png"),segmented);
            }
            const auto cca=experimentCCA(roi);
            types["CCA"].push_back(experimentCCAType(roi,cca));
            ccaCSV<<filename<<','<<i+1<<','<<cca.lightCount<<','<<cca.nonLightCount<<','<<cca.largestLightPct<<','<<cca.largestNonLightPct<<','<<cca.boundaryLightPct<<'\n';
            saveImage(root/"ExperimentCCAResults"/stem/("ball_"+std::to_string(i+1)+"_light.png"),cca.lightMask);
            ids["KMeansHue"].push_back(experimentKMeansHueIdentity(roi));
            // Histogram diagnostic is exported below, independently of hue classification.
            const int bestColour=candidates[i].colours.bestColour;
            if(bestColour>=1) {
                const auto view=experimentHistogramBackprojection(roi,model,bestColour);
                if(!view.empty()) saveImage(root/"ExperimentHistogramBackprojectionResults"/stem/("ball_"+std::to_string(i+1)+"_bestColour.png"),view);
            }
        }
        // Produce per-method annotated images for type-only experiments.
        // Text labels use the predicted type rather than a guessed ball number.
        for(const auto& [name,pred]:types) {
            cv::Mat annotated=image.clone();
            for(size_t i=0;i<pred.size();++i) {
                cv::Point centre(cvRound(base.filteredCircles[i][0]),cvRound(base.filteredCircles[i][1]));
                cv::circle(annotated,centre,cvRound(base.filteredCircles[i][2]),cv::Scalar(0,220,255),2);
                cv::putText(annotated,pred[i],centre+cv::Point(-20,-25),
                            cv::FONT_HERSHEY_SIMPLEX,0.40,cv::Scalar(255,255,255),2,cv::LINE_AA);
            }
            saveImage(root/("Experiment"+name+"Results")/stem/"predicted_types.png",annotated);
        }
        // Show representative hue-identity predictions with exactly the same
        // detected positions as the HSV final approach.
        {
            const auto& hueNumbers=ids["KMeansHue"];
            std::vector<std::string> hueTypes;
            for(int n:hueNumbers) hueTypes.push_back(submissionType(n));
            saveFinalResults(image,root/"ExperimentKMeansHueResults"/stem,
                             base.filteredCircles,transformCentres(base.filteredCircles),
                             hueTypes,hueNumbers);
        }
        // Type classification evaluations (do not present as exact identities).
        for(const auto& [name,pred]:types) {
            int correct=0;
            for(size_t i=0;i<pred.size();++i) {
                int j=matching.predictionToGroundTruth[i];
                if(j>=0 && pred[i]==submissionType(truth[j].number)) ++correct;
            }
            recognition<<name<<','<<filename<<','<<matching.TP<<','<<correct<<",,type-only\n";
        }
        // Hue baseline evaluates complete identities independently.
        for(const auto& [name,pred]:ids) {
            int correctType=0,correctIdentity=0;
            for(size_t i=0;i<pred.size();++i) {
                int j=matching.predictionToGroundTruth[i];
                if(j<0) continue;
                correctType+=submissionType(pred[i])==submissionType(truth[j].number);
                correctIdentity+=pred[i]==truth[j].number;
            }
            recognition<<name<<','<<filename<<','<<matching.TP<<','<<correctType<<','<<correctIdentity<<",independent hue-based identity\n";
        }
        // Cue radius alternative: same final evidence but change only reservation.
        const auto padded=paddedCandidates(image,candidates,model);
        const auto expected=uniquePrediction("ExpectedRadiusCue",padded,experimentExpectedRadiusCue(candidates));
        const auto dir=root/"ExperimentExpectedRadiusCueResults"/stem;
        std::vector<std::string> predictedTypes;
        for(int n:expected.numbers) predictedTypes.push_back(submissionType(n));
        saveFinalResults(image,dir,expected.circles,expected.centres,predictedTypes,expected.numbers);
        int exact=0,correctType=0;
        for(size_t i=0;i<expected.numbers.size();++i) {
            const int j=matching.predictionToGroundTruth[i];
            if(j>=0) {exact+=expected.numbers[i]==truth[j].number;
                      correctType+=submissionType(expected.numbers[i])==submissionType(truth[j].number);}
        }
        recognition<<"ExpectedRadiusCue,"<<filename<<','<<matching.TP<<','<<correctType<<','<<exact<<",cue heuristic only\n";
    }
}

} // namespace pool_submission

void MyApplication() {
    const char* setting = std::getenv("POOL_BALL_MODE");
    const std::string mode = setting ? setting : "selected";
    if (mode == "grid-search")
        pool_submission::runGridSearch();
    else if (mode == "selected")
        pool_submission::runSubmission();
    else if (mode == "experiments")
        {
        pool_submission::runGridSearch(); // First experiment: 480-configuration search
        pool_submission::runHistoricalExperiments();
    }
    else
        throw std::invalid_argument("POOL_BALL_MODE must be selected, grid-search or experiments");
}

#ifdef POOL_BALL_STANDALONE
int main() {
    try {
        MyApplication();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
#endif
