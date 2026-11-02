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
inline const std::string configGroundTruthPath = "data/ground_truth/BallLocations.csv";
inline const std::string configOutputDirectory = "output/";
inline const std::string configTuningImage = "AllBalls.jpeg";

inline const std::string configRawOutputDirectory = configOutputDirectory + "detect_circles/";
inline const std::string configFilteredOutputDirectory = configOutputDirectory + "detect_circles_filtered/";
inline const std::string configValidationOutputDirectory = configOutputDirectory + "validation/";
inline const std::string configSanityOutputDirectory = configOutputDirectory + "sanity_check/";
inline const std::string configExtractionOutputDirectory = configOutputDirectory + "recognition/AllBalls/";

inline const std::string configApplicationOutputDirectory = configOutputDirectory + "myApplication/";

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

using namespace cv;
using namespace std;

// ===============================
// MAIN PIPELINE HELPER FUNCTIONS
// ===============================
vector<Point> getTableCorners() {
    return configTableCorners;
}

/*
Calculates the perspective transformation that maps the four image corners onto the rectangular table.
*/
Mat getTableTransformation() {
    // Original image corners
    vector<Point2f> source;

    for (const Point& corner : getTableCorners()) {
        source.emplace_back(corner.x, corner.y);
    }

    // Transformed table coordinates
    vector<Point2f> destination = {Point2f(0, 0), Point2f(configTableWidth - 1, 0), Point2f(configTableWidth - 1, configTableHeight - 1), Point2f(0, configTableHeight - 1)};

    return getPerspectiveTransform(source, destination);
}

/*
Converts detected circle centres from original image coordinates into table coordinates.
*/
vector<Point2f> transformCentres(const vector<Vec3f>& circles) {
    Mat matrix = getTableTransformation();

    vector<Point2f> centres;

    for (const Vec3f& c : circles) {
        centres.push_back(Point2f(c[0], c[1]));
    }

    vector<Point2f> transformed;

    if (!centres.empty()) {
        perspectiveTransform(centres, transformed, matrix);
    }

    return transformed;
}

/*
Creates an overhead view of the image by applying the table perspective transformation.
*/
Mat transformImage(const Mat& image) {
    Mat matrix = getTableTransformation();
    Mat transformed;
    warpPerspective(image, transformed, matrix, Size(configTableWidth, configTableHeight));
    return transformed;
}

/*
Removes detected circles whose transformed centres are closer than pocketRadius to any of the six pocket locations.
*/
vector<Vec3f> filterPocketDetections(const vector<Vec3f>& circles) {

    // Table dimensions after perspective transformation
    const float width = configTableWidth;
    const float height = configTableHeight;

    // Approximate pocket locations in transformed coordinates
    vector<Point2f> pockets = {{0, 0}, {width / 2, 0}, {width - 1, 0}, {0, height - 1}, {width / 2, height - 1}, {width - 1, height - 1}};

    const float pocketRadius = configPocketRadius;

    vector<Point2f> transformedCentres = transformCentres(circles);

    vector<Vec3f> filtered;

    for (size_t i = 0; i < circles.size(); ++i) {
        bool nearPocket = false;

        for (const Point2f& pocket : pockets) {
            if (norm(transformedCentres[i] - pocket) < pocketRadius) {
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
vector<Vec3f> detectBalls(const Mat& image, const HoughParams& params) {
    Mat grey_image;
    cvtColor(image, grey_image, COLOR_BGR2GRAY);

    Mat blurred_image;
    GaussianBlur(grey_image, blurred_image, Size(5, 5), 1.5);

    vector<Vec3f> circles;

    HoughCircles(blurred_image, circles, HOUGH_GRADIENT, params.dp, params.minDist, params.param1, params.param2, params.minRadius, params.maxRadius);

    return circles;
}

/*
Keeps the pixels inside the table polygon and makes the rest of the image black.
*/
Mat maskTable(const Mat& image) {
    if (image.empty())
        return {};

    Mat mask = Mat::zeros(image.size(), CV_8UC1);
    fillConvexPoly(mask, getTableCorners(), Scalar(255));

    Mat maskedImage;
    bitwise_and(image, image, maskedImage, mask);

    return maskedImage;
}

/*
Runs table masking, circle detection and pocket checking while retaining the intermediate results.
*/
DetectionResult runDetection(const Mat& image, const HoughParams& params) {
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
BallROI extractBall(const Mat& image, const Vec3f& detectedCircle) {
    BallROI result;

    if (image.empty())
        return result;

    // Extract circle information from Hough detection
    int x = cvRound(detectedCircle[0]);
    int y = cvRound(detectedCircle[1]);
    int radius = max(1, cvRound(detectedCircle[2]));

    // Define square region around the ball
    Rect roi(x - radius, y - radius, 2 * radius + 1, 2 * radius + 1);

    // Ensure ROI stays inside the image
    roi &= Rect(0, 0, image.cols, image.rows);

    if (roi.empty())
        return result;

    // Extract the original colour pixels
    result.image = image(roi).clone();
    result.mask = Mat::zeros(result.image.size(), CV_8UC1);

    Point localCentre(x - roi.x, y - roi.y);

    circle(result.mask, localCentre, radius, Scalar(255), FILLED);

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
    vector<Point> pixels;

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
    Mat samples(total, 3, CV_32F);

    for (int i = 0; i < total; ++i) {

        Vec3b colour = ball.image.at<Vec3b>(pixels[i]);

        samples.at<float>(i, 0) = colour[0];
        samples.at<float>(i, 1) = colour[1];
        samples.at<float>(i, 2) = colour[2];
    }

    // K-means clustering
    Mat labels;
    Mat centres;

    TermCriteria criteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 50, 0.2);

    // Fixed seed for reproducibility
    uint64 previousState = theRNG().state;
    theRNG().state = 12345;
    kmeans(samples, k, labels, criteria, 3, KMEANS_PP_CENTERS, centres);
    theRNG().state = previousState;

    // Create cluster masks
    vector<Mat> masks(k);

    for (int j = 0; j < k; ++j) {
        masks[j] = Mat::zeros(ball.image.size(), CV_8UC1);
    }

    result.segmentedImage = Mat::zeros(ball.image.size(), CV_8UC3);

    // Reconstruct segmented image
    for (int i = 0; i < total; ++i) {

        int cluster = labels.at<int>(i, 0);
        Point p = pixels[i];
        Vec3b colour(saturate_cast<uchar>(centres.at<float>(cluster, 0)), saturate_cast<uchar>(centres.at<float>(cluster, 1)), saturate_cast<uchar>(centres.at<float>(cluster, 2)));
        result.segmentedImage.at<Vec3b>(p) = colour;
        masks[cluster].at<uchar>(p) = 255;
    }

    // Extract cluster information
    for (int j = 0; j < k; ++j) {

        int area = countNonZero(masks[j]);

        if (area == 0)
            continue;

        KMeansRegion region;

        region.mask = masks[j];
        region.percentage = 100.0 * area / total;
        region.colour = Vec3b(saturate_cast<uchar>(centres.at<float>(j, 0)), saturate_cast<uchar>(centres.at<float>(j, 1)), saturate_cast<uchar>(centres.at<float>(j, 2)));

        Mat bgr(1, 1, CV_8UC3);
        bgr.at<Vec3b>(0, 0) = region.colour;

        Mat hsv;
        cvtColor(bgr, hsv, COLOR_BGR2HSV);

        region.hsv = hsv.at<Vec3b>(0, 0);
        result.regions.push_back(region);
    }

    // Sort largest regions first
    sort(result.regions.begin(), result.regions.end(), [](const KMeansRegion& a, const KMeansRegion& b) {
        return a.percentage > b.percentage;
    });

    // compare cluster colours
    if (result.k == 2 && result.regions.size() == 2) {
        Vec3b colour1 = result.regions[0].colour;
        Vec3b colour2 = result.regions[1].colour;

        Mat bgr(1, 2, CV_8UC3);
        bgr.at<Vec3b>(0, 0) = colour1;
        bgr.at<Vec3b>(0, 1) = colour2;

        Mat hsv;
        cvtColor(bgr, hsv, COLOR_BGR2HSV);

        Vec3b hsv1 = hsv.at<Vec3b>(0, 0);
        Vec3b hsv2 = hsv.at<Vec3b>(0, 1);

        int H1 = hsv1[0];
        int S1 = hsv1[1];
        int V1 = hsv1[2];

        int H2 = hsv2[0];
        int S2 = hsv2[1];
        int V2 = hsv2[2];

        result.saturationDifference = abs(S1 - S2);
        result.brightnessDifference = abs(V1 - V2);

        if (S1 >= 60 && S2 >= 60 && V1 >= 70 && V2 >= 70) {
            int difference = abs(H1 - H2);
            result.hueDifference = min(difference, 180 - difference);
        }

        Vec3d difference(double(colour1[0] - colour2[0]), double(colour1[1] - colour2[1]), double(colour1[2] - colour2[2]));
        result.bgrDistance = norm(difference);
    }

    return result;
}

// ======================================
// K-MEANS WHITE-COVERAGE CLASSIFICATION
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
        Mat bgr(1, 1, CV_8UC3);
        bgr.at<Vec3b>(0, 0) = region.colour;

        Mat hsv;
        cvtColor(bgr, hsv, COLOR_BGR2HSV);

        Vec3b pixel = hsv.at<Vec3b>(0, 0);

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
        for (int h = 0; h < hCount; ++h)
            for (int s = 0; s < sCount; ++s)
                for (int t = 0; t < tCount; ++t)
                    for (int offset = -1; offset <= 1; ++offset) {
                        const int hh = axis == 0 ? (h + offset + hCount) % hCount : h;
                        const int ss = axis == 1 ? std::clamp(s + offset, 0, sCount - 1) : s;
                        const int tt = axis == 2 ? std::clamp(t + offset, 0, tCount - 1) : t;
                        target[at(h, s, t)] += source[at(hh, ss, tt)] * (offset == 0 ? 0.5f : 0.25f);
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
// Training labels/crop geometry were recorded from the labelled AllBalls image.
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

// The existing homography maps the cushions to pixel endpoints
// (0,0)..(1433,682). At this resolution, one pixel is approximately one mm.
constexpr double ballDiameterMm = 43.0;

string finalBallLabel(int number) {
    if (number == 0)
        return "Cue";

    if (number >= 1 && number <= 15)
        return to_string(number);

    return "Unknown";
}

/*
Draws a readable predicted ball label near a centre while keeping the text inside the image.
*/
void drawFinalLabel(Mat& image, Point centre, int number, Point offset = {}) {
    const string label = finalBallLabel(number);
    const double scale = number < 0 ? 0.36 : 0.45;
    int baseline = 0;

    const Size text = getTextSize(label, FONT_HERSHEY_SIMPLEX, scale, 1, &baseline);
    Point anchor(centre.x - text.width / 2 + offset.x, centre.y + text.height / 2 + offset.y);
    
    anchor.x = clamp(anchor.x, 2, max(2, image.cols - text.width - 2));
    anchor.y = clamp(anchor.y, text.height + 2, max(text.height + 2, image.rows - baseline - 2));
    
    putText(image, label, anchor, FONT_HERSHEY_SIMPLEX, scale, Scalar(0, 0, 0), 3, LINE_AA);
    putText(image, label, anchor, FONT_HERSHEY_SIMPLEX, scale, Scalar(255, 255, 255), 1, LINE_AA);
}

/*
Returns a drawing colour for each predicted ball identity.
*/
Scalar diagramBallColour(int number) {
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
Mat drawTableDiagram(const vector<Point2f>& centres, const vector<int>& numbers) {
    Mat diagram(configTableHeight, configTableWidth, CV_8UC3, Scalar(65, 115, 45));
    rectangle(diagram, {0, 0}, {diagram.cols - 1, diagram.rows - 1}, Scalar(160, 190, 160), 1);
    // Nominal physical size; never reuse a source-image Hough radius here.
    
    const double radius = ballDiameterMm / 2;
    constexpr int shift = 8;
    constexpr int fixedScale = 1 << shift;

    for (size_t i = 0; i < centres.size(); ++i) {
        const Point fixedCentre(cvRound(centres[i].x * fixedScale), cvRound(centres[i].y * fixedScale));
        const int fixedRadius = cvRound(radius * fixedScale);
        const Scalar colour = diagramBallColour(numbers[i]);

        circle(diagram, fixedCentre, fixedRadius, colour, FILLED, LINE_AA, shift);
        if (numbers[i] >= 9 && numbers[i] <= 15) {

            // Clip a coloured middle band to a white disk, preserving its round outline.
            const int half = cvCeil(radius) + 2;
            const Point origin(cvFloor(centres[i].x) - half, cvFloor(centres[i].y) - half);
            const Rect patch = Rect(origin.x, origin.y, 2 * half + 1, 2 * half + 1) & Rect(0, 0, diagram.cols, diagram.rows);
            
            if (!patch.empty()) {
                Mat mask = Mat::zeros(patch.size(), CV_8UC1);
                const Point localFixed = fixedCentre - Point(patch.x * fixedScale, patch.y * fixedScale);
                circle(mask, localFixed, fixedRadius, Scalar(255), FILLED, LINE_AA, shift);
                Mat ball(patch.size(), CV_8UC3, Scalar(245, 245, 245));
                const int y = cvRound(centres[i].y) - patch.y;
                rectangle(ball, {0, y - 10}, {patch.width - 1, y + 10}, colour, FILLED);
                ball.copyTo(diagram(patch), mask);
            }
        }
        circle(diagram, fixedCentre, fixedRadius, Scalar(230, 230, 230), 1, LINE_AA, shift);
    }
    // Draw labels last so neighbouring ball symbols cannot paint over them.
    for (size_t i = 0; i < centres.size(); ++i)
        drawFinalLabel(diagram, {cvRound(centres[i].x), cvRound(centres[i].y)}, numbers[i]);
    return diagram;
}

/*
Saves the numbered original image, overhead views, table diagram and a CSV of predicted ball identities and coordinates.
*/
void saveFinalResults(const Mat& image, const filesystem::path& directory, const vector<Vec3f>& circles, const vector<Point2f>& tableCentres, const vector<string>& types, const vector<int>& numbers) {
    if (circles.size() != tableCentres.size() || circles.size() != types.size() || circles.size() != numbers.size())
        throw runtime_error("Final result vectors have inconsistent candidate counts");

    filesystem::create_directories(directory);
    Mat original = image.clone();
    const Mat rectified = transformImage(image);
    Mat overhead = rectified.clone();

    for (size_t i = 0; i < circles.size(); ++i) {
        const Point sourceCentre(cvRound(circles[i][0]), cvRound(circles[i][1]));
        circle(original, sourceCentre, cvRound(circles[i][2]), Scalar(0, 220, 255), 2, LINE_AA);
        drawFinalLabel(original, sourceCentre, numbers[i]);

        const Point tableCentre(cvRound(tableCentres[i].x), cvRound(tableCentres[i].y));
        drawMarker(overhead, tableCentre, Scalar(0, 220, 255), MARKER_CROSS, 9, 1, LINE_AA);
        drawFinalLabel(overhead, tableCentre, numbers[i], {0, -12});
    }
    saveImage(directory / "original_numbered.png", original);
    saveImage(directory / "table_rectified.png", rectified);
    saveImage(directory / "table_numbered.png", overhead);
    saveImage(directory / "table_diagram.png", drawTableDiagram(tableCentres, numbers));

    ofstream csv(directory / "balls.csv");
    if (!csv.is_open())
        throw runtime_error("Could not create final results CSV: " + directory.string());

    csv << "candidate,predictedType,predictedNumber,originalX,originalY,originalRadiusPixels,tableX,tableY\n" << fixed << setprecision(6);
    for (size_t i = 0; i < circles.size(); ++i)
        csv << i + 1 << ',' << types[i] << ',' << numbers[i] << ',' << circles[i][0] << ',' << circles[i][1] << ',' << circles[i][2] << ',' << tableCentres[i].x << ',' << tableCentres[i].y << '\n';
    
    finishOutputFile(csv);
    cout << "Final images and table coordinates saved to: " << directory << endl;
}

/*
Returns a copy of the image with each detected circle boundary and centre marked.
*/
Mat drawDetectedBalls(const Mat& image, const vector<Vec3f>& circles) {
    Mat result = image.clone();

    for (const Vec3f& detected : circles) {
        Point center(cvRound(detected[0]), cvRound(detected[1]));

        int radius = cvRound(detected[2]);

        // Draw circle boundary
        circle(result, center, radius, Scalar(0, 0, 255), 2);

        // Draw centre
        circle(result, center, 2, Scalar(255, 0, 0), -1);
    }

    return result;
}

// ================================ 
// EVALUATION AGAINST GROUND TRUTH
// ================================

/*
Reads the annotated ball positions for one image from a CSV file and optionally records their ball IDs.
*/
vector<Point2f> loadGroundTruth(const string& csvPath, const string& imageName, vector<int>* ballIds = nullptr) {
    vector<Point2f> points;
    if (ballIds)
        ballIds->clear();
    ifstream file(csvPath);

    if (!file.is_open()) {
        cerr << "Could not open: " << csvPath << endl;
        return points;
    }

    string line;

    while (getline(file, line)) {
        stringstream ss(line);
        string filename;

        getline(ss, filename, ',');

        if (filename != imageName)
            continue;

        vector<string> values;
        string value;

        while (getline(ss, value, ',')) {
            values.push_back(value);
        }

        // Each ball has a Row and Column
        for (size_t i = 0; i + 1 < values.size(); i += 2) {
            if (values[i].empty() || values[i + 1].empty())
                continue;

            float row = stof(values[i]);
            float col = stof(values[i + 1]);

            points.push_back(Point2f(col, row));
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
EvaluationResult evaluateDetections(const vector<Point2f>& predictions, const vector<Point2f>& groundTruth, double tolerance) {
    // List valid matches, trying nearby ground-truth balls first.
    vector<vector<pair<double, size_t>>> candidates(predictions.size());

    for (size_t p = 0; p < predictions.size(); ++p) {
        for (size_t g = 0; g < groundTruth.size(); ++g) {
            double distance = hypot(double(predictions[p].x) - groundTruth[g].x, double(predictions[p].y) - groundTruth[g].y);
            if (distance <= tolerance)
                candidates[p].emplace_back(distance, g);
        }
        sort(candidates[p].begin(), candidates[p].end());
    }

    // Reassign earlier matches when needed to maximize the number of
    // one-to-one matches. Each ground-truth ball counts at most once.
    vector<int> matchedPrediction(groundTruth.size(), -1);
    function<bool(size_t, vector<bool>&)> tryMatch = [&](size_t p, vector<bool>& visited) -> bool {
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
        vector<bool> visited(groundTruth.size(), false);
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
        const Point2f& prediction = predictions[matchedPrediction[g]];
        totalError += hypot(double(prediction.x) - groundTruth[g].x, double(prediction.y) - groundTruth[g].y);
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
vector<LabelledBall> loadLabelledGroundTruth(const string& csvPath, const string& imageName) {
    vector<LabelledBall> balls;
    ifstream file(csvPath);

    if (!file.is_open())
        return balls;

    string line;

    while (getline(file, line)) {
        stringstream ss(line);
        string filename;

        getline(ss, filename, ',');

        if (filename != imageName)
            continue;

        vector<string> values;
        string value;

        while (getline(ss, value, ','))
            values.push_back(value);

        // 15 numbered balls + white cue ball
        for (int i = 0; i < 16; ++i) {
            int index = 2 * i;

            if (index + 1 >= static_cast<int>(values.size()))
                break;

            if (values[index].empty() || values[index + 1].empty())
                continue;

            float row = stof(values[index]);
            float col = stof(values[index + 1]);

            LabelledBall ball;
            ball.number = (i == 15) ? 0 : i + 1;
            ball.centre = Point2f(col, row);

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

// =========================================== 
// HOUGH PARAMETER SEARCH
// Runs only with POOL_BALL_MODE=grid-search
// =========================================== 
const string IMAGE_PATH = configMediaDirectory + configTuningImage;
const string CSV_PATH = configGroundTruthPath;
const double MATCH_TOLERANCE = configMatchTolerance;

/*
Searches 480 Hough parameter combinations on AllBalls and exports the best detection settings.
*/
void runGridSearch() {
    Mat image = imread(IMAGE_PATH);

    if (image.empty()) {
        cerr << "Could not open AllBalls.jpeg" << endl;
        return;
    }

    Mat maskedImage = maskTable(image);

    vector<Point2f> groundTruth = loadGroundTruth(CSV_PATH, configTuningImage);

    if (groundTruth.size() != 16) {
        cerr << "Invalid ground truth: " << groundTruth.size() << " balls" << endl;
        return;
    }

    // DEFINE PARAMETER SEARCH SPACE
    vector<double> dpValues = {1.0, 1.5, 2.0};
    vector<double> minDistValues = {15, 18, 20, 22, 25};
    vector<double> param1Values = {75, 100, 125, 150};
    vector<double> param2Values = {15, 20, 25, 30, 35, 40, 45, 50};
    // 480 configurations

    std::filesystem::create_directories(configOutputDirectory);
    ofstream file(configOutputDirectory + "grid_search.csv");

    if (!file.is_open()) {
        cerr << "Could not create output/grid_search.csv" << endl;
        return;
    }

    file << "dp,minDist,param1,param2,"
         << "minRadius,maxRadius,detections,"
         << "TP,FP,FN,precision,recall,F1,meanError\n";

    double bestTP = -1;
    int bestFP = numeric_limits<int>::max();
    double bestError = numeric_limits<double>::infinity();

    HoughParams bestParams;
    EvaluationResult bestResult;

    int count = 0;
    const size_t totalConfigurations = dpValues.size() * minDistValues.size() * param1Values.size() * param2Values.size();

    // ================================= 
    // TEST EVERY PARAMETER COMBINATION
    // ================================= 

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
                    vector<Vec3f> circles = detectBalls(maskedImage, params);

                    // Transform detected centres
                    vector<Point2f> predictions = transformCentres(circles);

                    // Evaluate
                    EvaluationResult result = evaluateDetections(predictions, groundTruth, MATCH_TOLERANCE);

                    // Save results
                    file << dp << "," << minDist << "," << p1 << "," << p2 << "," << params.minRadius << "," << params.maxRadius << "," << circles.size() << "," << result.TP << "," << result.FP << "," << result.FN << "," << result.precision
                         << "," << result.recall << "," << result.f1 << "," << result.meanError << "\n";

                    count++;

                    // Prioritise recall, then precision
                    // then centre localisation accuracy.
                    double candidateError = result.TP > 0 ? result.meanError : numeric_limits<double>::infinity();

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

                    cout << "\rTested " << count << "/" << totalConfigurations << " configurations" << flush;
                }
            }
        }
    }

    finishOutputFile(file);

    // =========================== 
    // DISPLAY BEST CONFIGURATION
    // =========================== 

    cout << fixed << setprecision(4);

    cout << "\n\n===== GRID SEARCH COMPLETE =====\n";
    cout << "Configurations tested: " << count << endl;

    cout << "\nBEST PARAMETERS:\n";
    cout << "dp: " << bestParams.dp << endl;
    cout << "minDist: " << bestParams.minDist << endl;
    cout << "param1: " << bestParams.param1 << endl;
    cout << "param2: " << bestParams.param2 << endl;
    cout << "minRadius: " << bestParams.minRadius << endl;
    cout << "maxRadius: " << bestParams.maxRadius << endl;

    cout << "\nBEST PERFORMANCE:\n";
    cout << "TP: " << bestResult.TP << endl;
    cout << "FP: " << bestResult.FP << endl;
    cout << "FN: " << bestResult.FN << endl;
    cout << "Precision: " << bestResult.precision << endl;
    cout << "Recall: " << bestResult.recall << endl;
    cout << "F1: " << bestResult.f1 << endl;

    if (bestResult.TP > 0)
        cout << "Mean error: " << bestResult.meanError << " pixels" << endl;
    else
        cout << "Mean error: N/A" << endl;

    // =========================================
    // SAVE VISUALISATION OF BEST CONFIGURATION
    // =========================================
    vector<Vec3f> bestCircles = detectBalls(maskedImage, bestParams);
    Mat bestImage = drawDetectedBalls(maskedImage, bestCircles);

    saveImage(configOutputDirectory + "best_detection.jpeg", bestImage);

    cout << "\nSaved: output/grid_search.csv" << endl;
    cout << "Saved: output/best_detection.jpeg" << endl;
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
            const auto destination = directory / "final_results" / stem;
            std::vector<std::string> types;
            for (int n : p.numbers)
                types.push_back(submissionType(n));
            saveFinalResults(image, destination, p.circles, p.centres, types, p.numbers);
            for (size_t i = 0; i < p.circles.size(); ++i)
                predictions << filename << ',' << p.method << ',' << i + 1 << ',' << p.numbers[i] << ',' << p.centres[i].x << ',' << p.centres[i].y << '\n';
        }

        const auto stageDir = directory / "processing" / stem;
        saveImage(stageDir / "masked.png", detection.maskedImage);
        cv::Mat gray, blur;
        cv::cvtColor(detection.maskedImage, gray, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(gray, blur, {5, 5}, 1.5);

        saveImage(stageDir / "blurred.png", blur);
        saveImage(stageDir / "raw_circles.png", drawDetectedBalls(image, detection.rawCircles));
        saveImage(stageDir / "filtered_circles.png", drawDetectedBalls(image, circles));
        
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

    auto matrix = checkedCSV(directory / "identity_confusion.csv");
    matrix << "method,trueNumber,predictedNumber,count\n";
    for (const auto& [key, n] : confusion)
        matrix << std::get<0>(key) << ',' << std::get<1>(key) << ',' << std::get<2>(key) << ',' << n << '\n';
    
    const auto selected = totals.find("largest_raw_radius");
    if (selected != totals.end()) {
        const auto& c = selected->second;
        std::cout << "Table1--6: " << c.tp << " TP, " << c.fp << " FP, " << c.fn << " FN; " << c.correct << '/' << c.tp << " correct identities, " << 100 * fraction(c.correct, c.gt) << "% end-to-end\n";
    }
}
} 

void MyApplication() {
    const char* setting = std::getenv("POOL_BALL_MODE");
    const std::string mode = setting ? setting : "selected";
    if (mode == "grid-search")
        pool_submission::runGridSearch();
    else if (mode == "selected")
        pool_submission::runSubmission();
    else
        throw std::invalid_argument("POOL_BALL_MODE must be selected or grid-search");
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
