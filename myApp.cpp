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

/*
=========================================
            CONFIGURATIONS
=========================================
*/

// ========Constants========
inline const std::string MEDIA_DIR = "Media/";
inline const std::string GROUND_TRUTH_PATH = "data/ground_truth/BallLocationsNew.csv";
inline const std::string OUTPUT_DIR = "final_output/";

inline const std::string TUNNNING_IMG = "AllBalls.jpeg";
inline const std::vector<std::string> VAL_IMG = {"Table1.jpeg", "Table2.jpeg", "Table3.jpeg", "Table4.jpeg", "Table5.jpeg", "Table6.jpeg"};

inline constexpr int TABLE_WIDTH = 1434;
inline constexpr int TABLE_HEIGHT = 683;
inline constexpr double MATCH_TOLERANCE = 15.0;
inline constexpr float ESTIMATE_POCKET_RADIUS = 25.0f;
inline constexpr int K_MEAN_CLUSTERS = 2;

inline const std::vector<cv::Point> TABLE_CORNERS = {{337, 35}, {1570, 21}, {1888, 451}, {38, 472}};

inline constexpr double BALL_DIAMETER_MM = 43.0;

// ========Structs========
struct HoughParams {
    double dp = 1.5;
    double minDist = 18.0;
    double param1 = 75.0;
    double param2 = 35.0;
    int minRadius = 15;
    int maxRadius = 30;
};

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

struct LabelledBall {
    int number; // 1-15, 0 = cue ball
    cv::Point2f centre;
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

// ========Helper Functions
inline std::vector<std::string> getAllImages() {
    auto images = VAL_IMG;
    images.insert(images.begin(), TUNNNING_IMG);
    return images;
}

std::vector<cv::Point> getTableCorners() {
    return TABLE_CORNERS;
}

// computes the perspective transformation to warp the table corners to a rectangle
cv::Mat getTableTransformation() {
    std::vector<cv::Point2f> source; // table coners

    for (const cv::Point& corner : getTableCorners()) {
        source.emplace_back(corner.x, corner.y);
    }

    // Transformed table coordinates
    std::vector<cv::Point2f> destination = {cv::Point2f(0, 0), cv::Point2f(TABLE_WIDTH - 1, 0), cv::Point2f(TABLE_WIDTH - 1, TABLE_HEIGHT - 1), cv::Point2f(0, TABLE_HEIGHT - 1)};

    return cv::getPerspectiveTransform(source, destination);
}

// Converts detected circle centres from original image coordinates into table coordinates.
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
=========================================
            PREPROCESSING
=========================================
*/

// ========DETECTION========

// Removes detected circles whose transformed centres are closer than pocketRadius to any of the six pocket locations.
std::vector<cv::Vec3f> filterPocketDetections(const std::vector<cv::Vec3f>& circles) {

    // Table dimensions after perspective transformation
    const float width = TABLE_WIDTH;
    const float height = TABLE_HEIGHT;

    // Approximate pocket locations in transformed coordinates
    std::vector<cv::Point2f> pockets = {{0, 0}, {width / 2, 0}, {width - 1, 0}, {0, height - 1}, {width / 2, height - 1}, {width - 1, height - 1}};

    const float pocketRadius = ESTIMATE_POCKET_RADIUS;

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

// Keeps the pixels inside the table polygon and makes the rest of the image black.
cv::Mat maskTable(const cv::Mat& image) {
    if (image.empty())
        return {};

    cv::Mat mask = cv::Mat::zeros(image.size(), CV_8UC1);
    cv::fillConvexPoly(mask, getTableCorners(), cv::Scalar(255));

    cv::Mat maskedImage;
    cv::bitwise_and(image, image, maskedImage, mask);

    return maskedImage;
}

// ========RECOGNITION========

// Crops the image around a detected circle and creates a mask identifying the pixels inside that circle.
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

// ========VISUALISATION/RESULTS========

// Creates an overhead view of the image by applying the table perspective transformation.
cv::Mat transformImage(const cv::Mat& image) {
    cv::Mat matrix = getTableTransformation();
    cv::Mat transformed;
    cv::warpPerspective(image, transformed, matrix, cv::Size(TABLE_WIDTH, TABLE_HEIGHT), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    return transformed;
}

// Creates any missing parent directories and saves an image.
void saveImage(const std::filesystem::path& path, const cv::Mat& image) {
    if (!path.parent_path().empty())
        std::filesystem::create_directories(path.parent_path());

    if (!cv::imwrite(path.string(), image))
        throw std::runtime_error("Could not save image: " + path.string());
}

// Flushes and closes an output file, reporting an error if writing did not finish successfully.
void finishOutputFile(std::ofstream& file) {
    file.flush();
    file.close();

    if (!file)
        throw std::runtime_error("Could not finish writing output CSV");
}

// get the label of the ball based on its number (0 = cue ball, 1-15 = numbered balls)
std::string finalBallLabel(int number) {
    if (number == 0)
        return "Cue";

    if (number >= 1 && number <= 15)
        return std::to_string(number);

    return "Unknown";
}

// Sets a drawing colour for each predicted ball identity.
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
=========================================
            DETECTION
=========================================
*/

// Finds circle candidates using Hough detection after converting the image to greyscale and blurring it.
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
Runs the Ball Dection Pipeline:
    1. Masks the table from the image.
    2. Detects circles using Hough Transform.
    3. Filters out circles that are too close to the pockets.
    4. Visualizes the results (TODO_ADD).
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
=========================================
            RECOGNITION
=========================================
*/

/*
=========================================
            EVALUATION
=========================================
*/

/*
=========================================
            VISUALIZATION
=========================================
*/

// draws the detected circles on top of the original image, with a red outline and a blue centre dot.
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

// Draws a readable predicted ball label near a centre while keeping the text inside the image.
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

// Creates an overhead table diagram showing the predicted ball positions, colours and identity labels.
cv::Mat drawTableDiagram(const std::vector<cv::Point2f>& centres, const std::vector<int>& numbers) {
    cv::Mat diagram(TABLE_HEIGHT, TABLE_WIDTH, CV_8UC3, cv::Scalar(65, 115, 45));
    cv::rectangle(diagram, {0, 0}, {diagram.cols - 1, diagram.rows - 1}, cv::Scalar(160, 190, 160), 1);
    
    const double radius = BALL_DIAMETER_MM / 2;
    constexpr int shift = 8;
    constexpr int fixedScale = 1 << shift;

    for (size_t i = 0; i < centres.size(); ++i) {
        const cv::Point fixedCentre(cvRound(centres[i].x * fixedScale), cvRound(centres[i].y * fixedScale));
        const int fixedRadius = cvRound(radius * fixedScale);
        const cv::Scalar colour = diagramBallColour(numbers[i]);

        cv::circle(diagram, fixedCentre, fixedRadius, colour, cv::FILLED, cv::LINE_AA, shift);
        if (numbers[i] >= 9 && numbers[i] <= 15) {

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

    for (size_t i = 0; i < centres.size(); ++i)
        drawFinalLabel(diagram, {cvRound(centres[i].x), cvRound(centres[i].y)}, numbers[i]);
    return diagram;
}

/*
=========================================
            RESULTS
=========================================
*/

/*
Saves the final results to a directory:
    - original_numbered.png: the original image with detected circles and predicted labels drawn on top.
    - table_rectified.png: the rectified table image after perspective transformation.
    - table_numbered.png: the rectified table image with predicted labels drawn on top.
    - table_diagram.png: a diagram showing the predicted ball positions, colours and identity labels.
    - balls.csv: a CSV file containing the predicted ball information (candidate number, predicted type, predicted number, original coordinates, radius, and transformed coordinates).
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
=========================================
            EXPERIMENTS
=========================================
*/


/*
=========================================
            MAIN APPLICATION
=========================================
*/
void MyApplication() {
    const HoughParams params;

    for (const std::string& filename : getAllImages()) {
        const cv::Mat image = cv::imread(MEDIA_DIR + filename);

        if (image.empty())
            throw std::runtime_error("Could not load: " + filename);

        const std::filesystem::path directory =
            std::filesystem::path(OUTPUT_DIR) /
            std::filesystem::path(filename).stem();

        // 1. Run detection.
        const DetectionResult detection = runDetection(image, params);

        // 2. Save the mask and detection images.
        saveImage(directory / "masked.png", detection.maskedImage);

        saveImage(directory / "raw_circles.png",
                  drawDetectedBalls(image, detection.rawCircles));

        saveImage(directory / "filtered_circles.png",
                  drawDetectedBalls(image, detection.filteredCircles));

        // 3. Extract each remaining ball from the original colour image.
        for (size_t i = 0; i < detection.filteredCircles.size(); ++i) {
            const BallROI ball =
                extractBall(image, detection.filteredCircles[i]);

            const std::string name = "ball_" + std::to_string(i + 1);

            saveImage(directory / "ROI" / (name + ".png"), ball.image);
            saveImage(directory / "ROI" / (name + "_mask.png"), ball.mask);
        }

        std::cout << filename
                  << ": " << detection.rawCircles.size() << " raw circles, "
                  << detection.filteredCircles.size() << " after filtering\n";
    }
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