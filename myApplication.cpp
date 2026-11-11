#include "myApplication.h"

/*
=========================================
            CONFIGURATIONS
=========================================
*/

// ========Constants========
inline const std::string MEDIA_DIR = "Media/";
inline const std::string GROUND_TRUTH_PATH = "data/ground_truth/BallLocationsNew.csv";
inline const std::string OUTPUT_DIR = "final_output/";

inline const std::string DETECTION_DIR = OUTPUT_DIR + "detection/";
inline const std::string RECOGNITION_DIR = OUTPUT_DIR + "recognition/";
inline const std::string FINAL_RES_DIR = OUTPUT_DIR + "final/";
inline const std::string EXPERIMENT_DIR = OUTPUT_DIR + "experiments/";

inline const std::string TUNNNING_IMG = "AllBalls.jpeg";
inline const std::vector<std::string> VAL_IMG = {"Table1.jpeg", "Table2.jpeg", "Table3.jpeg", "Table4.jpeg", "Table5.jpeg", "Table6.jpeg"};

inline constexpr int TABLE_WIDTH = 1434;
inline constexpr int TABLE_HEIGHT = 683;
inline constexpr double MATCH_TOLERANCE = 15.0;
inline constexpr float ESTIMATE_POCKET_RADIUS = 25.0f;
inline constexpr int K_MEAN_CLUSTERS = 2;

inline const std::vector<cv::Point> TABLE_CORNERS = {{337, 35}, {1570, 21}, {1888, 451}, {38, 472}};

inline constexpr double BALL_DIAMETER_MM = 43.0;

inline const int LIGHT_MIN_V = 170;
inline const int LIGHT_MAX_S = 165;
inline const int DARK_MAX_V = 90;

inline const double WHITE_BALL_MIN = 85.0;
inline const double SOLID_MIN = 70.0;

inline constexpr double TYPE_WIDTH = 0.10;

inline const cv::Scalar DETECTION_COLOUR{255, 0, 255}; // magenta
inline const cv::Scalar ERROR_COLOUR{0, 255, 255}; // yellow

// ========IMAGE LIST========
std::vector<std::string> getAllImages() {
    auto images = VAL_IMG;
    images.insert(images.begin(), TUNNNING_IMG);
    return images;
}

/*
=========================================
            GEOMETRY
=========================================
*/

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

// Creates an overhead view of the image by applying the table perspective transformation.
cv::Mat transformImage(const cv::Mat& image) {
    cv::Mat matrix = getTableTransformation();
    cv::Mat transformed;
    cv::warpPerspective(image, transformed, matrix, cv::Size(TABLE_WIDTH, TABLE_HEIGHT), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    return transformed;
}

/*
=========================================
            FILE INPUT/OUTPUT
=========================================
*/

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

// Reads the annotated positions and identities for one image, representing the cue ball with identity zero.
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

// Reads the annotated ball positions for one image
std::vector<cv::Point2f> loadGroundTruth(const std::string& csvPath, const std::string& imageName, std::vector<int>* ballIds) {

    const auto balls = loadLabelledGroundTruth(csvPath, imageName);

    std::vector<cv::Point2f> points;

    if (ballIds)
        ballIds->clear();

    for (const LabelledBall& ball : balls) {
        points.push_back(ball.centre);

        if (ballIds)
            ballIds->push_back(ball.number);
    }

    return points;
}

/*
=========================================
            VISUALISATION
=========================================
*/

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

// combine multiple images into one
cv::Mat combineImages(const std::vector<ImagePanel>& panels, int columns) {
    if (columns <= 0)
        throw std::invalid_argument("Image grid columns must be positive");

    constexpr int padding = 8;
    constexpr int labelHeight = 24;
    constexpr double fontScale = 0.4;
    int width = 0;
    int height = 0;
    int count = 0;

    for (const auto& panel : panels) {
        if (panel.image.empty())
            continue;
        if (panel.image.depth() != CV_8U || (panel.image.channels() != 1 && panel.image.channels() != 3 && panel.image.channels() != 4))
            throw std::invalid_argument("Image panels must be 8-bit grayscale, BGR, or BGRA");

        int baseline = 0;
        const auto labelSize = cv::getTextSize(panel.label, cv::FONT_HERSHEY_SIMPLEX, fontScale, 1, &baseline);
        width = std::max({width, panel.image.cols, labelSize.width});
        height = std::max(height, panel.image.rows);
        ++count;
    }

    if (count == 0)
        return {};

    columns = std::min(columns, count);
    const int rows = (count + columns - 1) / columns;
    const int cellWidth = width + 2 * padding;
    const int cellHeight = height + labelHeight + 2 * padding;
    cv::Mat combined(rows * cellHeight, columns * cellWidth, CV_8UC3, cv::Scalar(32, 32, 32));

    int index = 0;
    for (const auto& panel : panels) {
        if (panel.image.empty())
            continue;

        cv::Mat colour;
        if (panel.image.channels() == 1)
            cv::cvtColor(panel.image, colour, cv::COLOR_GRAY2BGR);
        else if (panel.image.channels() == 4)
            cv::cvtColor(panel.image, colour, cv::COLOR_BGRA2BGR);
        else
            colour = panel.image;

        const int x = (index % columns) * cellWidth + padding;
        const int y = (index / columns) * cellHeight + padding;
        cv::putText(combined, panel.label, cv::Point(x, y + 15), cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        colour.copyTo(combined(cv::Rect(x + (width - colour.cols) / 2, y + labelHeight + (height - colour.rows) / 2, colour.cols, colour.rows)));
        ++index;
    }
    return combined;
}

// draws the detected circles on top of the original image, with a red outline and a blue centre dot.
cv::Mat drawDetectedBalls(const cv::Mat& image, const std::vector<cv::Vec3f>& circles) {
    cv::Mat result = image.clone();

    for (const cv::Vec3f& detected : circles) {
        cv::Point center(cvRound(detected[0]), cvRound(detected[1]));

        int radius = cvRound(detected[2]);

        // Draw circle boundary
        cv::circle(result, center, radius, DETECTION_COLOUR, 2, cv::LINE_AA);

        // Draw centre
        cv::circle(result, center, 2, cv::Scalar(255, 0, 0), -1);
    }

    return result;
}

// Draws a readable predicted ball label near a centre while keeping the text inside the image.
void drawFinalLabel(cv::Mat& image, cv::Point centre, int number, cv::Point offset) {
    const std::string label = finalBallLabel(number);
    const double scale = number < 0 ? 0.45 : 1.0;
    int baseline = 0;

    const cv::Size text = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, scale, 3, &baseline);
    cv::Point anchor(centre.x - text.width / 2 + offset.x, centre.y + text.height / 2 + offset.y);

    anchor.x = std::clamp(anchor.x, 2, std::max(2, image.cols - text.width - 2));
    anchor.y = std::clamp(anchor.y, text.height + 2, std::max(text.height + 2, image.rows - baseline - 2));

    cv::putText(image, label, anchor, cv::FONT_HERSHEY_SIMPLEX, scale, cv::Scalar(0, 0, 0), 5, cv::LINE_AA);
    cv::putText(image, label, anchor, cv::FONT_HERSHEY_SIMPLEX, scale, cv::Scalar(255, 255, 255), 3, cv::LINE_AA);
}

// Creates an overhead table diagram showing the predicted ball positions, colours and identity labels.
cv::Mat drawTableDiagram(const std::vector<cv::Point2f>& centres, const std::vector<int>& numbers, const std::vector<cv::Scalar>& outlineColours = {}) {
    cv::Mat diagram(TABLE_HEIGHT, TABLE_WIDTH, CV_8UC3, cv::Scalar(65, 115, 45));
    cv::rectangle(diagram, {0, 0}, {diagram.cols - 1, diagram.rows - 1}, cv::Scalar(160, 190, 160), 1);

    const double radius = (BALL_DIAMETER_MM / 2.0) * 1.4;
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
                const int stripeHalfWidth = cvRound(radius * 0.45);
                cv::rectangle(ball, {0, y - stripeHalfWidth}, {patch.width - 1, y + stripeHalfWidth}, colour, cv::FILLED);
                ball.copyTo(diagram(patch), mask);
            }
        }
        const cv::Scalar outline = outlineColours.empty() ? DETECTION_COLOUR : outlineColours[i];
        cv::circle(diagram, fixedCentre, fixedRadius, outline, 4, cv::LINE_AA, shift);
    }

    for (size_t i = 0; i < centres.size(); ++i)
        drawFinalLabel(diagram, {cvRound(centres[i].x), cvRound(centres[i].y)}, numbers[i]);
    return diagram;
}

// draw ball predicted ball type onto the table
cv::Mat drawBallTypes(const cv::Mat& image, const std::vector<cv::Vec3f>& circles, const std::vector<BallTypeResult>& types) {

    if (circles.size() != types.size())
        throw std::runtime_error("Circles and types have different counts");

    cv::Mat result = drawDetectedBalls(image, circles);

    for (size_t i = 0; i < circles.size(); ++i) {
        const cv::Point centre(cvRound(circles[i][0]), cvRound(circles[i][1]));

        const std::string label = types[i].type;

        const cv::Point position = centre + cv::Point(-20, -cvRound(circles[i][2]) - 5);

        cv::putText(result, label, position, cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
        cv::putText(result, label, position, cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    }

    return result;
}

// draw ball predicted colours type onto the table
cv::Mat drawBallColours(const cv::Mat& image, const std::vector<cv::Vec3f>& circles, const std::vector<BallColourScores>& colours) {

    if (circles.size() != colours.size())
        throw std::runtime_error("Colours and circles have different counts");

    cv::Mat result = drawDetectedBalls(image, circles);

    for (size_t i = 0; i < circles.size(); ++i) {
        const std::string label = std::to_string(i + 1) + ": " + bodyColourName(colours[i].bestColour);

        const cv::Point position(cvRound(circles[i][0]) - 20, cvRound(circles[i][1] - circles[i][2]) - 5);

        cv::putText(result, label, position, cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
        cv::putText(result, label, position, cv::FONT_HERSHEY_SIMPLEX,0.45, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    }

    return result;
}

cv::Mat drawBodyPixelsGrid(
    const std::vector<BallROI>& balls,
    const std::vector<BallColourScores>& colours) {

    if (balls.size() != colours.size())
        throw std::runtime_error("Balls and colours have different counts");

    std::vector<ImagePanel> panels;

    for (size_t i = 0; i < balls.size(); ++i) {
        cv::Mat body = cv::Mat::zeros(balls[i].image.size(), balls[i].image.type());
        balls[i].image.copyTo(body, colours[i].bodyMask);
        const std::string label = "Ball " + std::to_string(i + 1) + ": " + bodyColourName(colours[i].bestColour);
        panels.push_back({label, body});
    }

    return combineImages(panels, 4);
}

/*
=========================================
            DETECTION
=========================================
*/

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
    4. Run evaluation
    5. Save the results.
*/
DetectionResult runDetection(const cv::Mat& image, const HoughParams& params, const std::filesystem::path directory, const std::string& filename) {
    DetectionResult result;

    if (image.empty())
        return result;

    result.maskedImage = maskTable(image);

    result.rawCircles = detectBalls(result.maskedImage, params);
    result.filteredCircles = filterPocketDetections(result.rawCircles);

    const auto predictions = transformCentres(result.filteredCircles);
    const auto groundTruth = loadGroundTruth(GROUND_TRUTH_PATH, filename);

    result.evaluation = evaluateDetections(predictions, groundTruth, MATCH_TOLERANCE);
    saveDetectionEvaluation(directory, result.evaluation);

    // save detection results
    saveImage(directory / "masked.png", result.maskedImage);
    saveImage(directory / "raw_circles.png", drawDetectedBalls(result.maskedImage, result.rawCircles));
    saveImage(directory / "filtered_circles.png", drawDetectedBalls(result.maskedImage, result.filteredCircles));

    return result;
}

/*
=========================================
            EXTRACTION
=========================================
*/

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

// Run extraction pipeline to extract detected ball and saving the results
std::vector<BallROI> runExtraction(const cv::Mat& image, const std::vector<cv::Vec3f>& circles, const std::filesystem::path& directory) {

    std::vector<BallROI> balls;

    std::vector<ImagePanel> cropPanels;
    std::vector<ImagePanel> maskPanels;

    for (size_t i = 0; i < circles.size(); ++i) {
        BallROI ball = extractBall(image, circles[i]);

        const std::string label = "Ball " + std::to_string(i + 1);
        cropPanels.push_back({label, ball.image});
        maskPanels.push_back({label, ball.mask});

        balls.push_back(ball);
    }

    const cv::Mat crops = combineImages(cropPanels, 4);
    const cv::Mat masks = combineImages(maskPanels, 4);

    if (!crops.empty())
        saveImage(directory / "ROI" / "ball_crops.png", crops);

    if (!masks.empty())
        saveImage(directory / "ROI" / "ball_masks.png", masks);

    return balls;
}

/*
=========================================
            RECOGNITION 
=========================================
*/

/*
======================
      BALL TYPE 
======================
*/

std::string identityType(int number) {
    if (number < 0)
        return "Unknown";

    if (number == 0)
        return "White";

    return number <= 8 ? "Solid" : "Striped";
}

// =============K MEANS CLUSTERING=============

/*
Apply K Means clustering
    1. collect the pixels inside the ball mask
    2. build the input matrix (each row has one pixel's three colour vals, BGR)
    3. Run K means
    4. create the segmented image 
    5. describe each cluster
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

    return result;
}

std::vector<KMeansResult> runKMeans(const std::vector<BallROI>& balls, int k, const std::filesystem::path& directory) {

    std::vector<KMeansResult> results;
    std::vector<ImagePanel> panels;

    for (size_t i = 0; i < balls.size(); ++i) {
        results.push_back(segmentBallKMeans(balls[i], k));

        panels.push_back({"Ball " + std::to_string(i + 1), results.back().segmentedImage});
    }

    const cv::Mat grid = combineImages(panels, 4);

    if (!grid.empty())
        saveImage(directory / "kmeans_grid.png", grid);

    return results;
}

// =============CLASSIFY TYPE=============

// Estimates whether a ball is white, solid or striped from the proportions of light, dark and coloured clusters.
BallTypeResult classifyBallType(const KMeansResult& segmentation) {
    BallTypeResult result;

    if (segmentation.regions.empty())
        return result;

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

// Estimates the ball type
TypeRecognitionResult runTypeRecognition(const cv::Mat& image, const std::vector<BallROI>& balls, 
                        const std::vector<cv::Vec3f>& circles, const std::filesystem::path& directory, 
                        const std::string& filename, const EvaluationContext& context) {

    std::vector<BallTypeResult> results;
    std::filesystem::create_directories(directory);

    std::ofstream csv(directory / "ball_types.csv");
    if (!csv.is_open())
        throw std::runtime_error("Could not create ball types CSV");

    csv << "candidate,type,lightPercent,colouredPercent,darkPercent\n";

    const auto segmentations = runKMeans(balls, K_MEAN_CLUSTERS, directory);

    for (size_t i = 0; i < balls.size(); ++i) {
        const BallTypeResult type = classifyBallType(segmentations[i]);

        results.push_back(type);

        csv << i + 1 << ',' << type.type << ',' << type.lightPercent << ',' << type.colouredPercent << ',' << type.darkPercent << '\n';
    }
    saveImage(directory / "table_types.png", drawBallTypes(image, circles, results));

    finishOutputFile(csv);
    return evaluateBallTypes(results, context);
}

/*
======================
      BALL COLOUR 
======================
*/

const char* bodyColourName(int colour) {
    static const char* names[] = {"unknown", "yellow", "blue", "red", "purple", "orange", "green", "maroon", "black"};
    return names[colour >= 1 && colour <= 8 ? colour : 0];
}

// =============COLOUR HISTOGRAMS=============

// Converts a ball crop to HSV and sets the hue of nearly neutral pixels to zero.
cv::Mat makeColourFeatures(const BallROI& ball, const BallHistogramOptions& p) {
    cv::Mat features;
    cv::cvtColor(ball.image, features, cv::COLOR_BGR2HSV);

    for (int y = 0; y < features.rows; ++y) {
        for (int x = 0; x < features.cols; ++x) {
            const auto bgr = ball.image.at<cv::Vec3b>(y, x);
            const int chroma = int(std::max({bgr[0], bgr[1], bgr[2]})) - std::min({bgr[0], bgr[1], bgr[2]});
            if (chroma < p.neutralChroma) {
                features.at<cv::Vec3b>(y, x)[0] = 0;
            }
        }
    }
    return features;
}

// Smooths neighbouring histogram bins, wrapping hue at its endpoints, and normalises the total bin weight to one.
cv::Mat smoothHistogram(const cv::Mat& histogram, const BallHistogramOptions& p) {
    cv::Mat current = histogram.clone();
    const int hCount = p.hueBins, sCount = p.saturationBins, tCount = p.toneBins;
    auto at = [=](int h, int s, int t) {
        return (h * sCount + s) * tCount + t;
    };

    // Smooth H, S, and V. Hue wraps; saturation and brightness clamp at endpoints.
    for (int axis = 0; axis < 3; ++axis) {
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

// Selects ball body pixels by shrinking the circular mask slightly and removing bright, low-saturation pixels.
cv::Mat histogramBodyMask(const BallROI& ball, const BallHistogramOptions& p) {
    if (ball.image.empty() || ball.image.type() != CV_8UC3 || ball.mask.type() != CV_8UC1 || ball.image.size() != ball.mask.size())
        throw std::invalid_argument("Invalid ball ROI");
    cv::Mat mask, hsv, light;
    cv::erode(ball.mask, mask, cv::getStructuringElement(cv::MORPH_ELLIPSE, {p.erosionKernel, p.erosionKernel}), {-1, -1}, 1, cv::BORDER_CONSTANT, 0);
    cv::cvtColor(ball.image, hsv, cv::COLOR_BGR2HSV);
    cv::inRange(hsv, cv::Scalar(0, 0, p.lightMinV), cv::Scalar(179, p.lightMaxS, 255), light);
    mask.setTo(0, light);
    return mask;
}

// Builds a smoothed, normalised colour histogram from the pixels selected by the supplied mask.
cv::Mat buildColourHistogram(const cv::Mat& features, const cv::Mat& mask, const BallHistogramOptions& p) {
    const int channels[] = {0, 1, 2}, bins[] = {p.hueBins, p.saturationBins, p.toneBins};
    const float hue[] = {0, 180}, other[] = {0, 256};
    const float* ranges[] = {hue, other, other};
    cv::Mat histogram;
    cv::calcHist(&features, 1, channels, mask, histogram, 3, bins, ranges);
    return smoothHistogram(histogram, p);
}

/*
Learn Reference Histograms
    1. Select the body pixels
    2. Builds their histograms
    3. Combines examples belonging to the same colour (e.g. the red full (3) and striped ball (11))
*/
BallColourModel buildReferenceHistograms(const std::vector<LabelledColourSample>& samples, const BallHistogramOptions& options) {
    BallColourModel model{};
    model.options = options;
    for (const auto& sample : samples) {
        if (sample.number < 0 || sample.number > 15)
            throw std::invalid_argument("Invalid training identity");

        if (sample.number == 0)
            continue;

        const auto mask = histogramBodyMask(sample.ball, options);

        if (cv::countNonZero(mask) < options.minimumBodyPixels)
            throw std::runtime_error("Insufficient body pixels for training ball " + std::to_string(sample.number));

        const int colour = (sample.number > 8 ? sample.number - 8 : sample.number) - 1;

        const auto features = makeColourFeatures(sample.ball, options);
        const auto histogram = buildColourHistogram(features, mask, options);

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

// Compares new ball with all 8 reference histograms
BallColourScores scoreBallColours(const BallROI& ball, const BallColourModel& model) {
    BallColourScores result;
    result.bodyMask = histogramBodyMask(ball, model.options);
    result.pixels = cv::countNonZero(result.bodyMask);

    if (result.pixels < model.options.minimumBodyPixels)
        return result;

    const auto features = makeColourFeatures(ball, model.options);
    const auto query = buildColourHistogram(features, result.bodyMask, model.options);

    for (size_t i = 0; i < model.histograms.size(); ++i) {
        const auto& histogram = model.histograms[i];
        if (histogram.empty() || histogram.type() != CV_32F || histogram.total() != query.total())
            throw std::invalid_argument("Invalid trained colour histogram");

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

// Use the reference image to build the reference histogram model
BallColourModel buildColourReferenceModel(const cv::Mat& referenceImage) {
    if (referenceImage.empty())
        throw std::runtime_error("Could not load the reference image");

    const auto truth = loadLabelledGroundTruth(GROUND_TRUTH_PATH, TUNNNING_IMG);

    if (truth.empty())
        throw std::runtime_error("No annotations for the reference image");

    const auto rawCircles = detectBalls(maskTable(referenceImage), HoughParams{});
    const auto circles = filterPocketDetections(rawCircles);

    std::vector<cv::Point2f> trueCentres;
    for (const LabelledBall& ball : truth)
        trueCentres.push_back(ball.centre);

    const EvaluationResult matching = evaluateDetections( transformCentres(circles), trueCentres, MATCH_TOLERANCE);

    std::vector<LabelledColourSample> samples;
    for (size_t g = 0; g < truth.size(); ++g) {
        if (truth[g].number == 0)
            continue; // Exclude the cue ball.

        const int p = matching.groundTruthToPrediction[g];

        if (p < 0)
            throw std::runtime_error(
                "Missing reference detection for ball " +
                std::to_string(truth[g].number));

        samples.push_back({
            truth[g].number,
            extractBall(referenceImage, circles[p])
        });
    }

    return buildReferenceHistograms(samples);
}

// =============CLASSIFY COLOUR=============

// Estimates whether colour of a ball
ColourRecognitionResult runColourRecognition(const cv::Mat& image, const std::vector<BallROI>& balls, 
                                                const std::vector<cv::Vec3f>& circles, const BallColourModel& model, 
                                                const std::filesystem::path& directory, const std::string& filename, 
                                                const EvaluationContext& context) {

    std::vector<BallColourScores> results;
    std::filesystem::create_directories(directory);

    std::ofstream csv(directory / "ball_colours.csv");
    if (!csv.is_open()) {
        throw std::runtime_error("Could not create ball colours CSV");
    }

    csv << "candidate,colour,similarity,margin,bodyPixels\n";

    for (size_t i = 0; i < balls.size(); ++i) {
        const BallColourScores scores = scoreBallColours(balls[i], model);

        const double bestSimilarity = scores.bestColour >= 1 ? scores.similarity[scores.bestColour - 1] : 0.0;

        csv << i + 1 << ',' << bodyColourName(scores.bestColour) << ',' << bestSimilarity << ',' << scores.margin << ',' << scores.pixels << '\n';

        results.push_back(scores);
    }

    finishOutputFile(csv);

    saveImage(directory / "table_colours.png", drawBallColours(image, circles, results));

    const cv::Mat grid = drawBodyPixelsGrid(balls, results);
    if (!grid.empty())
        saveImage(directory / "body_pixels_grid.png", grid);

    const ColourRecognitionResult evaluation = evaluateBallColours(results, context);

    std::ofstream evaluationCSV(directory / "colour_evaluation.csv");
    if (!evaluationCSV.is_open())
        throw std::runtime_error("Could not create colour evaluation CSV");

    evaluationCSV
        << "matched,correct,groundTruth,colourAccuracy,endToEndAccuracy\n"
        << evaluation.matched << ','
        << evaluation.correct << ','
        << evaluation.groundTruthCount << ','
        << evaluation.accuracy << ','
        << evaluation.endToEndAccuracy << '\n';

    finishOutputFile(evaluationCSV);
    return evaluation;
}

/*
======================
    DUP CUE BALLS
======================
*/

// Recomputes recognition evidence using an expanded crop while preserving the original detected circle and centre.
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

// When at least two candidates are classified as white, selects the one with the largest original detected radius.
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

// Restricts the selected candidate to the cue identity and prevents every other candidate from receiving that identity.
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

/*
======================
    BALL IDENTITY
======================
*/

/*
Maximise total assignment weight (Hungarian method)
    1. Takes a score matrix (detections are rows, possible identities are columns.)
    2. Assigns each detection to one column, using each column at most once.
    3. Rearranges earlier assignments when that improves the overall result.
    4. Returns the assignment with the highest total score
*/
std::vector<int> maximumAssignment(const std::vector<std::vector<double>>& weights) {
    const int n = int(weights.size());

    if (!n)
        return {};

    const int m = int(weights[0].size());

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

/*
Combines white coverage and colour similarities into scores for identities 0–15
    - Converts white coverage into support scores for cue, solid, and striped.
    - Combines type support with histogram colour similarities to score all 16 identities.
    - Records the highest-scoring identity for that candidate.
*/
void identityEvidence(Candidate& c) {
    const double light = std::clamp(c.baselineType.lightPercent / 100, 0.0, 1.0);
    const auto compatibility = [](double distance) {
        return std::exp(-0.5 * std::pow(distance / TYPE_WIDTH, 2));
    };
    
    const bool usable = c.colours.bestColour >= 1 || c.baselineType.type == "White";

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
        c.eligible[n] = usable && c.identityScore[n] > 1e-9;
}

// Assigns identities to candidates with the highest total score while allowing each identity once and any number of Unknowns.
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
======================
    FULL PIPELINE
======================
*/

IdentityRecognitionResult runIdentityRecognition( const cv::Mat& image, const std::vector<BallROI>& balls,
                            const std::vector<cv::Vec3f>& circles, const std::vector<BallTypeResult>& types,
                            const std::vector<BallColourScores>& colours, const BallColourModel& model,
                            const std::filesystem::path& directory, const std::string& filename, 
                            const EvaluationContext& context, bool printFinalResults) {

    if (balls.size() != circles.size() || types.size() != circles.size() || colours.size() != circles.size()) {
        throw std::runtime_error("Identity inputs have different counts");
    }

    // Combine the evidence already calculated for each detection.
    const auto centres = transformCentres(circles);
    std::vector<Candidate> original;

    for (size_t i = 0; i < circles.size(); ++i) {
        Candidate candidate;
        candidate.circle = circles[i];
        candidate.tableCentre = centres[i];
        candidate.roi = balls[i];
        candidate.baselineType = types[i];
        candidate.colours = colours[i];

        identityEvidence(candidate);
        original.push_back(candidate);
    }

    // Reanalyse originally white candidates with a larger ROI.
    auto candidates = original;

    for (size_t i = 0; i < candidates.size(); ++i) {
        if (original[i].baselineType.type == "White") {
            candidates[i] = rawWhitePadding(image, original[i], model, 1);
        }
    }

    // Resolve duplicate-white ambiguity using the original evidence.
    const int selectedCue = selectAmbiguousCue(original);
    reserveCueIdentity(candidates, selectedCue);

    // Assign each numbered identity at most once.
    const auto numbers = assignUnique(candidates);

    if (selectedCue >= 0 && numbers[selectedCue] != 0)
        throw std::runtime_error("Cue reservation failed");

    // Save final predictions before loading labels for evaluation.
    std::vector<std::string> finalTypes;

    for (int number : numbers) {
        finalTypes.push_back(number < 0 ? "Unknown" : number == 0 ? "White" : number <= 8 ? "Solid" : "Striped");
    }

    saveFinalResults(image, directory, circles, centres, finalTypes, numbers, context, printFinalResults);

    // Evaluate and save the summary.
    const IdentityRecognitionResult evaluation = evaluateBallIdentities(numbers, context);

    std::ofstream csv(directory / "identity_evaluation.csv");
    if (!csv.is_open())
        throw std::runtime_error("Could not create identity evaluation CSV");

    csv << "TP,FP,FN,correct,groundTruth,unknown,"
           "identityAccuracy,endToEndAccuracy\n";

    csv << evaluation.detection.TP << ','
        << evaluation.detection.FP << ','
        << evaluation.detection.FN << ','
        << evaluation.correct << ','
        << evaluation.groundTruthCount << ','
        << evaluation.unknown << ','
        << evaluation.accuracy << ','
        << evaluation.endToEndAccuracy << '\n';

    finishOutputFile(csv);
    return evaluation;
}

/*
=========================================
              EVALUATION
=========================================
*/

EvaluationResult evaluateDetections(const std::vector<cv::Point2f>& predictions, const std::vector<cv::Point2f>& groundTruth, double tolerance) {
    std::vector<std::vector<std::pair<double, size_t>>> candidates(predictions.size());

    for (size_t p = 0; p < predictions.size(); ++p) {
        for (size_t g = 0; g < groundTruth.size(); ++g) {
            double distance = std::hypot(double(predictions[p].x) - groundTruth[g].x, double(predictions[p].y) - groundTruth[g].y);
            if (distance <= tolerance)
                candidates[p].emplace_back(distance, g);
        }
        std::sort(candidates[p].begin(), candidates[p].end());
    }

    // Match each detection to at most one ground-truth ball, and vice versa.
    // Rearrange existing matches when doing so allows more balls to be matched.
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

// Helper to save the detection evaluation results
void saveDetectionEvaluation(const std::filesystem::path& directory, const EvaluationResult& evaluation) {

    std::filesystem::create_directories(directory);

    std::ofstream csv(directory / "detection_evaluation.csv");
    if (!csv.is_open())
        throw std::runtime_error("Could not create detection evaluation CSV");

    csv << "TP,FP,FN,precision,recall,F1,meanError\n";
    csv << evaluation.TP << ',' << evaluation.FP << ',' << evaluation.FN << ',' << evaluation.precision << ',' << evaluation.recall << ',' << evaluation.f1 << ',' << evaluation.meanError << '\n';

    finishOutputFile(csv);
}


EvaluationContext prepareEvaluation(
    const std::vector<cv::Vec3f>& circles,
    const std::string& filename) {

    EvaluationContext context;
    context.truth = loadLabelledGroundTruth(GROUND_TRUTH_PATH, filename);

    if (context.truth.empty())
        throw std::runtime_error("No ground truth for: " + filename);

    std::vector<cv::Point2f> centres;
    for (const LabelledBall& ball : context.truth)
        centres.push_back(ball.centre);

    context.matching = evaluateDetections(
        transformCentres(circles), centres, MATCH_TOLERANCE);

    return context;
}

TypeRecognitionResult evaluateBallTypes(const std::vector<BallTypeResult>& types, const EvaluationContext& context) {
    const auto& truth = context.truth;
    const auto& matching = context.matching;

    TypeRecognitionResult result;
    result.types = types;
    result.matched = matching.TP;
    result.groundTruthCount = static_cast<int>(truth.size());

    for (size_t i = 0; i < types.size(); ++i) {
        const int j = matching.predictionToGroundTruth[i];

        if (j < 0)
            continue;

        const int number = truth[j].number;
        const std::string trueType = number == 0 ? "White" : number <= 8 ? "Solid" : "Striped";

        if (types[i].type == trueType)
            ++result.correct;
    }

    if (result.matched > 0)
        result.accuracy = double(result.correct) / result.matched;
    result.endToEndAccuracy = double(result.correct) / result.groundTruthCount;

    return result;
}

ColourRecognitionResult evaluateBallColours(const std::vector<BallColourScores>& colours, const EvaluationContext& context) {
    const auto& truth = context.truth;
    const auto& matching = context.matching;

    ColourRecognitionResult result;
    result.colours = colours;

    for (size_t g = 0; g < truth.size(); ++g) {
        const int number = truth[g].number;

        if (number == 0)
            continue; // Exclude the cue from body-colour evaluation.

        ++result.groundTruthCount;

        const int p = matching.groundTruthToPrediction[g];
        if (p < 0)
            continue;

        ++result.matched;

        const int trueColour = number <= 8 ? number : number - 8;

        if (colours[p].bestColour == trueColour)
            ++result.correct;
    }

    if (result.matched > 0)
        result.accuracy = double(result.correct) / result.matched;
    if (result.groundTruthCount > 0)
        result.endToEndAccuracy = double(result.correct) / result.groundTruthCount;

    return result;
}

IdentityRecognitionResult evaluateBallIdentities(const std::vector<int>& numbers, const EvaluationContext& context) {
    const auto& truth = context.truth;
    const auto& matching = context.matching;

    IdentityRecognitionResult result;
    result.numbers = numbers;
    result.groundTruthCount = static_cast<int>(truth.size());

    result.detection = matching;

    for (size_t i = 0; i < numbers.size(); ++i) {
        if (numbers[i] < 0)
            ++result.unknown;

        const int g = result.detection.predictionToGroundTruth[i];

        if (g >= 0 && numbers[i] == truth[g].number)
            ++result.correct;
    }

    if (result.detection.TP > 0)
        result.accuracy = double(result.correct) / result.detection.TP;

    result.endToEndAccuracy = double(result.correct) / result.groundTruthCount;

    return result;
}

// Evaluate the system performance on specific ball groups
void accumulateBallGroups(const std::vector<int>& numbers, const EvaluationContext& context, std::map<std::string, BallGroupCounts>& groups) {

    if (numbers.size() != context.matching.predictionToGroundTruth.size())
        throw std::runtime_error("Identity count does not match detections");

    const auto colourOf = [](int number) {
        return number <= 8 ? number : number - 8;
    };

    for (size_t g = 0; g < context.truth.size(); ++g) {
        const int actual = context.truth[g].number;
        const int p = context.matching.groundTruthToPrediction[g];

        const std::string colour = actual == 0 ? "white" : bodyColourName(colourOf(actual));

        const std::string type = identityType(actual);

        const std::array<std::string, 3> keys = {
            "Colour: " + colour,
            "Type: " + type,
            "Ball: " + colour + " " + type +
                " (" + std::to_string(actual) + ")"
        };

        for (const auto& key : keys) {
            auto& count = groups[key];
            ++count.annotated;

            if (p < 0)
                continue;

            ++count.matched;
            const int predicted = numbers[p];

            if (predicted < 0 || predicted > 15)
                continue;

            count.correctColour += colourOf(predicted) == colourOf(actual);
            count.correctType += identityType(predicted) == type;
            count.correctIdentity += predicted == actual;
        }
    }
}

void printBallGroupEvaluation(const std::map<std::string, BallGroupCounts>& groups) {

    const auto percent = [](int correct, int total) {
        if (total == 0)
            return std::string("N/A");

        std::ostringstream text;
        text << std::fixed << std::setprecision(1) << 100.0 * correct / total << '%';

        return text.str();
    };

    std::cout << "\n---------Ball Groups: Table1–6---------\n"
              << std::left
              << std::setw(30) << "True group"
              << std::setw(8) << "Count"
              << std::setw(10) << "Detected"
              << std::setw(10) << "Colour"
              << std::setw(10) << "Type"
              << std::setw(10) << "Identity"
              << "End-to-end\n";

    for (const auto& [name, count] : groups) {
        std::cout << std::setw(30) << name
                  << std::setw(8) << count.annotated
                  << std::setw(10)
                  << percent(count.matched, count.annotated)
                  << std::setw(10)
                  << percent(count.correctColour, count.matched)
                  << std::setw(10)
                  << percent(count.correctType, count.matched)
                  << std::setw(10)
                  << percent(count.correctIdentity, count.matched)
                  << percent(count.correctIdentity, count.annotated)
                  << '\n';
    }
}

void printOverallEvaluation(const OverallEvaluation& totals) {
    const auto percent = [](int correct, int total) -> std::string {
        if (total == 0)
            return "N/A";

        std::ostringstream text;
        text << std::fixed << std::setprecision(2) << 100.0 * correct / total << "%";
        return text.str();
    };

    const int groundTruth = totals.TP + totals.FN;
    const int detections = totals.TP + totals.FP;

    const auto row = [&](const std::string& stage, int correct, int matched, int annotated) {
        std::cout << std::left << std::setw(16) << stage
                  << std::setw(12) << correct
                  << std::setw(20) << percent(correct, matched)
                  << percent(correct, annotated) << '\n';
    };

    std::cout << "\n"
              << "============================================================\n"
              << "              OVERALL EVALUATION: TABLE1–6\n"
              << "============================================================\n";

    std::cout << "Annotated balls: " << groundTruth << " | Detections: " << detections << '\n';

    std::cout << "TP: " << totals.TP << " | FP: " << totals.FP << " | FN: " << totals.FN << "\n\n";

    std::cout << "Detection precision: " << percent(totals.TP, detections) << '\n';

    std::cout << "Detection recall: " << percent(totals.TP, groundTruth) << '\n';

    std::cout << "Detection F1: " << percent(2 * totals.TP, 2 * totals.TP + totals.FP + totals.FN) << "\n\n";

    std::cout << std::left
              << std::setw(16) << "Recognition"
              << std::setw(12) << "Correct"
              << std::setw(20) << "Matched accuracy"
              << "End-to-end\n";

    std::cout << "------------------------------------------------------------\n";

    row("Type", totals.correctTypes, totals.TP, groundTruth);
    row("Colour", totals.correctColours, totals.matchedColours, totals.colourGroundTruth);
    row("Identity", totals.correctIdentities, totals.TP, groundTruth);

    std::cout << "\nUnknown identity predictions: " << totals.unknownIdentities << '\n';

    std::cout << "Colour metrics exclude the cue ball.\n"
              << "============================================================\n";

    printBallGroupEvaluation(totals.groups);
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
void saveFinalResults(const cv::Mat& image, const std::filesystem::path& directory, const std::vector<cv::Vec3f>& circles, const std::vector<cv::Point2f>& tableCentres, const std::vector<std::string>& types,
                      const std::vector<int>& numbers, const EvaluationContext& context, bool printFinalResults) {
    if (circles.size() != tableCentres.size() || circles.size() != types.size() || circles.size() != numbers.size())
        throw std::runtime_error("Final result vectors have inconsistent candidate counts");

    std::filesystem::create_directories(directory);

    cv::Mat original = image.clone();
    std::vector<cv::Scalar> outlineColours;

    for (size_t i = 0; i < circles.size(); ++i) {
        const int g = context.matching.predictionToGroundTruth[i];
        const bool correct = g >= 0 && numbers[i] == context.truth[g].number;
        const cv::Scalar colour = correct ? DETECTION_COLOUR : ERROR_COLOUR;

        outlineColours.push_back(colour);

        const cv::Point centre(cvRound(circles[i][0]), cvRound(circles[i][1]));

        cv::circle(original, centre, cvRound(circles[i][2]), colour, 2, cv::LINE_AA);
        cv::circle(original, centre, 2, cv::Scalar(255, 0, 0), cv::FILLED);
    }

    const cv::Mat rectified = transformImage(image);
    cv::Mat overhead = transformImage(original);

    for (size_t i = 0; i < circles.size(); ++i) {
        const cv::Point sourceCentre(cvRound(circles[i][0]), cvRound(circles[i][1]));

        drawFinalLabel(original, sourceCentre, numbers[i]);

        const cv::Point tableCentre(cvRound(tableCentres[i].x), cvRound(tableCentres[i].y));

        drawFinalLabel(overhead, tableCentre, numbers[i], {0, -12});
    }
    saveImage(directory / "original_numbered.png", original);
    saveImage(directory / "table_rectified.png", rectified);
    saveImage(directory / "table_numbered.png", overhead);
    
    
    const cv::Mat diagram =drawTableDiagram(tableCentres, numbers, outlineColours);
    saveImage(directory / "table_diagram.png", diagram);

    if (printFinalResults) {
        cv::Mat combined;
        cv::vconcat(overhead, diagram, combined);

        cv::namedWindow("Final results", cv::WINDOW_NORMAL);
        cv::resizeWindow("Final results", 950, 900);
        cv::imshow("Final results", combined);

        cv::waitKey(0);
        cv::destroyWindow("Final results");
    }

    std::ofstream csv(directory / "balls.csv");
    if (!csv.is_open())
        throw std::runtime_error("Could not create final results CSV: " + directory.string());

    csv << "candidate,predictedType,predictedNumber,originalX,originalY,originalRadiusPixels,tableX,tableY\n" << std::fixed << std::setprecision(6);
    for (size_t i = 0; i < circles.size(); ++i)
        csv << i + 1 << ',' << types[i] << ',' << numbers[i] << ',' << circles[i][0] << ',' << circles[i][1] << ',' << circles[i][2] << ',' << tableCentres[i].x << ',' << tableCentres[i].y << '\n';

    finishOutputFile(csv);
}

// outputs mentioned in the report
void reportSpecificOutputs() {
    using Row = std::map<std::string, std::string>;

    const auto readCSV = [](const std::string& path) {
        std::ifstream file(path);
        std::string line;
        if (!std::getline(file, line))
            throw std::runtime_error("Could not read: " + path);

        if (!line.empty() && line.back() == '\r')
            line.pop_back();

        std::vector<std::string> columns;
        std::stringstream header(line);
        for (std::string key; std::getline(header, key, ',');)
            columns.push_back(key);

        std::vector<Row> rows;
        while (std::getline(file, line)) {
            if (line.empty() || line == "\r")
                continue;

            Row row;
            std::stringstream values(line);
            for (const auto& key : columns)
                if (!std::getline(values, row[key], ','))
                    throw std::runtime_error("Incomplete CSV row: " + path);

            rows.push_back(row);
        }
        return rows;
    };

    const auto circleOf = [](const Row& r) {
        return cv::Vec3f(
            std::stof(r.at("originalX")),
            std::stof(r.at("originalY")),
            std::stof(r.at("originalRadiusPixels")));
    };

    const auto findCandidate = [&](const std::string& path, int candidate) {
        for (const auto& r : readCSV(path))
            if (std::stoi(r.at("candidate")) == candidate)
                return r;
        throw std::runtime_error("Candidate missing from: " + path);
    };

    std::ostringstream out;
    out << std::fixed << std::setprecision(2)
        << "\n---------Additional Report Figures---------\n";

    // Histogram examples: read saved similarities and margins.
    for (const auto& [table, candidate, ball] :
         std::vector<std::tuple<std::string, int, int>>{
             {"Table2", 6, 4}, {"Table1", 12, 10}
         }) {
        const auto r = findCandidate(
            RECOGNITION_DIR + table + "/ball_colours.csv", candidate);

        const double best = std::stod(r.at("similarity"));
        out << table << " ball " << ball
            << ": " << r.at("colour") << '=' << best
            << "; next best=" << best - std::stod(r.at("margin")) << '\n';
    }

    // Type examples: read K-means coverage and calculate HLS coverage.
    const cv::Mat reference = cv::imread(MEDIA_DIR + TUNNNING_IMG);
    if (reference.empty())
        throw std::runtime_error("Could not load reference image");

    out << "\nAllBalls type examples\n" << std::setprecision(1);

    for (const auto& [candidate, label] :
         std::vector<std::pair<int, std::string>>{
             {2, "Cue"}, {5, "Solid 1"}, {9, "Stripe 9"}
         }) {
        const auto geometry = findCandidate(
            FINAL_RES_DIR + "AllBalls/balls.csv", candidate);
        const auto type = findCandidate(
            RECOGNITION_DIR + "AllBalls/ball_types.csv", candidate);

        cv::Mat preview;
        const auto hls = pastHLSType(
            extractBall(reference, circleOf(geometry)), preview);
        const double light = std::stod(type.at("lightPercent"));

        out << label << ": HLS light=" << hls.lightPercent
            << "%; K=2 light=" << light
            << "%; non-light=" << 100 - light
            << "% -> " << type.at("type") << '\n';
    }

    const auto model = buildColourReferenceModel(reference);
    int matched = 0, paddingCorrect = 0, gains = 0, losses = 0;
    double distanceSum = 0;
    out << std::setprecision(2);

    for (const auto& filename : VAL_IMG) {
        const std::string table =
            std::filesystem::path(filename).stem().string();
        const cv::Mat image = cv::imread(MEDIA_DIR + filename);
        if (image.empty())
            throw std::runtime_error("Could not load: " + filename);

        std::vector<cv::Vec3f> circles;
        std::vector<Candidate> original;
        std::vector<int> oldColours;

        for (const auto& r : readCSV(FINAL_RES_DIR + table + "/balls.csv")) {
            Candidate c;
            c.circle = circleOf(r);
            c.roi = extractBall(image, c.circle);

            const auto segmentation = segmentBallKMeans(c.roi, 2);
            c.baselineType = classifyBallType(segmentation);
            c.colours = scoreBallColours(c.roi, model);
            identityEvidence(c);

            circles.push_back(c.circle);
            original.push_back(c);
            oldColours.push_back(pastHueColour(segmentation));
        }

        const auto context = prepareEvaluation(circles, filename);
        const auto detection = readCSV(
            DETECTION_DIR + table + "/detection_evaluation.csv").at(0);

        matched += std::stoi(detection.at("TP"));
        distanceSum += std::stod(detection.at("meanError"))
                     * std::stoi(detection.at("TP"));

        // Evaluate padding without applying cue reservation.
        auto padded = original;
        for (size_t i = 0; i < original.size(); ++i)
            if (original[i].baselineType.type == "White")
                padded[i] = rawWhitePadding(image, original[i], model, 1);

        paddingCorrect +=
            evaluateBallIdentities(assignUnique(padded), context).correct;

        for (size_t i = 0; i < original.size(); ++i) {
            const int g = context.matching.predictionToGroundTruth[i];
            if (g < 0)
                continue;

            const int number = context.truth[g].number;
            if (number > 0) {
                const int colour = number <= 8 ? number : number - 8;
                const bool before = oldColours[i] == colour;
                const bool after = original[i].colours.bestColour == colour;
                gains += !before && after;
                losses += before && !after;
            }

            if (table == "Table1" &&
                (number == 0 || number == 14 || number == 5 || number == 9)) {
                const auto r = findCandidate(
                    FINAL_RES_DIR + table + "/balls.csv", int(i) + 1);
                const cv::Point2f centre(
                    std::stof(r.at("tableX")), std::stof(r.at("tableY")));

                out << "Table1 ball " << number
                    << ": radius=" << original[i].circle[2]
                    << " px; centre error="
                    << cv::norm(centre - context.truth[g].centre)
                    << " table units\n";
            }
        }

        if (table == "Table2") {
            const auto& solid = original.at(5).identityScore;   // Candidate 6.
            const auto& stripe = original.at(10).identityScore; // Candidate 11.

            out << std::setprecision(3)
                << "\nTable2 crop | Score for 4 | Score for 12\n"
                << "Purple solid | " << solid[4] << " | " << solid[12]
                << "\nPurple stripe | " << stripe[4] << " | " << stripe[12]
                << "\nSelected pairing: " << solid[4] + stripe[12]
                << "; reverse pairing: " << solid[12] + stripe[4]
                << "\n\n" << std::setprecision(2);
        }
    }

    if (matched > 0)
        out << "Mean localisation error: "
            << distanceSum / matched << " table units\n";

    out << "After padding alone: " << paddingCorrect
        << '/' << matched << " correct identities\n"
        << "Colour corrections: " << gains
        << "; regressions: " << losses << '\n';

    std::cout << out.str();
}

/*
=========================================
            EXPERIMENTS
=========================================
*/

/*
======================
    GRID SEARCH
======================
*/

/*
Searches 480 Hough parameter combinations on AllBalls:
    1. load the training image
    2. Define values to test (dp, minDist, param1, param2)
    3. Run detection on each combination
    4. Evaluate each configuration (prioritise maximising TP, 
        then minimising FP, then localisation error)
    5. Save and Visualise best combination
*/
void runGridSearch() {
    cv::Mat image = cv::imread(MEDIA_DIR + TUNNNING_IMG);

    if (image.empty()) {
        std::cerr << "Could not open AllBalls.jpeg" << std::endl;
        return;
    }

    cv::Mat maskedImage = maskTable(image);

    std::vector<cv::Point2f> groundTruth = loadGroundTruth(GROUND_TRUTH_PATH, TUNNNING_IMG);

    if (groundTruth.size() != 16) {
        std::cerr << "Invalid ground truth: " << groundTruth.size() << " balls" << std::endl;
        return;
    }

    // define parameter search space (480 configurations)
    std::vector<double> dpValues = {1.0, 1.5, 2.0};
    std::vector<double> minDistValues = {15, 18, 20, 22, 25};
    std::vector<double> param1Values = {75, 100, 125, 150};
    std::vector<double> param2Values = {15, 20, 25, 30, 35, 40, 45, 50};

    std::string output_dir = EXPERIMENT_DIR + "grid_search/";
    std::filesystem::create_directories(output_dir);
    std::ofstream file(output_dir + "grid_search.csv");

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

    // test every parameter combination
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
                    const auto rawCircles = detectBalls(maskedImage, params);
                    const auto circles = filterPocketDetections(rawCircles);

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

    // display the results of the best configuration
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

    // save the visualisation of the best configuration
    const auto bestRawCircles = detectBalls(maskedImage, bestParams);
    const auto bestCircles = filterPocketDetections(bestRawCircles);
    cv::Mat bestImage = drawDetectedBalls(maskedImage, bestCircles);

    saveImage(output_dir + "best_detection.jpeg", bestImage);
}

/*
=========================================
            PAST SOLUTIONS
=========================================
*/

PastImage preparePastImage(const std::string& filename, const std::filesystem::path& root) {
    PastImage data;

    data.filename = filename;
    data.directory = root / std::filesystem::path(filename).stem();
    data.image = cv::imread(MEDIA_DIR + filename);

    if (data.image.empty())
        throw std::runtime_error("Could not load: " + filename);

    data.detection = runDetection(data.image, HoughParams{}, data.directory / "selected_hough", filename);
    data.context = prepareEvaluation(data.detection.filteredCircles, filename);
    data.balls = runExtraction(data.image, data.detection.filteredCircles, data.directory);
    data.segmentations = runKMeans(data.balls, 2, data.directory);

    return data;
}

/*
======================
    PAST RULES &
    THRESHOLDS
======================
*/

BallTypeResult pastHLSType(const BallROI& ball, cv::Mat& preview) {
    BallTypeResult result;

    cv::Mat hls;
    cv::cvtColor(ball.image, hls, cv::COLOR_BGR2HLS);

    preview = cv::Mat::zeros(ball.image.size(), CV_8UC3);

    const int total = cv::countNonZero(ball.mask);
    if (total == 0)
        return result;

    for (int y = 0; y < hls.rows; ++y) {
        for (int x = 0; x < hls.cols; ++x) {
            if (!ball.mask.at<uchar>(y, x))
                continue;

            const auto pixel = hls.at<cv::Vec3b>(y, x);
            const int lightness = pixel[1];
            const int saturation = pixel[2];

            cv::Vec3b colour(100, 100, 100);

            if (lightness >= 175 && saturation <= 85) {
                ++result.lightPercent;
                colour = {255, 255, 255};
            } else if (lightness <= 65) {
                ++result.darkPercent;
                colour = {35, 35, 35};
            } else if (saturation >= 85) {
                ++result.colouredPercent;
                colour = {255, 100, 0};
            }

            preview.at<cv::Vec3b>(y, x) = colour;
        }
    }

    result.lightPercent *= 100.0 / total;
    result.darkPercent *= 100.0 / total;
    result.colouredPercent *= 100.0 / total;

    const double nonLight = result.darkPercent + result.colouredPercent;

    if (result.lightPercent >= 85)
        result.type = "White";
    else if (nonLight >= 70)
        result.type = "Solid";
    else if (result.lightPercent > 0 && nonLight > 0)
        result.type = "Striped";

    return result;
}

int pastHueColour(const KMeansResult& segmentation) {
    const KMeansRegion* body = nullptr;

    for (const auto& region : segmentation.regions) {
        if (region.percentage < 2)
            continue;

        if (region.hsv[2] >= 170 && region.hsv[1] <= 165)
            continue;

        if (!body || region.percentage > body->percentage)
            body = &region;
    }

    if (!body)
        return -1;

    const int h = body->hsv[0];
    const int s = body->hsv[1];
    const int v = body->hsv[2];

    if (s <= 60)
        return v <= 90 ? 8 : -1;  // Black.
    if (h < 6 || h >= 170)
        return v < 150 ? 7 : 3;   // Maroon / red.
    if (h < 15)
        return 5;                // Orange.
    if (h < 35)
        return 1;                // Yellow.
    if (h >= 45 && h < 90)
        return 6;                // Green.
    if (h >= 100 && h < 130)
        return 2;                // Blue.
    if (h >= 130 && h < 165)
        return 4;                // Purple.

    return -1;
}

int pastHardIdentity(const BallTypeResult& type, int colour) {
    if (type.type == "White")
        return 0;

    if (colour < 1 || colour > 8)
        return -1;

    if (type.type == "Solid")
        return colour;

    if (type.type == "Striped" && colour <= 7)
        return colour + 8;

    return -1;
}

/*
======================
    MANUAL HOUGH
======================
*/

void runPastHoughComparison(const PastImage& data, std::map<std::string, EvaluationResult>& totals) {
    const HoughParams starting{2.0, 20.0, 150.0, 20.0, 15, 30};

    const auto initial = runDetection(data.image, starting, data.directory / "starting_hough", data.filename);

    const std::array<std::pair<std::string, DetectionResult>, 2> variants{{
        {"starting_hough", initial},
        {"selected_hough", data.detection}
    }};

    std::vector<ImagePanel> panels;

    for (const auto& item : variants) {
        const auto& result = item.second;
        const auto& evaluation = result.evaluation;

        panels.push_back({
            item.first +
                " TP=" + std::to_string(evaluation.TP) +
                " FP=" + std::to_string(evaluation.FP) +
                " FN=" + std::to_string(evaluation.FN),

            drawDetectedBalls(
                result.maskedImage, result.filteredCircles)
        });

        if (data.filename != TUNNNING_IMG) {
            auto& total = totals[item.first];

            total.TP += evaluation.TP;
            total.FP += evaluation.FP;
            total.FN += evaluation.FN;

            total.meanError += evaluation.meanError * evaluation.TP;
        }
    }

    saveImage(data.directory / "hough_comparison.png", combineImages(panels, 2));
}

/*
======================
    MANUAL HLS
    RULES
======================
*/

std::vector<BallTypeResult> runPastTypeComparison(const PastImage& data, std::vector<PastSummaryRow>& rows) {
    const auto& circles = data.detection.filteredCircles;

    std::vector<BallTypeResult> hls;
    std::vector<BallTypeResult> k2;
    std::vector<ImagePanel> pixels;

    for (size_t i = 0; i < data.balls.size(); ++i) {
        cv::Mat preview;

        hls.push_back(pastHLSType(data.balls[i], preview));
        k2.push_back(classifyBallType(data.segmentations[i]));

        pixels.push_back({"Ball " + std::to_string(i + 1) + ": " + hls.back().type, preview});
    }

    for (const auto& method : {std::string("hls"), std::string("k2")}) {
        const auto& types = method == "hls" ? hls : k2;
        const auto evaluation = evaluateBallTypes(types, data.context);

        rows.push_back({
            data.filename,
            "type_" + method,
            evaluation.correct,
            evaluation.matched,
            evaluation.groundTruthCount
        });
    }

    saveImage(
        data.directory / "type_comparison.png",
        combineImages({
            {"HLS", drawBallTypes(data.image, circles, hls)},
            {"K=2", drawBallTypes(data.image, circles, k2)}
        }, 2));

    if (!pixels.empty()) {
        saveImage(
            data.directory / "hls_grid.png",
            combineImages(pixels, 4));
    }

    return k2;
}

/*
======================
    K MEANS HUE
    RULES
======================
*/

std::vector<BallColourScores> runPastColourComparison(const PastImage& data, const BallColourModel& model, std::vector<PastSummaryRow>& rows) {
    const auto& circles = data.detection.filteredCircles;

    std::vector<BallColourScores> hue;

    for (const auto& segmentation : data.segmentations) {
        BallColourScores colour;
        colour.bestColour = pastHueColour(segmentation);

        hue.push_back(colour);
    }

    const auto oldResult = evaluateBallColours(hue, data.context);
    const auto current = runColourRecognition(data.image, data.balls, circles, model, data.directory / "histogram", data.filename, data.context);

    rows.push_back({data.filename, "colour_hue", oldResult.correct, oldResult.matched, oldResult.groundTruthCount});
    rows.push_back({data.filename, "colour_histogram", current.correct, current.matched, current.groundTruthCount});

    saveImage(
        data.directory / "colour_comparison.png",
        combineImages({
            {
                "K=2 representative hue",
                drawBallColours(data.image, circles, hue)
            },
            {
                "HSV histogram",
                drawBallColours(data.image, circles, current.colours)
            }
        }, 2));

    return current.colours;
}

/*
======================
    PAST SOLUTIONS
    COMPARAISONS
======================
*/

void runPastIdentityComparison(
    const PastImage& data,
    const std::vector<BallTypeResult>& types,
    const std::vector<BallColourScores>& colours,
    const BallColourModel& model,
    std::vector<PastSummaryRow>& rows) {

    const auto& circles = data.detection.filteredCircles;
    const auto centres = transformCentres(circles);
    const auto directory = data.directory / "identity";

    std::vector<Candidate> candidates;
    std::vector<int> hardHue;
    std::vector<int> hardHistogram;
    std::vector<int> softLocal;

    for (size_t i = 0; i < circles.size(); ++i) {
        Candidate candidate;

        candidate.circle = circles[i];
        candidate.tableCentre = centres[i];
        candidate.roi = data.balls[i];
        candidate.baselineType = types[i];
        candidate.colours = colours[i];

        identityEvidence(candidate);
        candidates.push_back(candidate);

        hardHue.push_back(pastHardIdentity(types[i], pastHueColour(data.segmentations[i])));
        hardHistogram.push_back(pastHardIdentity(types[i], colours[i].bestColour));

        softLocal.push_back(candidate.eligible[candidate.localBest] ? candidate.localBest : -1);
    }

    const std::vector<std::pair<std::string, std::vector<int>>> variants{
        {"hard_hue", hardHue},
        {"hard_histogram", hardHistogram},
        {"soft_local", softLocal},
        {"soft_unique", assignUnique(candidates)}
    };

    std::vector<ImagePanel> panels;

    for (const auto& variant : variants) {
        const auto evaluation = evaluateBallIdentities(
            variant.second, data.context);

        rows.push_back({
            data.filename,
            "identity_" + variant.first,
            evaluation.correct,
            evaluation.detection.TP,
            evaluation.groundTruthCount,
            evaluation.unknown
        });

        std::vector<std::string> labels;

        for (int number : variant.second)
            labels.push_back(identityType(number));

        saveFinalResults(data.image, directory / variant.first, circles, centres, labels, variant.second, data.context);

        panels.push_back({
            variant.first,
            cv::imread(
                (directory / variant.first / "table_numbered.png").string())
        });
    }

    // Reuse the complete current pipeline, including cue corrections.
    const auto final = runIdentityRecognition(data.image, data.balls, circles, types, colours, model, directory / "final", data.filename, data.context);

    rows.push_back({
        data.filename, "identity_final",
        final.correct, final.detection.TP,
        final.groundTruthCount, final.unknown
    });

    panels.push_back({
        "Final with cue corrections",
        cv::imread(
            (directory / "final" / "table_numbered.png").string())
    });

    saveImage(data.directory / "identity_comparison.png", combineImages(panels, 2));
}

/*
======================
    PAST SOLUTIONS
    RESULTS
======================
*/

void savePastSummaries(const std::filesystem::path& root, const std::vector<PastSummaryRow>& rows, const std::map<std::string, EvaluationResult>& detectionTotals) {
    std::ofstream csv(root / "recognition_comparison.csv");

    if (!csv)
        throw std::runtime_error("Could not create comparison CSV");

    const auto ratio = [](int numerator, int denominator) {
        return denominator ? double(numerator) / denominator : 0.0;
    };

    csv << "image,method,correct,matched,annotated," "accuracy,endToEnd,unknown\n";

    const auto write = [&](const PastSummaryRow& row) {
        csv << row.image << ',' << row.method << ','
            << row.correct << ',' << row.matched << ','
            << row.annotated << ','
            << ratio(row.correct, row.matched) << ','
            << ratio(row.correct, row.annotated) << ','
            << row.unknown << '\n';
    };

    std::map<std::string, PastSummaryRow> totals;

    for (const auto& row : rows) {
        write(row);

        if (row.image == TUNNNING_IMG)
            continue;

        auto& total = totals[row.method];

        total.correct += row.correct;
        total.matched += row.matched;
        total.annotated += row.annotated;
        total.unknown += row.unknown;
    }

    std::cout << "\nPast designs: Table1-6 (AllBalls excluded)\n";

    for (auto entry : totals) {
        auto& total = entry.second;

        total.image = "TOTAL_TABLE1_6";
        total.method = entry.first;

        write(total);

        std::cout
            << total.method << ": "
            << total.correct << '/' << total.matched << " matched; "
            << total.correct << '/' << total.annotated
            << " end-to-end\n\n";
    }

    finishOutputFile(csv);

    for (auto entry : detectionTotals) {
        auto& evaluation = entry.second;

        evaluation.precision = ratio(evaluation.TP, evaluation.TP + evaluation.FP);
        evaluation.recall = ratio(evaluation.TP, evaluation.TP + evaluation.FN);
        evaluation.f1 = ratio(2 * evaluation.TP, 2 * evaluation.TP + evaluation.FP + evaluation.FN);
        evaluation.meanError = evaluation.TP ? evaluation.meanError / evaluation.TP : 0;

        saveDetectionEvaluation(root / "overall" / entry.first, evaluation);

        std::cout
            << entry.first
            << ": TP=" << evaluation.TP
            << " FP=" << evaluation.FP
            << " FN=" << evaluation.FN
            << " F1=" << 100 * evaluation.f1 << "%\n\n";
    }

    std::cout << "\n";
}

/*
======================
    PAST SOLUTIONS
    RUNNER
======================
*/

void runPastDesigns() {
    const auto root = std::filesystem::path(EXPERIMENT_DIR) / "past_designs";

    const cv::Mat reference = cv::imread(MEDIA_DIR + TUNNNING_IMG);
    const auto model = buildColourReferenceModel(reference);

    std::vector<PastSummaryRow> rows;
    std::map<std::string, EvaluationResult> detectionTotals;

    for (const auto& filename : getAllImages()) {
        const auto data = preparePastImage(filename, root);

        runPastHoughComparison(data, detectionTotals);

        const auto types = runPastTypeComparison(data, rows);
        const auto colours = runPastColourComparison(data, model, rows);

        runPastIdentityComparison(data, types, colours, model, rows);
    }

    savePastSummaries(root, rows, detectionTotals);
}


/*
=========================================
            MAIN APPLICATION
=========================================
*/

void runMainPipeline() {
    const HoughParams params;
    OverallEvaluation totals;
    const cv::Mat referenceImage = cv::imread(MEDIA_DIR + TUNNNING_IMG);

    // run histogram training -> to get the 8 reference colour histograms
    const BallColourModel colourModel = buildColourReferenceModel(referenceImage);

    for (const std::string& filename : getAllImages()) {
        const cv::Mat image = cv::imread(MEDIA_DIR + filename);

        const std::filesystem::path directory_detection = std::filesystem::path(DETECTION_DIR) / std::filesystem::path(filename).stem();
        const std::filesystem::path directory_recognition = std::filesystem::path(RECOGNITION_DIR) / std::filesystem::path(filename).stem();
        const std::filesystem::path directory_final = std::filesystem::path(FINAL_RES_DIR) / std::filesystem::path(filename).stem();

        if (image.empty())
            throw std::runtime_error("Could not load: " + filename);

        // Run detection.
        const DetectionResult detection = runDetection(image, params, directory_detection, filename);

        std::cout << "---------Detection Results---------\n";
        std::cout << filename << ": " << detection.rawCircles.size() << " raw circles, " << detection.filteredCircles.size() << " after filtering\n";

        std::cout << "Precision: " << 100.0 * detection.evaluation.precision << "%\n"
                  << "Recall: " << 100.0 * detection.evaluation.recall << "%\n"
                  << "F1: " << 100.0 * detection.evaluation.f1 << "%\n";
        
        const EvaluationContext context = prepareEvaluation(detection.filteredCircles, filename);

        // Run Ball Extraction
        const std::vector<BallROI> balls = runExtraction(image, detection.filteredCircles, directory_detection);

        // Run Recognition Pipeline
        const TypeRecognitionResult recognition = runTypeRecognition(image, balls, detection.filteredCircles, directory_recognition, filename, context);

        std::cout << "---------Ball Type Results---------\n";
        std::cout << filename << ": " << recognition.correct << "/" << recognition.matched << " matched balls correctly typed\n";

        if (recognition.matched > 0) {
            std::cout << "Type accuracy: " << 100.0 * recognition.accuracy << "%\n";
        }

        std::cout << "Correctly detected and typed: " << recognition.correct << "/" << recognition.groundTruthCount << " (" << 100.0 * recognition.endToEndAccuracy << "%)\n";

        // run colour recognition
        const ColourRecognitionResult colourRecognition = runColourRecognition(
            image, balls, detection.filteredCircles,
            colourModel, directory_recognition, filename, context);

        std::cout << "---------Ball Colour Results---------\n";
        std::cout << filename << ": "
                << colourRecognition.correct << "/"
                << colourRecognition.matched
                << " matched numbered balls correctly coloured\n";

        if (colourRecognition.matched > 0) {
            std::cout << "Colour accuracy: " << 100.0 * colourRecognition.accuracy << "%\n";
        } else {
            std::cout << "Colour accuracy: N/A\n";
        }

        std::cout << "Correctly detected and coloured: "
                << colourRecognition.correct << "/"
                << colourRecognition.groundTruthCount
                << " (" << 100.0 * colourRecognition.endToEndAccuracy << "%)\n";
        
        // run identity recognition
        const IdentityRecognitionResult identity = runIdentityRecognition(image, balls, 
                detection.filteredCircles, recognition.types, colourRecognition.colours, 
                colourModel, directory_final, filename, context, true);

        std::cout << "---------Ball Identity Results---------\n";
        std::cout << filename << ": "
                << identity.correct << "/" << identity.detection.TP
                << " matched balls correctly numbered\n";

        if (identity.detection.TP > 0) {
            std::cout << "Identity accuracy: " << 100.0 * identity.accuracy << "%\n";
        } else {
            std::cout << "Identity accuracy: N/A\n";
        }

        std::cout << "Correctly detected and numbered: "
                << identity.correct << "/" << identity.groundTruthCount
                << " (" << 100.0 * identity.endToEndAccuracy << "%)\n";

        std::cout << "Unknown predictions: " << identity.unknown << '\n';

        std::cout << "\n";
        
        // final evaluation
        if (filename != TUNNNING_IMG) {
            totals.TP += detection.evaluation.TP;
            totals.FP += detection.evaluation.FP;
            totals.FN += detection.evaluation.FN;

            totals.correctTypes += recognition.correct;

            totals.correctColours += colourRecognition.correct;
            totals.matchedColours += colourRecognition.matched;
            totals.colourGroundTruth += colourRecognition.groundTruthCount;

            totals.correctIdentities += identity.correct;
            totals.unknownIdentities += identity.unknown;

            accumulateBallGroups(identity.numbers, context, totals.groups);
        }
    }
    printOverallEvaluation(totals);
}

void MyApplication() {
    const bool runGridSeachExp = true;
    const bool runPast = true;
    const bool runReport = true;

    if (runGridSeachExp) {
        std::cout << "===========GRID SEARCH===========\n";
        runGridSearch();
        std::cout << "\n\n";
    }

    if (runPast){
        std::cout << "===========PAST DESIGNS===========\n";
        runPastDesigns();
        std::cout << "\n\n";
    }

    if (runReport) {
        std::cout << "===========REPORT SPECIFIC OUTPUT===========\n";
        reportSpecificOutputs();
        std::cout << "\n\n";
    }

    std::cout << "===========MAIN PIPELINE===========\n";
    runMainPipeline();
    std::cout << "\n";
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
