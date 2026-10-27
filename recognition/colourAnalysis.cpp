
#include "colourAnalysis.h"

#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>

using namespace cv;
using namespace std;

// --------------------------------------------------
// Measure connected components in a binary mask.
// Ignore very small regions caused by noise.
// --------------------------------------------------

static void analyseComponents(
    const Mat& binaryMask,
    int totalBallPixels,
    double& largestRegionPercent,
    int& componentCount,
    Mat* visualisation = nullptr
) {
    Mat labels, componentStats, centroids;

    int nLabels = connectedComponentsWithStats(
        binaryMask,
        labels,
        componentStats,
        centroids,
        8,
        CV_32S
    );

    largestRegionPercent = 0.0;
    componentCount = 0;

    if (visualisation != nullptr) {
        *visualisation = Mat::zeros(
            binaryMask.size(), CV_8UC3
        );
    }

    const int MIN_COMPONENT_AREA = 5;

    const Vec3b palette[] = {
        Vec3b(255, 100, 60),
        Vec3b(60, 220, 100),
        Vec3b(80, 80, 255),
        Vec3b(255, 220, 60),
        Vec3b(200, 80, 200)
    };

    int paletteIndex = 0;

    for (int label = 1; label < nLabels; ++label) {
        int area = componentStats.at<int>(
            label, CC_STAT_AREA
        );

        if (area < MIN_COMPONENT_AREA)
            continue;

        componentCount++;

        double percentage =
            100.0 * area / totalBallPixels;

        largestRegionPercent = max(
            largestRegionPercent,
            percentage
        );

        if (visualisation != nullptr) {
            Vec3b colour =
                palette[paletteIndex % 5];

            visualisation->setTo(
                Scalar(
                    colour[0],
                    colour[1],
                    colour[2]
                ),
                labels == label
            );

            paletteIndex++;
        }
    }
}

// --------------------------------------------------
// Analyse ball colours and spatial regions
// --------------------------------------------------

ColourStats analyseBallColours(const BallROI& ball) {
    ColourStats result;

    if (ball.image.empty() || ball.mask.empty())
        return result;

    Mat hls, hsv;

    cvtColor(ball.image, hls, COLOR_BGR2HLS);
    cvtColor(ball.image, hsv, COLOR_BGR2HSV);

    // OpenCV HLS: H 0-179, L/S 0-255
    // Experimental HLS thresholds
    const int WHITE_MIN_L = 165;
    const int WHITE_MAX_S = 110;

    const int DARK_MAX_L = 65;
    const int DARK_MAX_S = 90;

    const int COLOUR_MIN_S = 70;
    const int COLOUR_MIN_V = 35;

    // HSV settings for representative hue
    const int HUE_MIN_S = 100;
    const int HUE_MIN_V = 55;
    const int HUE_MAX_V = 235;
    const int HUE_TOLERANCE = 8;

    result.whiteMask =
        Mat::zeros(ball.image.size(), CV_8UC1);

    result.colourMask =
        Mat::zeros(ball.image.size(), CV_8UC1);

    result.darkMask =
        Mat::zeros(ball.image.size(), CV_8UC1);

    result.dominantColourMask =
        Mat::zeros(ball.image.size(), CV_8UC1);

    int white = 0;
    int coloured = 0;
    int dark = 0;
    int other = 0;
    int total = 0;

    int hueHistogram[180] = {0};
    int eligibleHuePixels = 0;

    // ----------------------------------------------
    // STEP 1: Pixel classification
    // ----------------------------------------------

    for (int y = 0; y < hls.rows; ++y) {
        for (int x = 0; x < hls.cols; ++x) {

            if (ball.mask.at<uchar>(y, x) == 0)
                continue;

            Vec3b hlsPixel = hls.at<Vec3b>(y, x);
            Vec3b hsvPixel = hsv.at<Vec3b>(y, x);

            int L = hlsPixel[1];
            int S = hlsPixel[2];

            int H = hsvPixel[0];
            int hsvS = hsvPixel[1];
            int V = hsvPixel[2];

            total++;

        // 1. White pixels
        if (L >= WHITE_MIN_L && S <= WHITE_MAX_S) {
            white++;
            result.whiteMask.at<uchar>(y, x) = 255;
        }

        // 2. Coloured pixels
        // Check saturation before classifying as dark.
        else if (S >= COLOUR_MIN_S && V >= COLOUR_MIN_V) {
            coloured++;
            result.colourMask.at<uchar>(y, x) = 255;
        }

        // 3. Dark, relatively neutral pixels
        else if (L <= DARK_MAX_L && S < DARK_MAX_S) {
            dark++;
            result.darkMask.at<uchar>(y, x) = 255;
        }

        // 4. Unclassified pixels
        else {
            other++;
        }

            // Hue histogram only includes sufficiently
            // saturated and visible pixels.
            if (hsvS >= HUE_MIN_S &&
                V >= HUE_MIN_V &&
                V <= HUE_MAX_V) {

                hueHistogram[H]++;
                eligibleHuePixels++;
            }
        }
    }

    if (total == 0)
        return result;

    result.whitePercent =
        100.0 * white / total;

    result.colourPercent =
        100.0 * coloured / total;

    result.darkPercent =
        100.0 * dark / total;

    result.otherPercent =
        100.0 * other / total;

    // ----------------------------------------------
    // STEP 2: Morphological opening
    // ----------------------------------------------
    // Remove small isolated pixels before analysing
    // connected regions.
    //
    // Keep raw masks unchanged for pixel percentages.

    Mat cleanWhite, cleanColour;

    Mat kernel = getStructuringElement(
        MORPH_ELLIPSE,
        Size(3, 3)
    );

    morphologyEx(
        result.whiteMask,
        cleanWhite,
        MORPH_OPEN,
        kernel
    );

    morphologyEx(
        result.colourMask,
        cleanColour,
        MORPH_OPEN,
        kernel
    );

    // Keep results inside the ball.
    bitwise_and(
        cleanWhite, ball.mask, cleanWhite
    );

    bitwise_and(
        cleanColour, ball.mask, cleanColour
    );

    // ----------------------------------------------
    // STEP 3: Connected-component measurements
    // ----------------------------------------------

    Mat whiteComponents, colourComponents;

    analyseComponents(
        cleanWhite,
        total,
        result.largestWhiteRegionPercent,
        result.whiteComponentCount,
        &whiteComponents
    );

    analyseComponents(
        cleanColour,
        total,
        result.largestColourRegionPercent,
        result.colourComponentCount,
        &colourComponents
    );

    // Make a combined component visualisation.
    // Coloured regions are drawn using the palette.
    // White regions are shown in white.
    result.componentVisualisation =
        colourComponents.clone();

    result.componentVisualisation.setTo(
        Scalar(255, 255, 255),
        cleanWhite
    );

    // ----------------------------------------------
    // STEP 4: Dominant hue
    // ----------------------------------------------

    if (eligibleHuePixels == 0)
        return result;

    int dominantHue = -1;
    int highestCount = 0;

    // Smooth using neighbouring histogram bins.
    // Hue wraps around between 179 and 0.

    for (int h = 0; h < 180; ++h) {
        int count = 0;

        for (int offset = -2; offset <= 2; ++offset) {
            int index = (h + offset + 180) % 180;
            count += hueHistogram[index];
        }

        if (count > highestCount) {
            highestCount = count;
            dominantHue = h;
        }
    }

    result.dominantHue = dominantHue;

    int matchingPixels = 0;

    for (int y = 0; y < hsv.rows; ++y) {
        for (int x = 0; x < hsv.cols; ++x) {

            if (ball.mask.at<uchar>(y, x) == 0)
                continue;

            Vec3b pixel = hsv.at<Vec3b>(y, x);

            int H = pixel[0];
            int S = pixel[1];
            int V = pixel[2];

            if (S < HUE_MIN_S ||
                V < HUE_MIN_V ||
                V > HUE_MAX_V)
                continue;

            int difference = abs(H - dominantHue);

            int hueDistance = min(
                difference,
                180 - difference
            );

            if (hueDistance <= HUE_TOLERANCE) {
                result.dominantColourMask
                    .at<uchar>(y, x) = 255;

                matchingPixels++;
            }
        }
    }

    result.dominantHuePercent =
        100.0 * matchingPixels / total;

    return result;
}
