
#include "connectedComponentAnalysis.h"

#include <opencv2/opencv.hpp>
#include <algorithm>

using namespace cv;
using namespace std;

namespace {

    // Keep consistent with ballTypeClassifier.cpp.
    constexpr int LIGHT_MIN_V = 170;
    constexpr int LIGHT_MAX_S = 165;

    // Ignore components smaller than 2% of ball area.
    constexpr double MIN_COMPONENT_PERCENT = 2.0;

    struct ComponentStats {
        int count = 0;
        double largestPercent = 0;
        vector<Mat> regions;
    };

    // Is this K-means cluster classified as light?
    bool isLightCluster(const KMeansRegion& region) {
        Mat bgr(1, 1, CV_8UC3);
        bgr.at<Vec3b>(0, 0) = region.colour;

        Mat hsv;
        cvtColor(bgr, hsv, COLOR_BGR2HSV);

        Vec3b pixel = hsv.at<Vec3b>(0, 0);

        int S = pixel[1];
        int V = pixel[2];

        return V >= LIGHT_MIN_V && S <= LIGHT_MAX_S;
    }

    ComponentStats measureComponents(const Mat& mask, int ballArea) {
        ComponentStats result;

        Mat labels, stats, centroids;

        int count = connectedComponentsWithStats(mask, labels, stats, centroids, 8, CV_32S);

        // Label 0 is the background.
        for (int i = 1; i < count; ++i) {

            int area = stats.at<int>(i, CC_STAT_AREA);

            double percentage = 100.0 * area / ballArea;

            if (percentage < MIN_COMPONENT_PERCENT)
                continue;

            result.count++;
            Mat componentMask = (labels == i);
            result.regions.push_back(componentMask);
            result.largestPercent = max(result.largestPercent, percentage);
        }

        return result;
    }

}

CCAFeatures analyseConnectedComponents(const BallROI& ball, const KMeansResult& segmentation) {
    CCAFeatures result;

    if (ball.image.empty() || ball.mask.empty() || segmentation.regions.empty())
        return result;

    int ballArea = countNonZero(ball.mask);

    if (ballArea == 0)
        return result;

    // STEP 1: Build light and non-light masks

    Mat lightMask = Mat::zeros(ball.mask.size(), CV_8UC1);

    for (const KMeansRegion& region : segmentation.regions) {

        if (isLightCluster(region)) {
            bitwise_or(lightMask, region.mask, lightMask);
        }
    }

    // Restrict to the circular ball ROI.
    bitwise_and(lightMask, ball.mask, lightMask);

    Mat nonLightMask;

    bitwise_not(lightMask, nonLightMask);

    bitwise_and(nonLightMask, ball.mask, nonLightMask);

    // STEP 2: Connected-component measurements

    ComponentStats lightStats = measureComponents(lightMask, ballArea);

    ComponentStats nonLightStats = measureComponents(nonLightMask, ballArea);

    result.lightComponents = lightStats.count;

    result.largestLightComponentPercent = lightStats.largestPercent;

    result.nonLightComponents = nonLightStats.count;

    result.largestNonLightComponentPercent = nonLightStats.largestPercent;

    // STEP 3: Light coverage near ball boundary

    // Use an inner edge band of the circular mask.
    Mat erodedMask;

    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));

    erode(ball.mask, erodedMask, kernel);

    Mat boundaryBand;
    subtract(ball.mask, erodedMask, boundaryBand);

    int boundaryPixels = countNonZero(boundaryBand);

    if (boundaryPixels > 0) {

        Mat lightOnBoundary;

        bitwise_and(lightMask, boundaryBand, lightOnBoundary);

        result.lightBoundaryCoverage = 100.0 * countNonZero(lightOnBoundary) / boundaryPixels;
    }

    result.lightMask = lightMask;
    result.nonLightMask = nonLightMask;
    result.boundaryBand = boundaryBand;
    result.lightRegions = lightStats.regions;
    result.nonLightRegions = nonLightStats.regions;

    return result;
}
