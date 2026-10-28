
#include "ballTypeClassifier.h"

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int classifyBallNumber(const KMeansResult& segmentation, const BallTypeResult& predictedType) {
    if (segmentation.regions.empty())
        return -1;
    if (predictedType.type == "White")
        return 0;
    if (predictedType.type != "Solid" && predictedType.type != "Striped")
        return -1;

    // Use the largest non-light cluster, including dark colours such as purple.
    const KMeansRegion* body = nullptr;
    for (const auto& region : segmentation.regions) {
        const int saturation = region.hsv[1];
        const int brightness = region.hsv[2];
        if (brightness >= 170 && saturation <= 165)
            continue;
        if (region.percentage < 2.0)
            continue;
        if (!body || region.percentage > body->percentage)
            body = &region;
    }
    if (!body)
        return -1;

    const int hue = body->hsv[0];
    const int saturation = body->hsv[1];
    const int brightness = body->hsv[2];

    // Black is a number decision; its type remains Solid.
    if (saturation <= 60) {
        if (brightness <= 90 && predictedType.type == "Solid")
            return 8;
        return -1;
    }

    // Initial rules for this image set, using OpenCV's 0-179 hue scale.
    // Red and maroon share a hue range, so brightness separates them here.
    // These thresholds need evaluation under different lighting conditions.
    int solidNumber = -1;
    if (hue < 6 || hue >= 170)
        solidNumber = brightness < 150 ? 7 : 3; // Maroon / red
    else if (hue < 15)
        solidNumber = 5; // Orange
    else if (hue < 35)
        solidNumber = 1; // Yellow
    else if (hue >= 45 && hue < 90)
        solidNumber = 6; // Green
    else if (hue >= 100 && hue < 130)
        solidNumber = 2; // Blue
    else if (hue >= 130 && hue < 165)
        solidNumber = 4; // Purple

    if (solidNumber == -1)
        return -1;
    return predictedType.type == "Striped" ? solidNumber + 8 : solidNumber;
}

BallTypeResult classifyBallType(const KMeansResult& segmentation) {
    BallTypeResult result;

    if (segmentation.regions.empty())
        return result;

    // Experimental thresholds
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
        }
        else if (V <= DARK_MAX_V) {
            result.darkPercent += region.percentage;
        }
        else {
            result.colouredPercent += region.percentage;
        }
    }

    // Classify using cluster proportions
    if (result.lightPercent >= WHITE_BALL_MIN) {
        result.type = "White";
    } else if (result.colouredPercent + result.darkPercent
             >= SOLID_MIN) {
        result.type = "Solid";
    } else if (result.lightPercent > 0 &&
             result.colouredPercent + result.darkPercent > 0) {
        result.type = "Striped";
    }

    return result;
}


BallTypeResult classifyBallTypeWithCCA( const KMeansResult& segmentation, const CCAFeatures& cca) {
    // First compute the existing K-means prediction.
    BallTypeResult result = classifyBallType(segmentation);

    const double light = result.lightPercent;
    const double nonLight = result.colouredPercent + result.darkPercent;

    // Experimental thresholds based on our
    // initial AllBalls.jpeg observations.
    const double WHITE_MIN_LIGHT = 85.0;
    const double WHITE_MIN_BOUNDARY = 75.0;

    const double STRIPE_MIN_LIGHT_COMPONENT = 20.0;
    const double STRIPE_MIN_BOUNDARY = 20.0;

    const double SOLID_MAX_BOUNDARY = 12.0;

    // ==========================================
    // CASE 1: White cue ball
    // ==========================================

    // Require both high light coverage and
    // strong boundary coverage.
    if (light >= WHITE_MIN_LIGHT &&
        cca.largestLightComponentPercent >= 80.0 &&
        cca.lightBoundaryCoverage >= WHITE_MIN_BOUNDARY &&
        cca.nonLightComponents == 0) {

        result.type = "White";
        return result;
    }

    // ==========================================
    // CASE 2: Strong evidence for a stripe
    // ==========================================

    // The largest light component should occupy
    // a meaningful part of the ball and extend
    // towards its boundary.
    if (cca.largestLightComponentPercent >= STRIPE_MIN_LIGHT_COMPONENT &&
        cca.lightBoundaryCoverage >= STRIPE_MIN_BOUNDARY && nonLight >= 10.0) {

        result.type = "Striped";
        return result;
    }

    // CASE 3: Strong evidence for a solid

    // Mostly non-light, little boundary contact,
    // and one dominant connected non-light region.
    if (nonLight >= 70.0 &&
        cca.lightBoundaryCoverage <= SOLID_MAX_BOUNDARY &&
        cca.nonLightComponents == 1) {

        result.type = "Solid";
        return result;
    }

    // Retain the existing K-means classification.
    return result;
}
