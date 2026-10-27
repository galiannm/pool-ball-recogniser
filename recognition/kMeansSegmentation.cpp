
#include "kMeansSegmentation.h"

#include <opencv2/opencv.hpp>
#include <algorithm>
#include <vector>
#include <cmath>

using namespace cv;
using namespace std;

KMeansResult segmentBallKMeans(const BallROI& ball, int k) {
    KMeansResult result;
    result.k = k;

    if (ball.image.empty() || ball.mask.empty() || k < 2)
        return result;

    // STEP 1: Collect ball pixels
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

    // STEP 2: Prepare BGR samples for K-means
    Mat samples(total, 3, CV_32F);

    for (int i = 0; i < total; ++i) {

        Vec3b colour = ball.image.at<Vec3b>(pixels[i]);

        samples.at<float>(i, 0) = colour[0]; // B
        samples.at<float>(i, 1) = colour[1]; // G
        samples.at<float>(i, 2) = colour[2]; // R
    }

    // STEP 3: K-means clustering
    Mat labels;
    Mat centres;

    TermCriteria criteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 50, 0.2);

    // Fixed seed for reproducibility
    uint64 previousState = theRNG().state;
    theRNG().state = 12345;

    kmeans(samples, k, labels, criteria, 3, KMEANS_PP_CENTERS, centres);

    theRNG().state = previousState;

    // STEP 4: Create cluster masks
    vector<Mat> masks(k);

    for (int j = 0; j < k; ++j) {
        masks[j] = Mat::zeros(
            ball.image.size(),
            CV_8UC1
        );
    }

    result.segmentedImage = Mat::zeros(ball.image.size(), CV_8UC3);

    // STEP 5: Reconstruct segmented image
    for (int i = 0; i < total; ++i) {

        int cluster = labels.at<int>(i, 0);
        Point p = pixels[i];

        Vec3b colour(
            saturate_cast<uchar>(
                centres.at<float>(cluster, 0)
            ),
            saturate_cast<uchar>(
                centres.at<float>(cluster, 1)
            ),
            saturate_cast<uchar>(
                centres.at<float>(cluster, 2)
            )
        );

        result.segmentedImage.at<Vec3b>(p) = colour;

        masks[cluster].at<uchar>(p) = 255;
    }

    // STEP 6: Extract cluster information
    for (int j = 0; j < k; ++j) {

        int area = countNonZero(masks[j]);

        if (area == 0)
            continue;

        KMeansRegion region;

        region.mask = masks[j];

        region.percentage = 100.0 * area / total;

        region.colour = Vec3b(
            saturate_cast<uchar>(centres.at<float>(j, 0)),
            saturate_cast<uchar>(centres.at<float>(j, 1)),
            saturate_cast<uchar>(centres.at<float>(j, 2))
        );

        Mat bgr(1,1,CV_8UC3);
        bgr.at<Vec3b>(0,0) = region.colour;

        Mat hsv;
        cvtColor(bgr, hsv, COLOR_BGR2HSV);

        region.hsv = hsv.at<Vec3b>(0,0);

        result.regions.push_back(region);
    }

    // Sort largest regions first
    sort(
        result.regions.begin(),
        result.regions.end(),
        [](const KMeansRegion& a,
           const KMeansRegion& b) {
            return a.percentage > b.percentage;
        }
    );

    // compare cluster colours
    if (result.k == 2 && result.regions.size() == 2) {
        Vec3b colour1 = result.regions[0].colour;
        Vec3b colour2 = result.regions[1].colour;

        Mat bgr(1, 2, CV_8UC3);
        bgr.at<Vec3b>(0, 0) = colour1;
        bgr.at<Vec3b>(0, 1) = colour2;

        Mat hsv;
        cvtColor(bgr, hsv, COLOR_BGR2HSV);

        Vec3b hsv1 = hsv.at<Vec3b>(0,0);
        Vec3b hsv2 = hsv.at<Vec3b>(0,1);

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

        Vec3d difference (
            double(colour1[0] - colour2[0]),
            double(colour1[1] - colour2[1]),
            double(colour1[2] - colour2[2])
        );

        result.bgrDistance = norm(difference);
    }

    return result;
}
