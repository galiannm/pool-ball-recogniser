#include "extractBall.h"

#include <algorithm>

using namespace cv;
using namespace std;

BallROI extractBall(const Mat& image, const Vec3f& detectedCircle){
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

    // Create binary mask
    result.mask = Mat::zeros(result.image.size(), CV_8UC1);

    // Circle centre relative to cropped image
    Point localCentre(x - roi.x, y - roi.y);

    circle(result.mask, localCentre, radius, Scalar(255), FILLED);

    return result;
}
