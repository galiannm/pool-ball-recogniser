#include "ballDetection.h"
#include "detection/geometricFilter.h"
#include "geometry/perspectiveTransform.h"

using namespace cv;
using namespace std;

vector<Vec3f> detectBalls(const Mat& image, const HoughParams& params){
    Mat grey_image;
    cvtColor(image, grey_image, COLOR_BGR2GRAY);

    Mat blurred_image;
    GaussianBlur(grey_image, blurred_image, Size(5, 5), 1.5);

    vector<Vec3f> circles;

    HoughCircles(
        blurred_image,
        circles,
        HOUGH_GRADIENT,
        params.dp,
        params.minDist,
        params.param1,
        params.param2,
        params.minRadius,
        params.maxRadius
    );

    return circles;
}

vector<Vec3f> detectAndFilterBalls(const Mat& image, const HoughParams& params){
    return runDetection(image, params).filteredCircles;
}

DetectionResult runDetection(const Mat& image, const HoughParams& params){
    DetectionResult result;

    if (image.empty())
        return result;

    result.maskedImage = maskTable(image);

    result.rawCircles = detectBalls(result.maskedImage, params);
    result.filteredCircles =
        filterPocketDetections(result.rawCircles);

    return result;
}

Mat maskTable(const Mat& image) {
    if (image.empty())
        return {};

    Mat mask = Mat::zeros(image.size(), CV_8UC1);
    fillConvexPoly(mask, getTableCorners(), Scalar(255));

    Mat maskedImage;
    bitwise_and(image, image, maskedImage, mask);

    return maskedImage;
}
