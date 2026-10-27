
#include "perspectiveTransform.h"
#include "config.h"

using namespace cv;
using namespace std;

// ======================================================
// SHARED PERSPECTIVE TRANSFORMATION MATRIX
// ======================================================

Mat getTableTransformation()
{
    // Original image corners
    vector<Point2f> source;

    for (const Point& corner : getTableCorners())
    {
        source.emplace_back(corner.x, corner.y);
    }

    // Transformed table coordinates
    vector<Point2f> destination = {
        Point2f(0, 0),
        Point2f(config::tableWidth - 1, 0),
        Point2f(config::tableWidth - 1, config::tableHeight - 1),
        Point2f(0, config::tableHeight - 1)
    };

    return getPerspectiveTransform(source, destination);
}

vector<Point> getTableCorners()
{
    return config::tableCorners;
}

// TRANSFORM DETECTED BALL CENTRES

vector<Point2f> transformCentres(
    const vector<Vec3f>& circles
)
{
    Mat matrix = getTableTransformation();

    vector<Point2f> centres;

    for (const Vec3f& c : circles)
    {
        centres.push_back(Point2f(c[0], c[1]));
    }

    vector<Point2f> transformed;

    if (!centres.empty())
    {
        perspectiveTransform(centres, transformed, matrix);
    }

    return transformed;
}

// TRANSFORM ENTIRE IMAGE

Mat transformImage(const Mat& image)
{
    Mat matrix = getTableTransformation();
    Mat transformed;
    warpPerspective(image, transformed, matrix,Size(config::tableWidth, config::tableHeight));
    return transformed;
}
