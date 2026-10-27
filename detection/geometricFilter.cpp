
#include "geometricFilter.h"
#include "config.h"
#include "geometry/perspectiveTransform.h"

#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

vector<Vec3f> filterPocketDetections(
    const vector<Vec3f>& circles
)
{
    // Table dimensions after perspective transformation
    const float width = config::tableWidth;
    const float height = config::tableHeight;

    // Approximate pocket locations in transformed coordinates
    vector<Point2f> pockets = {
        {0, 0},
        {width / 2, 0},
        {width - 1, 0},
        {0, height - 1},
        {width / 2, height - 1},
        {width - 1, height - 1}
    };

    // Initial threshold to test, not yet validated
    const float pocketRadius = config::pocketRadius;

    vector<Point2f> transformedCentres =
        transformCentres(circles);

    vector<Vec3f> filtered;

    for (size_t i = 0; i < circles.size(); ++i)
    {
        bool nearPocket = false;

        for (const Point2f& pocket : pockets)
        {
            if (norm(transformedCentres[i] - pocket)
                < pocketRadius)
            {
                nearPocket = true;
                break;
            }
        }

        if (!nearPocket)
        {
            filtered.push_back(circles[i]);
        }
    }

    return filtered;
}
