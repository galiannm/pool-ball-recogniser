#include "finalResults.h"
#include "output.h"
#include "config.h"
#include "geometry/perspectiveTransform.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

using namespace cv;
using namespace std;

namespace {

// The existing homography maps the cushions to pixel endpoints
// (0,0)..(1433,682). At this resolution, one pixel is approximately one mm.
constexpr double ballDiameterMm = 43.0;

string finalBallLabel(int number) {
    if (number == 0) return "Cue";
    if (number >= 1 && number <= 15) return to_string(number);
    return "Unknown";
}

void drawFinalLabel(Mat& image, Point centre, int number, Point offset = {}) {
    const string label = finalBallLabel(number);
    const double scale = number < 0 ? 0.36 : 0.45;
    int baseline = 0;
    const Size text = getTextSize(label, FONT_HERSHEY_SIMPLEX, scale, 1, &baseline);
    Point anchor(centre.x - text.width / 2 + offset.x,
                 centre.y + text.height / 2 + offset.y);
    anchor.x = clamp(anchor.x, 2, max(2, image.cols - text.width - 2));
    anchor.y = clamp(anchor.y, text.height + 2, max(text.height + 2, image.rows - baseline - 2));
    putText(image, label, anchor, FONT_HERSHEY_SIMPLEX, scale, Scalar(0, 0, 0), 3, LINE_AA);
    putText(image, label, anchor, FONT_HERSHEY_SIMPLEX, scale, Scalar(255, 255, 255), 1, LINE_AA);
}

Scalar diagramBallColour(int number) {
    // BGR drawing colours represent the predicted identity, not measured pixels.
    switch (number >= 9 && number <= 15 ? number - 8 : number) {
        case 0: return {245, 245, 245};
        case 1: return {20, 215, 245};
        case 2: return {200, 70, 30};
        case 3: return {45, 45, 220};
        case 4: return {145, 55, 125};
        case 5: return {20, 130, 245};
        case 6: return {60, 145, 35};
        case 7: return {50, 40, 115};
        case 8: return {25, 25, 25};
        default: return {145, 145, 145};
    }
}

Mat drawTableDiagram(const vector<Point2f>& centres, const vector<int>& numbers) {
    Mat diagram(config::tableHeight, config::tableWidth, CV_8UC3, Scalar(65, 115, 45));
    rectangle(diagram, {0, 0}, {diagram.cols - 1, diagram.rows - 1}, Scalar(160, 190, 160), 1);
    // Nominal physical size; never reuse a source-image Hough radius here.
    const double radius = ballDiameterMm / 2;
    constexpr int shift = 8;
    constexpr int fixedScale = 1 << shift;
    for (size_t i = 0; i < centres.size(); ++i) {
        const Point fixedCentre(cvRound(centres[i].x * fixedScale), cvRound(centres[i].y * fixedScale));
        const int fixedRadius = cvRound(radius * fixedScale);
        const Scalar colour = diagramBallColour(numbers[i]);
        circle(diagram, fixedCentre, fixedRadius, colour, FILLED, LINE_AA, shift);
        if (numbers[i] >= 9 && numbers[i] <= 15) {
            // Clip a coloured middle band to a white disk, preserving its round outline.
            const int half = cvCeil(radius) + 2;
            const Point origin(cvFloor(centres[i].x) - half, cvFloor(centres[i].y) - half);
            const Rect patch = Rect(origin.x, origin.y, 2 * half + 1, 2 * half + 1)
                             & Rect(0, 0, diagram.cols, diagram.rows);
            if (!patch.empty()) {
                Mat mask = Mat::zeros(patch.size(), CV_8UC1);
                const Point localFixed = fixedCentre - Point(patch.x * fixedScale, patch.y * fixedScale);
                circle(mask, localFixed, fixedRadius, Scalar(255), FILLED, LINE_AA, shift);
                Mat ball(patch.size(), CV_8UC3, Scalar(245, 245, 245));
                const int y = cvRound(centres[i].y) - patch.y;
                rectangle(ball, {0, y - 10}, {patch.width - 1, y + 10}, colour, FILLED);
                ball.copyTo(diagram(patch), mask);
            }
        }
        circle(diagram, fixedCentre, fixedRadius, Scalar(230, 230, 230), 1, LINE_AA, shift);
    }
    // Draw labels last so neighbouring ball symbols cannot paint over them.
    for (size_t i = 0; i < centres.size(); ++i)
        drawFinalLabel(diagram, {cvRound(centres[i].x), cvRound(centres[i].y)}, numbers[i]);
    return diagram;
}

} // namespace

void saveFinalResults(const Mat& image, const filesystem::path& directory, const vector<Vec3f>& circles,
                      const vector<Point2f>& tableCentres, const vector<string>& types,
                      const vector<int>& numbers) {
    if (circles.size() != tableCentres.size() || circles.size() != types.size() ||
        circles.size() != numbers.size())
        throw runtime_error("Final result vectors have inconsistent candidate counts");

    filesystem::create_directories(directory);
    Mat original = image.clone();
    const Mat rectified = transformImage(image);
    Mat overhead = rectified.clone();
    for (size_t i = 0; i < circles.size(); ++i) {
        const Point sourceCentre(cvRound(circles[i][0]), cvRound(circles[i][1]));
        circle(original, sourceCentre, cvRound(circles[i][2]), Scalar(0, 220, 255), 2, LINE_AA);
        drawFinalLabel(original, sourceCentre, numbers[i]);

        const Point tableCentre(cvRound(tableCentres[i].x), cvRound(tableCentres[i].y));
        drawMarker(overhead, tableCentre, Scalar(0, 220, 255), MARKER_CROSS, 9, 1, LINE_AA);
        drawFinalLabel(overhead, tableCentre, numbers[i], {0, -12});
    }
    saveImage(directory / "original_numbered.png", original);
    saveImage(directory / "table_rectified.png", rectified);
    saveImage(directory / "table_numbered.png", overhead);
    saveImage(directory / "table_diagram.png", drawTableDiagram(tableCentres, numbers));

    ofstream csv(directory / "balls.csv");
    if (!csv.is_open()) throw runtime_error("Could not create final results CSV: " + directory.string());
    csv << "candidate,predictedType,predictedNumber,originalX,originalY,originalRadiusPixels,tableX,tableY\n"
        << fixed << setprecision(6);
    for (size_t i = 0; i < circles.size(); ++i)
        csv << i + 1 << ',' << types[i] << ',' << numbers[i] << ','
            << circles[i][0] << ',' << circles[i][1] << ',' << circles[i][2] << ','
            << tableCentres[i].x << ',' << tableCentres[i].y << '\n';
    finishOutputFile(csv);
    cout << "Final images and table coordinates saved to: " << directory << endl;
}

