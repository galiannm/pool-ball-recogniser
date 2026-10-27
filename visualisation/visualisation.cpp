#include "visualisation.h"

#include <string>

using namespace cv;
using namespace std;

Mat drawDetectedBalls(const Mat &image, const vector<Vec3f> &circles)
{
    Mat result = image.clone();

    for (const Vec3f &detected : circles)
    {
        Point center(cvRound(detected[0]), cvRound(detected[1]));

        int radius = cvRound(detected[2]);

        // Draw circle boundary
        circle(result, center, radius, Scalar(0, 0, 255), 2);

        // Draw centre
        circle(result, center, 2, Scalar(255, 0, 0), -1);
    }

    return result;
}

Mat drawValidationMatches(const Mat &transformedImage, const vector<Point2f> &predictions, const vector<Point2f> &groundTruth, const EvaluationResult &result)
{
    Mat canvas = transformedImage.clone();

    // Draw ground truth
    for (size_t g = 0; g < groundTruth.size(); ++g)
    {
        Point gt(cvRound(groundTruth[g].x), cvRound(groundTruth[g].y));

        if (result.groundTruthToPrediction[g] == -1)
        {
            // Missed ball
            drawMarker(canvas, gt, Scalar(255, 0, 0), MARKER_TILTED_CROSS, 25, 3);
        }
        else
        {
            // Ground-truth centre
            circle(canvas, gt, 7, Scalar(255, 0, 255), FILLED);
        }
    }

    // Draw detections
    for (size_t p = 0; p < predictions.size(); ++p)
    {
        Point centre(cvRound(predictions[p].x), cvRound(predictions[p].y));

        bool matched = result.predictionToGroundTruth[p] != -1;

        Scalar colour = matched ? Scalar(0, 255, 0) : Scalar(0, 0, 255);

        circle(canvas, centre, 10, colour, 2);

        if (matched)
        {
            int g = result.predictionToGroundTruth[p];

            line(
                canvas, centre,
                Point(cvRound(groundTruth[g].x), cvRound(groundTruth[g].y)),
                Scalar(255, 255, 255), 1);
        }
    }

    return canvas;
}

Mat drawGroundTruthOverlay(const Mat &transformedImage, const vector<Point2f> &groundTruth, const vector<int> &ballIds)
{
    Mat result = transformedImage.clone();

    for (size_t i = 0; i < groundTruth.size(); ++i)
    {
        Point centre(cvRound(groundTruth[i].x), cvRound(groundTruth[i].y));

        // Magenta circle
        circle(result, centre, 10, Scalar(255, 0, 255), 2);

        // Small filled centre
        circle(result, centre, 3, Scalar(255, 255, 255), FILLED);

        // Label with the original CSV ball number; column 16 is the cue ball.
        putText(result, ballIds[i] == 16 ? "White" : to_string(ballIds[i]), centre + Point(12, -12), FONT_HERSHEY_SIMPLEX, 0.55, Scalar(255, 255, 255), 2);
    }

    return result;
}

Mat drawClassifiedBalls(const Mat &image, const vector<Vec3f> &circles,
                        const vector<string> &predictedTypes, const vector<int> &predictedNumbers)
{
    Mat result = image.clone();

    for (size_t i = 0; i < circles.size(); ++i)
    {
        const Vec3f &detected = circles[i];

        Point centre(cvRound(detected[0]), cvRound(detected[1]));

        int radius = cvRound(detected[2]);

        string type = "Unknown";

        if (i < predictedTypes.size())
            type = predictedTypes[i];

        // Choose label and drawing colour
        string label;
        Scalar colour;

        if (type == "White")
        {
            label = "W";
            colour = Scalar(255, 255, 255);
        }
        else if (type == "Solid")
        {
            label = "S";
            colour = Scalar(0, 255, 0);
        }
        else if (type == "Striped")
        {
            label = "ST";
            colour = Scalar(0, 255, 255);
        }
        else
        {
            label = "?";
            colour = Scalar(0, 0, 255);
        }

        if (!predictedNumbers.empty()) {
            int number = i < predictedNumbers.size() ? predictedNumbers[i] : -1;
            if (number == 0)
                label += " (cue)";
            else if (number > 0)
                label += " " + to_string(number);
            else
                label += " ?";
        }

        // Draw detected circle
        circle(result, centre, radius, colour, 2);

        // Draw centre
        circle(result, centre, 2, colour, FILLED);

        // Position label beside the ball
        Point textPosition(centre.x + radius + 5, centre.y - radius);

        putText(result, label, textPosition, FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 0), 3, LINE_AA);
        putText(result, label, textPosition, FONT_HERSHEY_SIMPLEX, 0.6, colour, 2, LINE_AA);
    }

    return result;
}
