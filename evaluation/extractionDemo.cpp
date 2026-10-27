#include "extractionDemo.h"
#include "recognition/extractBall.h"
#include "detection/ballDetection.h"
#include "config.h"
#include "io/output.h"

#include <filesystem>
#include <iostream>

using namespace cv;
using namespace std;

void testBallExtraction()
{
    Mat image = imread(config::mediaDirectory + config::tuningImage);

    if (image.empty())
    {
        cerr << "Could not open image" << endl;
        return;
    }

    HoughParams params;

    DetectionResult result = runDetection(image, params);
    const vector<Vec3f>& circles = result.filteredCircles;

    cout << "Before filtering: " << result.rawCircles.size()
        << " | After filtering: " << circles.size()
        << endl;

    filesystem::create_directories(config::extractionOutputDirectory);

    for (size_t i = 0; i < circles.size(); ++i)
    {
        BallROI ball = extractBall(image, circles[i]);

        if (ball.image.empty())
            continue;

        // Black background outside circular mask
        Mat isolated = Mat::zeros(
            ball.image.size(),
            ball.image.type()
        );

        ball.image.copyTo(isolated, ball.mask);

        string filename = config::extractionOutputDirectory + "ball_" + to_string(i + 1) + ".png";

        saveImage(filename, isolated);
    }

    cout << "Extracted " << circles.size() << " ball candidates." << endl;
}
