
#include "recognition/extractBall.h"
#include "recognition/colourAnalysis.h"
#include "detection/ballDetection.h"
#include "colourAnalysisDemo.h"
#include "config.h"
#include "io/output.h"
#include "visualisation/combineImages.h"

#include <opencv2/opencv.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>

using namespace cv;
using namespace std;

const char* hueToColour(int hue) {
    if (hue < 0)
        return "Unknown";

    if (hue < 10 || hue >= 170) return "Red";
    if (hue < 25)  return "Orange";
    if (hue < 35)  return "Yellow";
    if (hue < 85)  return "Green";
    if (hue < 100) return "Cyan";
    if (hue < 130) return "Blue";
    if (hue < 150) return "Purple";

    return "Pink";
}

void testColourAnalysis() {

    // ----------------------------------------------
    // STEP 1: Load image and detect balls
    // ----------------------------------------------

    Mat image = imread(
        config::mediaDirectory + config::tuningImage
    );

    if (image.empty()) {
        cerr << "Could not open tuning image" << endl;
        return;
    }

    HoughParams params;

    vector<Vec3f> circles =
        detectAndFilterBalls(image, params);

    // ----------------------------------------------
    // STEP 2: Create output directory
    // ----------------------------------------------

    string outputDir =
        config::extractionOutputDirectory +
        "test/";

    filesystem::create_directories(outputDir);

    ofstream file(
        outputDir + "colour_stats.csv"
    );

    if (!file.is_open()) {
        cerr << "Could not create colour_stats.csv"
             << endl;
        return;
    }

    // ----------------------------------------------
    // STEP 3: CSV headers
    // ----------------------------------------------

    file << "candidate,x,y,radius,"
         << "whitePercent,colourPercent,"
         << "darkPercent,otherPercent,"
         << "largestWhiteRegionPercent,"
         << "largestColourRegionPercent,"
         << "whiteComponentCount,"
         << "colourComponentCount,"
         << "dominantHue,"
         << "dominantHuePercent,"
         << "dominantColour\n";

    file << fixed << setprecision(2);

    cout << fixed << setprecision(2);

    // ----------------------------------------------
    // STEP 4: Analyse each detected ball
    // ----------------------------------------------

    for (size_t i = 0; i < circles.size(); ++i) {

        BallROI ball = extractBall(
            image,
            circles[i]
        );

        if (ball.image.empty() || ball.mask.empty())
            continue;

        ColourStats stats =
            analyseBallColours(ball);

        string prefix =
            outputDir +
            "candidate_" +
            to_string(i + 1);

        // ------------------------------------------
        // Save diagnostic images
        // ------------------------------------------

        saveImage(prefix + "_comparison.png", combineImages({
            {"Original", ball.image},
            {"White mask", stats.whiteMask},
            {"Colour mask", stats.colourMask},
            {"Dark mask", stats.darkMask},
            {"Dominant colour", stats.dominantColourMask},
            {"Components", stats.componentVisualisation}
        }, 3));

        // ------------------------------------------
        // Save measurements to CSV
        // ------------------------------------------

        file << (i + 1) << ","
             << circles[i][0] << ","
             << circles[i][1] << ","
             << circles[i][2] << ","

             << stats.whitePercent << ","
             << stats.colourPercent << ","
             << stats.darkPercent << ","
             << stats.otherPercent << ","

             << stats.largestWhiteRegionPercent << ","
             << stats.largestColourRegionPercent << ","

             << stats.whiteComponentCount << ","
             << stats.colourComponentCount << ","

             << stats.dominantHue << ","
             << stats.dominantHuePercent << ","
             << hueToColour(stats.dominantHue)
             << "\n";

        // ------------------------------------------
        // Console output
        // ------------------------------------------

        cout << "\nBall candidate " << i + 1 << endl;

        cout << "  White: "
             << stats.whitePercent << "%" << endl;

        cout << "  Colour: "
             << stats.colourPercent << "%" << endl;

        cout << "  Dark: "
             << stats.darkPercent << "%" << endl;

        cout << "  Other: "
             << stats.otherPercent << "%" << endl;

        cout << "  Largest white region: "
             << stats.largestWhiteRegionPercent
             << "%" << endl;

        cout << "  Largest colour region: "
             << stats.largestColourRegionPercent
             << "%" << endl;

        cout << "  White components: "
             << stats.whiteComponentCount << endl;

        cout << "  Colour components: "
             << stats.colourComponentCount << endl;

        cout << "  Dominant hue: "
             << stats.dominantHue << endl;

        cout << "  Approximate colour: "
             << hueToColour(stats.dominantHue)
             << endl;
    }

    finishOutputFile(file);

    cout << "\nColour analysis completed."
         << endl;

    cout << "Results saved to: "
         << outputDir << endl;
}
