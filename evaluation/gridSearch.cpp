#include "io/output.h"
#include "config.h"

#include "gridSearch.h"
#include "evaluation.h"
#include "detection/ballDetection.h"
#include "geometry/perspectiveTransform.h"
#include "visualisation/visualisation.h"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <limits>
#include <iomanip>
#include <filesystem>

using namespace cv;
using namespace std;

const string IMAGE_PATH = config::mediaDirectory + config::tuningImage;
const string CSV_PATH = config::groundTruthPath;
const double MATCH_TOLERANCE = config::matchTolerance;

// ======================================================
// STEP 1: TEST CURRENT PARAMETERS (BASELINE)
// ======================================================

void testCurrentParameters() {
    Mat image = imread(IMAGE_PATH);

    if (image.empty()) {
        cerr << "Could not open AllBalls.jpeg" << endl;
        return;
    }

    Mat maskedImage = maskTable(image);

    vector<Point2f> groundTruth = loadGroundTruth(CSV_PATH, config::tuningImage);

    if (groundTruth.size() != 16) {
        cerr << "Expected 16 balls, found "
             << groundTruth.size() << endl;
        return;
    }

    // Current default parameters
    HoughParams params;

    // Detect balls
    vector<Vec3f> circles = detectBalls(maskedImage, params);

    // Transform centres into table coordinates
    vector<Point2f> predictions = transformCentres(circles);

    // Compare against ground truth
    EvaluationResult result = evaluateDetections(predictions, groundTruth, MATCH_TOLERANCE);

    cout << fixed << setprecision(4);

    cout << "\n===== CURRENT CONFIGURATION =====\n";
    cout << "Ground truth: " << groundTruth.size() << endl;
    cout << "Detections: " << circles.size() << endl;
    cout << "TP: " << result.TP << endl;
    cout << "FP: " << result.FP << endl;
    cout << "FN: " << result.FN << endl;
    cout << "Precision: " << result.precision << endl;
    cout << "Recall: " << result.recall << endl;
    cout << "F1: " << result.f1 << endl;
    cout << "Mean error: " << result.meanError
         << " pixels" << endl;

    // Save a visualisation of the baseline
    Mat visualisation = drawDetectedBalls(maskedImage, circles);
    saveImage(config::outputDirectory + "baseline.jpeg", visualisation);
}

// STEP 2: AUTOMATED GRID SEARCH
void runGridSearch() {
    Mat image = imread(IMAGE_PATH);

    if (image.empty()) {
        cerr << "Could not open AllBalls.jpeg" << endl;
        return;
    }

    Mat maskedImage = maskTable(image);

    vector<Point2f> groundTruth = loadGroundTruth(CSV_PATH, config::tuningImage);

    if (groundTruth.size() != 16) {
        cerr << "Invalid ground truth: "
             << groundTruth.size() << " balls" << endl;
        return;
    }

    // DEFINE PARAMETER SEARCH SPACE
    vector<double> dpValues = {1.0, 1.5, 2.0};
    vector<double> minDistValues = {15, 18, 20, 22, 25};
    vector<double> param1Values = {75, 100, 125, 150};
    vector<double> param2Values = {15, 20, 25, 30, 35, 40, 45, 50};
    // 480 configurations

    std::filesystem::create_directories(config::outputDirectory);
    ofstream file(config::outputDirectory + "grid_search.csv");

    if (!file.is_open()) {
        cerr << "Could not create output/grid_search.csv" << endl;
        return;
    }

    file << "dp,minDist,param1,param2,"
         << "minRadius,maxRadius,detections,"
         << "TP,FP,FN,precision,recall,F1,meanError\n";

    double bestTP = -1;
    int bestFP = numeric_limits<int>::max();
    double bestError = numeric_limits<double>::infinity();

    HoughParams bestParams;
    EvaluationResult bestResult;

    int count = 0;
    const size_t totalConfigurations = dpValues.size() * minDistValues.size() *
                                       param1Values.size() * param2Values.size();

    // ==================================================
    // TEST EVERY PARAMETER COMBINATION
    // ==================================================

    for (double dp : dpValues) {
        for (double minDist : minDistValues) {
            for (double p1 : param1Values) {
                for (double p2 : param2Values) {
                    HoughParams params;

                    params.dp = dp;
                    params.minDist = minDist;
                    params.param1 = p1;
                    params.param2 = p2;

                    // Fixed during this first search
                    params.minRadius = 15;
                    params.maxRadius = 30;

                    // Detect balls
                    vector<Vec3f> circles = detectBalls(maskedImage, params);

                    // Transform detected centres
                    vector<Point2f> predictions = transformCentres(circles);

                    // Evaluate
                    EvaluationResult result = evaluateDetections( predictions, groundTruth, MATCH_TOLERANCE);

                    // Save results
                    file << dp << ","
                         << minDist << ","
                         << p1 << ","
                         << p2 << ","
                         << params.minRadius << ","
                         << params.maxRadius << ","
                         << circles.size() << ","
                         << result.TP << ","
                         << result.FP << ","
                         << result.FN << ","
                         << result.precision << ","
                         << result.recall << ","
                         << result.f1 << ","
                         << result.meanError << "\n";

                    count++;

                    // Prioritise recall, then precision
                    // then centre localisation accuracy.
                    double candidateError =
                        result.TP > 0
                        ? result.meanError
                        : numeric_limits<double>::infinity();

                    bool better = (
                        // 1. Detect more actual balls
                        result.TP > bestTP ||

                        // 2. Same number of balls: fewer false positives
                        (result.TP == bestTP && result.FP < bestFP) ||

                        // 3. Same TP and FP: better localisation
                        (result.TP == bestTP && result.FP == bestFP && candidateError < bestError));

                    if (better) {
                        bestTP = result.TP;
                        bestFP = result.FP;
                        bestError = candidateError;

                        bestParams = params;
                        bestResult = result;
                    }

                    cout << "\rTested " << count << "/" << totalConfigurations << " configurations" << flush;
                }
            }
        }
    }

    finishOutputFile(file);

    // ==================================================
    // DISPLAY BEST CONFIGURATION
    // ==================================================

    cout << fixed << setprecision(4);

    cout << "\n\n===== GRID SEARCH COMPLETE =====\n";
    cout << "Configurations tested: " << count << endl;

    cout << "\nBEST PARAMETERS:\n";
    cout << "dp: " << bestParams.dp << endl;
    cout << "minDist: " << bestParams.minDist << endl;
    cout << "param1: " << bestParams.param1 << endl;
    cout << "param2: " << bestParams.param2 << endl;
    cout << "minRadius: " << bestParams.minRadius << endl;
    cout << "maxRadius: " << bestParams.maxRadius << endl;

    cout << "\nBEST PERFORMANCE:\n";
    cout << "TP: " << bestResult.TP << endl;
    cout << "FP: " << bestResult.FP << endl;
    cout << "FN: " << bestResult.FN << endl;
    cout << "Precision: " << bestResult.precision << endl;
    cout << "Recall: " << bestResult.recall << endl;
    cout << "F1: " << bestResult.f1 << endl;

    if (bestResult.TP > 0)
        cout << "Mean error: " << bestResult.meanError << " pixels" << endl;
    else
        cout << "Mean error: N/A" << endl;

    // ==================================================
    // SAVE VISUALISATION OF BEST CONFIGURATION
    // ==================================================

    vector<Vec3f> bestCircles = detectBalls(maskedImage, bestParams);

    Mat bestImage = drawDetectedBalls(maskedImage, bestCircles);

    saveImage(config::outputDirectory + "best_detection.jpeg", bestImage);

    cout << "\nSaved: output/grid_search.csv" << endl;
    cout << "Saved: output/best_detection.jpeg" << endl;
}
