#include "io/output.h"
#include "config.h"

#include "validation.h"

#include "evaluation/evaluation.h"
#include "detection/ballDetection.h"
#include "geometry/perspectiveTransform.h"
#include "visualisation/visualisation.h"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iomanip>
#include <filesystem>

using namespace cv;
using namespace std;

namespace fs = std::filesystem;

void runValidation()
{
    // Parameters selected using AllBalls.jpeg
    HoughParams params;

    const double MATCH_TOLERANCE = config::matchTolerance;
    const string CSV_PATH = config::groundTruthPath;
    const string OUTPUT_DIR = config::validationOutputDirectory;

    const auto& imageNames = config::validationImages;

    fs::create_directories(OUTPUT_DIR);

    ofstream file(OUTPUT_DIR + "validation_results.csv");

    if (!file.is_open())
    {
        cerr << "Could not create validation CSV" << endl;
        return;
    }

    file << "image,groundTruth,detections,"
         << "TP,FP,FN,precision,recall,F1,meanError\n";

    int totalTP = 0;
    int totalFP = 0;
    int totalFN = 0;
    int imagesEvaluated = 0;

    cout << fixed << setprecision(4);

    cout << "\n===== VALIDATION STARTED =====\n";

    for (const string& imageName : imageNames)
    {
        Mat image = imread(config::mediaDirectory + imageName);

        if (image.empty())
        {
            cerr << "Could not open: " << imageName << endl;
            continue;
        }

        // Ground truth for this image
        vector<Point2f> groundTruth =
            loadGroundTruth(CSV_PATH, imageName);

        if (groundTruth.empty())
        {
            cerr << "No ground truth for: "
                 << imageName << endl;
            continue;
        }

        // Stage 1: Detect circles
        // vector<Vec3f> circles = detectBalls(image, params);
        vector<Vec3f> circles = detectAndFilterBalls(image, params);

        // Stage 2: Transform detected centres
        vector<Point2f> predictions = transformCentres(circles);

        // Stage 3: Evaluate
        EvaluationResult result = evaluateDetections(predictions, groundTruth, MATCH_TOLERANCE);

        // Transform image using the same matrix
        Mat transformedImage = transformImage(image);

        // Draw predictions and ground truth together
        Mat matchImage = drawValidationMatches(transformedImage, predictions, groundTruth, result);

        // Create output filename
        string baseName = imageName.substr(
            0, imageName.find_last_of('.')
        );

        // Save validation visualisation
        saveImage(
            OUTPUT_DIR + baseName + "_matches.png",
            matchImage
        );

        // Write CSV row
        file << imageName << ","
             << groundTruth.size() << ","
             << circles.size() << ","
             << result.TP << ","
             << result.FP << ","
             << result.FN << ","
             << result.precision << ","
             << result.recall << ","
             << result.f1 << ","
             << result.meanError << "\n";

        // Print individual results
        cout << "\n" << imageName << endl;
        cout << "Ground truth: " << groundTruth.size() << endl;
        cout << "Detections: " << circles.size() << endl;
        cout << "TP: " << result.TP << endl;
        cout << "FP: " << result.FP << endl;
        cout << "FN: " << result.FN << endl;
        cout << "Precision: " << result.precision << endl;
        cout << "Recall: " << result.recall << endl;
        cout << "F1: " << result.f1 << endl;
        cout << "Mean error: " << result.meanError << endl;

        // Save visualisation
        Mat visualisation =
            drawDetectedBalls(image, circles);

        string outputPath =
            OUTPUT_DIR + imageName.substr(
                0, imageName.find_last_of('.')
            ) + "_detection.jpeg";

        saveImage(outputPath, visualisation);

        // Accumulate results
        totalTP += result.TP;
        totalFP += result.FP;
        totalFN += result.FN;
        imagesEvaluated++;
    }

    finishOutputFile(file);

    // Overall metrics across all evaluated images
    double precision = 0.0;
    double recall = 0.0;
    double f1 = 0.0;

    if (totalTP + totalFP > 0)
        precision = double(totalTP) / (totalTP + totalFP);

    if (totalTP + totalFN > 0)
        recall = double(totalTP) / (totalTP + totalFN);

    if (precision + recall > 0)
        f1 = 2 * precision * recall / (precision + recall);

    cout << "\n===== OVERALL VALIDATION =====\n";
    cout << "Images evaluated: " << imagesEvaluated << endl;
    cout << "Total TP: " << totalTP << endl;
    cout << "Total FP: " << totalFP << endl;
    cout << "Total FN: " << totalFN << endl;
    cout << "Overall precision: " << precision << endl;
    cout << "Overall recall: " << recall << endl;
    cout << "Overall F1: " << f1 << endl;

    cout << "\nResults saved to: "
         << OUTPUT_DIR << endl;
}
