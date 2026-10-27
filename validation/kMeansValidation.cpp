
#include "kMeansValidation.h"

#include "detection/ballDetection.h"
#include "recognition/extractBall.h"
#include "recognition/kMeansSegmentation.h"
#include "recognition/ballTypeClassifier.h"
#include "recognition/connectedComponentAnalysis.h"
#include "evaluation/evaluation.h"
#include "geometry/perspectiveTransform.h"
#include "config.h"
#include "io/output.h"
#include "visualisation/combineImages.h"

#include <opencv2/opencv.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <iomanip>

using namespace cv;
using namespace std;

string trueBallType(int number) {
    if (number == 0)
        return "White";

    if (number >= 1 && number <= 8)
        return "Solid";

    return "Striped";
}

void runRecognitionValidation() {

    const auto& images = config::validationImages;

    HoughParams params;

    // Overall statistics
    int totalGT = 0;
    int matched = 0;

    int baselineCorrectTotal = 0;
    int hybridCorrectTotal = 0;

    int changedPredictions = 0;
    int correctedByCCA = 0;
    int worsenedByCCA = 0;

    for (const string& name : images) {

        Mat image = imread(
            config::mediaDirectory + name
        );

        if (image.empty()) {
            cerr << "Could not load: " << name << endl;
            continue;
        }

        // ==========================================
        // STEP 1: Create output directory
        // ==========================================

        const filesystem::path outputDir =
            filesystem::path(config::validationOutputDirectory) /
            "recognition" / filesystem::path(name).stem();

        filesystem::create_directories(outputDir);

        const filesystem::path csvPath = outputDir / "recognition_results.csv";

        ofstream csv(csvPath);

        if (!csv.is_open()) {
            cerr << "Could not create recognition CSV: " << csvPath << '\n';
            return;
        }

        csv << "image,candidate,ballNumber,"
            << "trueType,baselineType,hybridType,"
            << "baselineCorrect,hybridCorrect,"
            << "lightPercent,colouredPercent,darkPercent,"
            << "lightComponents,largestLightComponentPercent,"
            << "lightBoundaryCoverage,"
            << "nonLightComponents,largestNonLightComponentPercent,"
            << "hueDifference,saturationDifference,"
            << "brightnessDifference,bgrDistance,"
            << "baselineNumber,hybridNumber,"
            << "baselineNumberCorrect,hybridNumberCorrect\n";

        csv << fixed << setprecision(6);

        // ==========================================
        // STEP 2: Load ground truth
        // ==========================================

        vector<LabelledBall> groundTruth = loadLabelledGroundTruth(config::groundTruthPath, name);

        vector<Point2f> gtCentres;

        for (const LabelledBall& ball : groundTruth) {
            gtCentres.push_back(ball.centre);
        }

        totalGT += static_cast<int>(
            groundTruth.size()
        );

        // ==========================================
        // STEP 3: Detect and match balls
        // ==========================================

        vector<Vec3f> circles = detectAndFilterBalls(image, params);

        vector<Point2f> predictions = transformCentres(circles);

        EvaluationResult evaluation = evaluateDetections(predictions, gtCentres, config::matchTolerance);

        // Per-image statistics
        int imageMatched = 0;

        int imageBaselineCorrect = 0;
        int imageHybridCorrect = 0;

        int imageBaselineNumberCorrect = 0;
        int imageHybridNumberCorrect = 0;

        int imageChanged = 0;
        int imageCorrectedByCCA = 0;
        int imageWorsenedByCCA = 0;

        // STEP 4: Classify matched detections
        for (size_t i = 0; i < circles.size(); ++i) {

            int gtIndex =
                evaluation.predictionToGroundTruth[i];

            // Ignore unmatched detections:
            // these are detection false positives.
            if (gtIndex == -1)
                continue;

            // Extract ROI
            BallROI roi = extractBall(image, circles[i]);

            if (roi.image.empty() || roi.mask.empty())
                continue;

            // --------------------------------------
            // K-means segmentation
            // --------------------------------------

            KMeansResult segmentation =
                segmentBallKMeans(roi, config::recognitionKMeansClusters);

            if (segmentation.regions.empty())
                continue;

            BallTypeResult baseline =classifyBallType(segmentation);
            CCAFeatures cca = analyseConnectedComponents(roi, segmentation);
            BallTypeResult hybrid = classifyBallTypeWithCCA(segmentation, cca);

            int number = groundTruth[gtIndex].number;
            int baselineNumber = classifyBallNumber(segmentation, baseline);
            int hybridNumber = classifyBallNumber(segmentation, hybrid); 

            string actual = trueBallType(number);

            bool baselineCorrect = (baseline.type == actual);
            bool hybridCorrect = (hybrid.type == actual);

            bool baselineNumberCorrect = (baselineNumber == number);
            bool hybridNumberCorrect = (hybridNumber == number);

            bool changed = (baseline.type != hybrid.type);

            matched++;
            imageMatched++;

            if (baselineCorrect) {
                baselineCorrectTotal++;
                imageBaselineCorrect++;
            }

            if (hybridCorrect) {
                hybridCorrectTotal++;
                imageHybridCorrect++;
            }

            if (baselineNumberCorrect) {
                imageBaselineNumberCorrect++;
            }

            if (hybridNumberCorrect) {
                imageHybridNumberCorrect++;
            }

            if (changed) {
                changedPredictions++;
                imageChanged++;
            }

            // CCA fixes an existing mistake.
            if (!baselineCorrect && hybridCorrect) {
                correctedByCCA++;
                imageCorrectedByCCA++;
            }

            // CCA changes a correct prediction
            // into an incorrect prediction.
            if (baselineCorrect && !hybridCorrect) {
                worsenedByCCA++;
                imageWorsenedByCCA++;
            }

            // STEP 7: Save CSV results

            csv << name << ","
                << (i + 1) << ","
                << number << ","
                << actual << ","
                << baseline.type << ","
                << hybrid.type << ","
                << (baselineCorrect ? 1 : 0) << ","
                << (hybridCorrect ? 1 : 0) << ","
                << baseline.lightPercent << ","
                << baseline.colouredPercent << ","
                << baseline.darkPercent << ","
                << cca.lightComponents << ","
                << cca.largestLightComponentPercent << ","
                << cca.lightBoundaryCoverage << ","
                << cca.nonLightComponents << ","
                << cca.largestNonLightComponentPercent << ","
                << segmentation.hueDifference << ","
                << segmentation.saturationDifference << ","
                << segmentation.brightnessDifference << ","
                << segmentation.bgrDistance << ","
                << baselineNumber << ","
                << hybridNumber << ","
                << baselineNumberCorrect << ","
                << hybridNumberCorrect << ","
                << "\n";

            Mat isolated = Mat::zeros(roi.image.size(), roi.image.type());
            roi.image.copyTo(isolated, roi.mask);

            Mat isolatedSegmentation;
            if (!segmentation.segmentedImage.empty()) {
                isolatedSegmentation = Mat::zeros(segmentation.segmentedImage.size(), segmentation.segmentedImage.type());
                segmentation.segmentedImage.copyTo(isolatedSegmentation, roi.mask);
            }

            vector<ImagePanel> panels = {
                {"Original", isolated},
                {"Segmented", isolatedSegmentation},
                {"Light mask", cca.lightMask},
                {"Non-light mask", cca.nonLightMask},
                {"Boundary band", cca.boundaryBand}
            };
            for (size_t j = 0; j < segmentation.regions.size(); ++j) {
                panels.push_back({"Cluster " + to_string(j + 1), segmentation.regions[j].mask});
            }

            string prefix = "candidate_" + to_string(i + 1) + "_ball_" + to_string(number);
            filesystem::path diagnosticDir = outputDir / "cca";
            if (!baselineCorrect || !hybridCorrect) {
                diagnosticDir = outputDir / "misclassified";
                prefix += "_true_" + actual + "_baseline_" + baseline.type + "_hybrid_" + hybrid.type;
                for (size_t j = 0; j < cca.lightRegions.size(); ++j) {
                    panels.push_back({"CCA light region " + to_string(j + 1), cca.lightRegions[j]});
                }
                for (size_t j = 0; j < cca.nonLightRegions.size(); ++j) {
                    panels.push_back({"CCA non-light region " + to_string(j + 1), cca.nonLightRegions[j]});
                }
            }
            saveImage(diagnosticDir / (prefix + "_comparison.png"), combineImages(panels));
        }

        finishOutputFile(csv);

        cout << "\n===== " << name << " =====" << endl;
        cout << "Results: " << csvPath.string() << endl;
        cout << "Matched balls: " << imageMatched << endl;
        cout << "Baseline correct: " << imageBaselineCorrect << endl;
        cout << "Hybrid correct: " << imageHybridCorrect << endl;
        cout << "Baseline Number correct: " << imageHybridNumberCorrect << endl;
        cout << "Hybrid Number correct: " << imageHybridNumberCorrect << endl;


        if (imageMatched > 0) {


            cout << "Baseline accuracy: "
                 << double(imageBaselineCorrect) /
                    imageMatched
                 << endl;

            cout << "Hybrid accuracy: "
                 << double(imageHybridCorrect) /
                    imageMatched
                 << endl;
        }

        cout << "Changed predictions: " << imageChanged << endl;

        cout << "Corrected by CCA: " << imageCorrectedByCCA << endl;

        cout << "Worsened by CCA: " << imageWorsenedByCCA << endl;
    }

    // ==============================================
    // STEP 10: Overall results
    // ==============================================

    cout << "\n===== RECOGNITION VALIDATION =====\n";

    cout << "Ground-truth balls: " << totalGT << endl;

    cout << "Matched balls classified: " << matched << endl;

    cout << "Baseline correct: " << baselineCorrectTotal << endl;

    cout << "Hybrid correct: " << hybridCorrectTotal << endl;

    if (matched > 0) {

        cout << fixed << setprecision(4);

        cout << "Baseline accuracy: "
             << double(baselineCorrectTotal) /
                matched
             << endl;

        cout << "Hybrid accuracy: "
             << double(hybridCorrectTotal) /
                matched
             << endl;
    }

    if (totalGT > 0) {

        cout << "Baseline end-to-end recall: "
             << double(baselineCorrectTotal) /
                totalGT
             << endl;

        cout << "Hybrid end-to-end recall: "
             << double(hybridCorrectTotal) /
                totalGT
             << endl;
    }

    cout << "\n===== CCA IMPACT =====\n";

    cout << "Changed predictions: " << changedPredictions << endl;

    cout << "Errors corrected by CCA: " << correctedByCCA << endl;

    cout << "New errors introduced by CCA: " << worsenedByCCA << endl;

    cout << "Net improvement: "
         << correctedByCCA - worsenedByCCA
         << " correctly classified balls"
         << endl;
}
