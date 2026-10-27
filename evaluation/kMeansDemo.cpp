#include "kMeansDemo.h"

#include "recognition/extractBall.h"
#include "recognition/kMeansSegmentation.h"
#include "recognition/ballTypeClassifier.h"
#include "recognition/connectedComponentAnalysis.h"
#include "detection/ballDetection.h"
#include "evaluation/evaluation.h"
#include "geometry/perspectiveTransform.h"
#include "config.h"
#include "io/output.h"
#include "visualisation/combineImages.h"

#include <opencv2/opencv.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>

using namespace cv;
using namespace std;

void testKMeansSegmentation() {

    Mat image = imread(config::mediaDirectory + config::tuningImage);

    if (image.empty()) {
        cerr << "Could not load tuning image" << endl;
        return;
    }

    // detect balls
    HoughParams params;

    vector<Vec3f> circles = detectAndFilterBalls(image, params);

    // Match detections to ground truth
    vector<LabelledBall> groundTruth =
        loadLabelledGroundTruth(
            config::groundTruthPath,
            config::tuningImage
        );

    vector<Point2f> gtCentres;

    for (const LabelledBall& ball : groundTruth) {
        gtCentres.push_back(ball.centre);
    }

    vector<Point2f> predictions = transformCentres(circles);

    EvaluationResult evaluation = evaluateDetections(
        predictions,
        gtCentres,
        config::matchTolerance
    );    

    // prepare output files
    string outputDir = config::extractionOutputDirectory + "kmeans/";

    filesystem::create_directories(outputDir);

    ofstream csv(outputDir + "kmeans_features.csv");

    if (!csv.is_open()) {
        cerr << "Could not open K-means CSV" << endl;
        return;
    }

    ofstream ccaCSV(outputDir + "cca_features.csv");

    if (!ccaCSV.is_open()) {
        cerr << "Could not open CCA CSV" << endl;
        return;
    }

    ccaCSV
        << "candidate,actualBallNumber,"
        << "lightPercent,colouredPercent,darkPercent,"
        << "lightComponents,largestLightComponentPercent,"
        << "lightBoundaryCoverage,"
        << "nonLightComponents,largestNonLightComponentPercent,"
        << "predictedType\n";

    csv << fixed << setprecision(6);

    // CSV header

    csv << "candidate,ballNumber,k,x,y,radius";

    for (int j = 1; j <= 5; ++j) {
        string prefix = "Cluster " + to_string(j);

        csv << "," << prefix << " (%)"
            << ",\"" << prefix << " (B,G,R)\""
            << ",\"" << prefix << " (H,S,V)\"";
    }

    csv << "\n";

    ofstream classificationCSV(outputDir + "ball_type_predictions.csv");

    if (!classificationCSV.is_open()) {
        cerr << "Could not open classification CSV" << endl;
        return;
    }

    classificationCSV
        << "candidate,actualBallNumber,predictedBallNumber,predictedType,"
        << "lightPercent,colouredPercent,darkPercent\n";

    // Analyse all detected balls
    for (size_t i = 0; i < circles.size(); ++i) {

        // Find the real ball number using ground truth
        int gtIndex = evaluation.predictionToGroundTruth[i];

        int ballNumber = -1;

        if (gtIndex >= 0 &&
            gtIndex < static_cast<int>(groundTruth.size())) {
            ballNumber = groundTruth[gtIndex].number;
        }

        BallROI ball = extractBall(image, circles[i]);

        if (ball.image.empty() || ball.mask.empty())
            continue;

        vector<ImagePanel> panels = {{"Original", ball.image}};

        for (int k = 2; k <= 5; ++k) {

            KMeansResult result = segmentBallKMeans(ball, k);

            if (result.regions.empty())
                continue;

            if (k == 2) {

                BallTypeResult kMeansPrediction = classifyBallType(result);
                CCAFeatures cca = analyseConnectedComponents(ball, result);
                BallTypeResult hybridPrediction = classifyBallTypeWithCCA(result, cca);
                int predictedBallNumber = classifyBallNumber(result, kMeansPrediction);

                classificationCSV
                    << (i + 1) << ","
                    << ballNumber << ","
                    << predictedBallNumber << ","
                    << kMeansPrediction.type << ","
                    << kMeansPrediction.lightPercent << ","
                    << kMeansPrediction.colouredPercent << ","
                    << kMeansPrediction.darkPercent << "\n";

                ccaCSV
                    << (i + 1) << ","
                    << ballNumber << ","
                    << hybridPrediction.lightPercent << ","
                    << hybridPrediction.colouredPercent << ","
                    << hybridPrediction.darkPercent << ","
                    << cca.lightComponents << ","
                    << cca.largestLightComponentPercent << ","
                    << cca.lightBoundaryCoverage << ","
                    << cca.nonLightComponents << ","
                    << cca.largestNonLightComponentPercent << ","
                    << hybridPrediction.type << "\n";

                cout << "Candidate " << (i + 1)
                    << " | Ball number: " << ballNumber 
                    << " | K-means: " << kMeansPrediction.type
                    << " | K-means + CCA: " << hybridPrediction.type
                    << " | Light components: " << cca.lightComponents
                    << " | Largest light: " << cca.largestLightComponentPercent << "%"
                    << " | Light boundary: " << cca.lightBoundaryCoverage << "%"
                    << " | Non-light components: " << cca.nonLightComponents
                    << endl;
            }

            const string label = "K=" + to_string(k);
            panels.push_back({label + " segmented", result.segmentedImage});
            for (size_t j = 0; j < result.regions.size(); ++j) {
                panels.push_back({label + " cluster " + to_string(j + 1), result.regions[j].mask});
            }

            // CSV: ball information
            csv << (i + 1) << ","
                << ballNumber << ","
                << k << ","
                << circles[i][0] << ","
                << circles[i][1] << ","
                << circles[i][2];

            // Quote colour tuples so their commas stay within one CSV cell.
            // Leave unused cluster columns empty.
            for (int j = 0; j < 5; ++j) {

                if (j < static_cast<int>(result.regions.size())) {

                    const KMeansRegion& r = result.regions[j];

                    csv << "," << r.percentage
                        << ",\"(" << static_cast<int>(r.colour[0])
                        << ", " << static_cast<int>(r.colour[1])
                        << ", " << static_cast<int>(r.colour[2]) << ")\""
                        << ",\"(" << static_cast<int>(r.hsv[0])
                        << ", " << static_cast<int>(r.hsv[1])
                        << ", " << static_cast<int>(r.hsv[2]) << ")\"";

                } else {
                    csv << ",,,";
                }
            }

            csv << "\n";

            cout << "Candidate " << (i + 1)
                << " | Ball " << ballNumber
                << " | K=" << k;

            for (size_t j = 0;
                 j < result.regions.size(); ++j) {

                cout << " | C" << (j + 1)
                     << ": "
                     << fixed << setprecision(1)
                     << result.regions[j].percentage
                     << "%";
            }

            cout << endl;
        }
        if (panels.size() > 1) {
            saveImage(outputDir + "candidate_" + to_string(i + 1) + "_comparison.png",
                      combineImages(panels));
        }
    }

    csv.close();

    cout << "\nK-means experiment complete."
         << endl;

    cout << "Results: " << outputDir << endl;
    finishOutputFile(classificationCSV);
    ccaCSV.close();
}
