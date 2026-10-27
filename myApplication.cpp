#include "io/output.h"

#include <opencv2/opencv.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "myApplication.h"
#include "config.h"
#include "detection/ballDetection.h"
#include "recognition/extractBall.h"
#include "recognition/kMeansSegmentation.h"
#include "recognition/ballTypeClassifier.h"
#include "geometry/perspectiveTransform.h"
#include "visualisation/visualisation.h"

using namespace cv;
using namespace std;

void MyApplication() {
    HoughParams params;

    for (const string& filename : config::allImages()) {
        const string path = config::mediaDirectory + filename;

        Mat image = imread(path, IMREAD_COLOR);

        if (image.empty()) {
            cout << "Could not open: " << path << endl;
            continue;
        }

        cout << "Processing: " << path << endl;

        /**
         * -------------------------------------------------------
         * =============== STAGE 1: Ball detection ===============
         * -------------------------------------------------------
         */

        DetectionResult result = runDetection(image, params);

        cout << "Before filtering: " << result.rawCircles.size()
            << " | After filtering: " << result.filteredCircles.size()
            << endl;

        Mat circle_image = drawDetectedBalls(result.maskedImage, result.rawCircles);

        Mat circle_image_filtered = drawDetectedBalls(result.maskedImage, result.filteredCircles);

        Mat comparison;
        vconcat(circle_image, circle_image_filtered, comparison);

        namedWindow("Top: original | Bottom: filtered", WINDOW_NORMAL);
        imshow("Top: original | Bottom: filtered", comparison);

        saveImage(filesystem::path(config::applicationOutputDirectory) / "detection" / filename, circle_image);
        saveImage(filesystem::path(config::applicationOutputDirectory) / "detection_filtered" / filename, circle_image_filtered);

        waitKey(0);

        /**
         * -------------------------------------------------------
         * ============ STAGE 2: Ball Type Recognition ===========
         * -------------------------------------------------------
         */

        const vector<Vec3f>& circles = result.filteredCircles;

        vector<KMeansResult> segmentations;
        vector<BallTypeResult> classifications;
        segmentations.reserve(circles.size());
        classifications.reserve(circles.size());

        vector<string> predictedTypes;
        predictedTypes.reserve(circles.size());

        // Transform ball centres into table coordinates.
        vector<Point2f> tableCentres = transformCentres(circles);

        // Output directory for recognition.
        const filesystem::path recognitionDir = filesystem::path(config::applicationOutputDirectory) / "recognition";

        filesystem::create_directories(recognitionDir);

        const string stem = filesystem::path(filename).stem().string();

        ofstream csv(recognitionDir / (stem + "_recognition.csv"));

        if (!csv.is_open()) {
            cerr << "Could not create recognition CSV for " << filename << endl;
            continue;
        }

        csv << "candidate,originalX,originalY,"
            << "tableX,tableY,radius,type,"
            << "lightPercent,colouredPercent,darkPercent\n";

        for (size_t i = 0; i < circles.size(); ++i) {

            const Vec3f& detected = circles[i];

            BallROI roi = extractBall(image, detected);

            string type = "Unknown";
            KMeansResult segmentation;
            BallTypeResult classification;

            if (!roi.image.empty() && !roi.mask.empty()) {

                segmentation = segmentBallKMeans(roi, config::recognitionKMeansClusters);

                if (!segmentation.regions.empty()) {
                    classification = classifyBallType(segmentation);
                    type = classification.type;
                }
            }

            segmentations.push_back(segmentation);
            classifications.push_back(classification);
            predictedTypes.push_back(type);

            csv << (i + 1) << ","
                << detected[0] << ","
                << detected[1] << ","
                << tableCentres[i].x << ","
                << tableCentres[i].y << ","
                << detected[2] << ","
                << type << ","
                << classification.lightPercent << ","
                << classification.colouredPercent << ","
                << classification.darkPercent << "\n";

            cout << "Candidate " << (i + 1)
                 << " | " << type
                 << " | Table centre: ("
                 << tableCentres[i].x << ", "
                 << tableCentres[i].y << ")"
                 << endl;
        }

        finishOutputFile(csv);

        // Draw predicted types on the original image.
        Mat recognitionImage = drawClassifiedBalls(image, circles, predictedTypes);

        namedWindow("Ball Recognition", WINDOW_NORMAL);
        imshow("Ball Recognition", recognitionImage);

        saveImage(recognitionDir / (stem + "_classified.png"), recognitionImage);

        waitKey(0);

        /**
         * -------------------------------------------------------
         * ============ STAGE 3: Ball Number Recognition =========
         * -------------------------------------------------------
         */
        vector<int> predictedNumbers;
        predictedNumbers.reserve(circles.size());

        ofstream numberCSV(recognitionDir / (stem + "_numbers.csv"));

        if (!numberCSV.is_open()) {
            cerr << "Could not create number CSV for " << filename << endl;
            continue;
        }

        numberCSV << "candidate,predictedType,predictedNumber\n";

        for (size_t i = 0; i < circles.size(); ++i) {
            int number = classifyBallNumber(
                segmentations[i],
                classifications[i]
            );

            predictedNumbers.push_back(number);

            numberCSV << (i + 1) << ","
                    << predictedTypes[i] << ","
                    << number << "\n";
        }

        finishOutputFile(numberCSV);

        // Draw and save the numbered image.
        Mat numberedImage = drawClassifiedBalls(
            image, circles, predictedTypes, predictedNumbers
        );

        namedWindow("Ball Numbers", WINDOW_NORMAL);
        imshow("Ball Numbers", numberedImage);

        saveImage(
            recognitionDir / (stem + "_numbered.png"),
            numberedImage
        );

        waitKey(0);
    }

    destroyAllWindows();
}
