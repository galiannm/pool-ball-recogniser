#include "io/output.h"
#include "config.h"
#include "sanityCheck.h"

#include "evaluation.h"
#include "visualisation/visualisation.h"
#include "geometry/perspectiveTransform.h"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>

using namespace cv;
using namespace std;

namespace fs = std::filesystem;

void runSanityCheck()
{
    const string CSV_PATH = config::groundTruthPath;
    const string OUTPUT_DIR = config::sanityOutputDirectory;

    const auto imageNames = config::allImages();

    fs::create_directories(OUTPUT_DIR);

    cout << "\n===== SANITY CHECK STARTED =====\n";

    for (const string& imageName : imageNames)
    {
        Mat image = imread(config::mediaDirectory + imageName);

        if (image.empty())
        {
            cerr << "Could not open image: " << imageName << endl;
            continue;
        }

        vector<int> ballIds;
        vector<Point2f> groundTruth =
            loadGroundTruth(CSV_PATH, imageName, &ballIds);

        if (groundTruth.empty())
        {
            cerr << "No ground truth found for: " << imageName << endl;
            continue;
        }

        Mat transformedImage = transformImage(image);

        Mat overlay = drawGroundTruthOverlay(
            transformedImage,
            groundTruth,
            ballIds
        );

        string baseName = imageName.substr(
            0, imageName.find_last_of('.')
        );

        string outputPath =
            OUTPUT_DIR + baseName + "_groundtruth_overlay.png";

        saveImage(outputPath, overlay);

        cout << imageName
             << " -> " << groundTruth.size()
             << " ground-truth points saved to "
             << outputPath << endl;
    }

    cout << "===== SANITY CHECK COMPLETE =====\n";
}
