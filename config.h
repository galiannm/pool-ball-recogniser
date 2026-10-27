#pragma once

#include <opencv2/core.hpp>
#include <string>
#include <vector>

struct HoughParams
{
    double dp = 1.5;
    double minDist = 18.0;
    double param1 = 75.0;
    double param2 = 35.0;
    int minRadius = 15;
    int maxRadius = 30;
};

namespace config
{
    inline const std::string mediaDirectory = "Media/";
    inline const std::string groundTruthPath = "data/ground_truth/BallLocations.csv";
    inline const std::string outputDirectory = "output/";
    inline const std::string tuningImage = "AllBalls.jpeg";

    inline const std::string rawOutputDirectory = outputDirectory + "detect_circles/";
    inline const std::string filteredOutputDirectory = outputDirectory + "detect_circles_filtered/";
    inline const std::string validationOutputDirectory = outputDirectory + "validation/";
    inline const std::string sanityOutputDirectory = outputDirectory + "sanity_check/";
    inline const std::string extractionOutputDirectory = outputDirectory + "recognition/AllBalls/";

    inline const std::string applicationOutputDirectory = outputDirectory + "myApplication/";

    inline constexpr int tableWidth = 1434;
    inline constexpr int tableHeight = 683;
    inline constexpr double matchTolerance = 15.0;
    inline constexpr float pocketRadius = 25.0f;
    inline constexpr int recognitionKMeansClusters = 2;

    inline const std::vector<cv::Point> tableCorners = {
        {337, 35},
        {1570, 21},
        {1888, 451},
        {38, 472}
    };

    inline const std::vector<std::string> validationImages = {
        "Table1.jpeg",
        "Table2.jpeg",
        "Table3.jpeg",
        "Table4.jpeg",
        "Table5.jpeg",
        "Table6.jpeg"
    };

    inline std::vector<std::string> allImages()
    {
        auto images = validationImages;
        images.insert(images.begin(), tuningImage);
        return images;
    }
}
