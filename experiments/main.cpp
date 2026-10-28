#include "radiusPadding.h"
#include "recognitionGridSearch.h"
#include "ccaGridSearch.h"
#include "watershedDetection.h"
#include "segmentationGuidedDetection.h"
#include "config.h"

#include <exception>
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    if (argc > 3 || (argc > 1 && std::string(argv[1]) == "--help")) {
        std::cout << "Usage: BallExperiments [image-path|--all] [output-directory]\n"
                  << "       BallExperiments --recognition-grid [output-directory]\n"
                  << "       BallExperiments --cca-grid [output-directory]\n"
                  << "       BallExperiments --watershed [output-directory]\n"
                  << "       BallExperiments --segmentation-hough [output-directory]\n"
                  << "Defaults: Media/AllBalls.jpeg output/experiments/radius_padding\n";
        return argc > 3 ? 1 : 0;
    }

    try {
        if (argc > 1 && std::string(argv[1]) == "--segmentation-hough") {
            experiments::runSegmentationGuidedExperiment(argc > 2
                ? argv[2] : "output/experiments/segmentation_guided_detection");
            return 0;
        }
        if (argc > 1 && std::string(argv[1]) == "--watershed") {
            experiments::runWatershedDetectionExperiment(argc > 2
                ? argv[2] : "output/experiments/watershed_detection");
            return 0;
        }
        if (argc > 1 && std::string(argv[1]) == "--cca-grid") {
            experiments::runCCAGridSearch(argc > 2
                ? argv[2] : "output/experiments/cca_grid_search");
            return 0;
        }
        if (argc > 1 && std::string(argv[1]) == "--recognition-grid") {
            experiments::runRecognitionGridSearch(argc > 2
                ? argv[2] : "output/experiments/recognition_grid_search");
            return 0;
        }
        const std::filesystem::path output = argc > 2
            ? argv[2] : "output/experiments/radius_padding";
        if (argc > 1 && std::string(argv[1]) == "--all") {
            for (const auto& filename : config::allImages())
                experiments::runRadiusPaddingExperiment(
                    std::filesystem::path(config::mediaDirectory) / filename, output);
        } else {
            experiments::runRadiusPaddingExperiment(argc > 1
                ? std::filesystem::path(argv[1])
                : std::filesystem::path(config::mediaDirectory) / config::tuningImage,
                output);
        }
    } catch (const std::exception& error) {
        std::cerr << "Experiment failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
