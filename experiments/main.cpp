#include "radiusPadding.h"
#include "config.h"

#include <exception>
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    if (argc > 3 || (argc > 1 && std::string(argv[1]) == "--help")) {
        std::cout << "Usage: BallExperiments [image-path|--all] [output-directory]\n"
                  << "Defaults: Media/AllBalls.jpeg output/experiments/radius_padding\n";
        return argc > 3 ? 1 : 0;
    }

    try {
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
