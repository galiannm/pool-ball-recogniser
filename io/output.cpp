#include "output.h"

#include <opencv2/imgcodecs.hpp>
#include <stdexcept>

void saveImage(const std::filesystem::path& path, const cv::Mat& image)
{
    if (!path.parent_path().empty())
        std::filesystem::create_directories(path.parent_path());

    if (!cv::imwrite(path.string(), image))
        throw std::runtime_error("Could not save image: " + path.string());
}

void finishOutputFile(std::ofstream& file)
{
    file.flush();
    file.close();
    if (!file)
        throw std::runtime_error("Could not finish writing output CSV");
}
