// Report-only diagnostics. Reads existing results and writes only report v6/assets.
// Do not define POOL_BALL_STANDALONE: this file has its own entry point.
#include "../myApplication.cpp"

int main() {
    try {
        const std::filesystem::path output = "report v6/assets";
        std::filesystem::create_directories(output);
        const auto image = cv::imread(MEDIA_DIR + TUNNNING_IMG);
        if (image.empty()) throw std::runtime_error("Missing AllBalls reference image");
        const auto model = buildColourReferenceModel(image);

        std::ifstream saved(std::filesystem::path(FINAL_RES_DIR) / "AllBalls/balls.csv");
        if (!saved) throw std::runtime_error("Missing saved AllBalls circles");
        std::string line;
        std::getline(saved, line);
        std::vector<cv::Vec3f> circles;
        while (std::getline(saved, line)) {
            std::stringstream row(line);
            std::vector<std::string> fields;
            std::string value;
            while (std::getline(row, value, ',')) fields.push_back(value);
            circles.emplace_back(std::stof(fields.at(3)), std::stof(fields.at(4)), std::stof(fields.at(5)));
        }
        const auto context = prepareEvaluation(circles, TUNNNING_IMG);
        std::ofstream samples(output / "reference_examples.csv");
        samples << "number,candidate,x,y,radius,bodyPixels,lightPercent,hlsLightPercent,type\n";
        std::map<int, cv::Mat> yellowHistograms;
        for (size_t i = 0; i < circles.size(); ++i) {
            const int g = context.matching.predictionToGroundTruth[i];
            if (g < 0) continue;
            const int number = context.truth[g].number;
            if (number != 0 && number != 1 && number != 9) continue;
            const auto ball = extractBall(image, circles[i]);
            const auto segmentation = segmentBallKMeans(ball, 2);
            const auto type = classifyBallType(segmentation);
            cv::Mat hlsPreview;
            const auto hlsType = pastHLSType(ball, hlsPreview);
            cv::Mat lightMask = cv::Mat::zeros(ball.mask.size(), CV_8UC1);
            for (const auto& region : segmentation.regions) {
                if (region.hsv[2] >= LIGHT_MIN_V && region.hsv[1] <= LIGHT_MAX_S)
                    cv::bitwise_or(lightMask, region.mask, lightMask);
            }
            const double measuredLight = 100.0 * cv::countNonZero(lightMask) / cv::countNonZero(ball.mask);
            if (std::abs(measuredLight - type.lightPercent) > 1e-6)
                throw std::runtime_error("Light mask disagrees with measured coverage");
            const auto mask = histogramBodyMask(ball, model.options);
            cv::Mat body = cv::Mat::zeros(ball.image.size(), CV_8UC3);
            ball.image.copyTo(body, mask);
            const auto prefix = "reference_" + std::to_string(number);
            saveImage(output / (prefix + "_original.png"), ball.image);
            saveImage(output / (prefix + "_mask.png"), ball.mask);
            saveImage(output / (prefix + "_kmeans.png"), segmentation.segmentedImage);
            saveImage(output / (prefix + "_hls.png"), hlsPreview);
            saveImage(output / (prefix + "_light.png"), lightMask);
            saveImage(output / (prefix + "_body.png"), body);
            samples << number << ',' << i + 1 << ',' << circles[i][0] << ','
                    << circles[i][1] << ',' << circles[i][2] << ','
                    << cv::countNonZero(mask) << ',' << type.lightPercent << ','
                    << hlsType.lightPercent << ',' << type.type << '\n';
            if (number != 0)
                yellowHistograms[number] = buildColourHistogram(makeColourFeatures(ball, model.options), mask, model.options);
        }
        finishOutputFile(samples);
        if (yellowHistograms.size() != 2)
            throw std::runtime_error("Missing yellow reference partner");
        const cv::Mat averaged = (yellowHistograms.at(1) + yellowHistograms.at(9)) * 0.5;
        if (cv::norm(averaged, model.histograms[0], cv::NORM_INF) > 1e-6)
            throw std::runtime_error("Saved crops do not reproduce the yellow reference");

        std::ofstream histograms(output / "yellow_histograms.csv");
        histograms << "hueBin,saturationBin,valueBin,solid1,stripe9,referenceYellow\n";
        histograms << std::setprecision(10);
        const auto& options = model.options;
        for (int h = 0; h < options.hueBins; ++h) {
            for (int s = 0; s < options.saturationBins; ++s) {
                for (int v = 0; v < options.toneBins; ++v) {
                    const int index[] = {h, s, v};
                    histograms << h << ',' << s << ',' << v << ','
                               << yellowHistograms.at(1).at<float>(index) << ','
                               << yellowHistograms.at(9).at<float>(index) << ','
                               << model.histograms[0].at<float>(index) << '\n';
                }
            }
        }
        finishOutputFile(histograms);

        std::ofstream references(output / "reference_construction.csv");
        references << "colour,numberedExamples,crops\n";
        for (int colour = 1; colour <= 8; ++colour) {
            const int expected = colour == 8 ? 1 : 2;
            if (model.examples[colour - 1] != expected)
                throw std::runtime_error("Unexpected reference example count");
            references << bodyColourName(colour) << ',' << colour;
            if (colour != 8) references << '/' << colour + 8;
            references << ',' << model.examples[colour - 1] << '\n';
        }
        finishOutputFile(references);
        std::cout << "Reference montage exported; all eight example counts and yellow averaging verified.\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
