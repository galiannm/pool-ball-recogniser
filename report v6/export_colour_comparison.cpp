// Report-only evidence: reads saved detections and writes only report v6/assets.
#include "../myApplication.h"

// Keep the report exporter usable while the working application omits its
// validation helper. A restored, non-template application helper takes precedence.
template<class Options> void validate(const Options& p) {
    if (p.hueBins < 2 || p.saturationBins < 2 || p.toneBins < 2 || p.erosionKernel < 1 ||
        p.erosionKernel % 2 == 0 || p.minimumBodyPixels < 1 || p.lightMinV < 0 ||
        p.lightMinV > 255 || p.lightMaxS < 0 || p.lightMaxS > 255 ||
        p.neutralChroma < 0 || p.neutralChroma > 255)
        throw std::invalid_argument("Invalid ball histogram settings");
}
#include "../myApplication.cpp"

std::vector<cv::Vec3f> savedCircles(const std::string& name) {
    std::ifstream input(std::filesystem::path(FINAL_RES_DIR) / name / "balls.csv");
    if (!input) throw std::runtime_error("Missing saved circles: " + name);
    std::vector<cv::Vec3f> circles;
    std::string line;
    std::getline(input, line);
    while (std::getline(input, line)) {
        std::stringstream row(line);
        std::vector<std::string> values;
        std::string field;
        while (std::getline(row, field, ',')) values.push_back(field);
        circles.emplace_back(std::stof(values.at(3)), std::stof(values.at(4)), std::stof(values.at(5)));
    }
    return circles;
}

int main() {
    try {
        const std::filesystem::path output = "report v6/assets";
        const auto reference = cv::imread(MEDIA_DIR + TUNNNING_IMG);
        const auto model = buildColourReferenceModel(reference);
        const auto options = model.options;
        const auto trainingCircles = savedCircles("AllBalls");
        const auto trainingContext = prepareEvaluation(trainingCircles, TUNNNING_IMG);
        std::map<int, cv::Mat> trainingHistograms;
        for (size_t i = 0; i < trainingCircles.size(); ++i) {
            const int g = trainingContext.matching.predictionToGroundTruth[i];
            if (g < 0 || trainingContext.truth[g].number == 0) continue;
            const int number = trainingContext.truth[g].number;
            const auto ball = extractBall(reference, trainingCircles[i]);
            const auto mask = histogramBodyMask(ball, options);
            cv::Mat body = cv::Mat::zeros(ball.image.size(), CV_8UC3);
            ball.image.copyTo(body, mask);
            saveImage(output / ("colour_reference_" + std::to_string(number) + "_body.png"), body);
            trainingHistograms[number] = buildColourHistogram(makeColourFeatures(ball, options), mask, options);
        }
        std::ofstream models(output / "colour_reference_histograms.csv");
        models << "hueBin,saturationBin,valueBin";
        for (int number = 1; number <= 15; ++number) models << ",ball" << number;
        for (int colour = 1; colour <= 8; ++colour) models << ',' << bodyColourName(colour);
        models << '\n' << std::setprecision(10);
        for (int h = 0; h < options.hueBins; ++h)
            for (int s = 0; s < options.saturationBins; ++s)
                for (int v = 0; v < options.toneBins; ++v) {
                    const int index[] = {h, s, v};
                    models << h << ',' << s << ',' << v;
                    for (int number = 1; number <= 15; ++number)
                        models << ',' << trainingHistograms.at(number).at<float>(index);
                    for (const auto& histogram : model.histograms) models << ',' << histogram.at<float>(index);
                    models << '\n';
                }
        finishOutputFile(models);

        std::ofstream evidence(output / "colour_method_evidence.csv");
        evidence << "image,candidate,trueNumber,trueColour,kmeansColour,histogramColour,meanH,meanS,meanV,bodyPixels";
        for (int colour = 1; colour <= 8; ++colour) evidence << ',' << bodyColourName(colour);
        evidence << '\n' << std::setprecision(10);
        std::ofstream queries(output / "colour_gain_histograms.csv");
        queries << "image,candidate,hueBin,saturationBin,valueBin,mass\n" << std::setprecision(10);
        int matched = 0, kmeansCorrect = 0, histogramCorrect = 0, gained = 0, lost = 0;
        for (const auto& filename : VAL_IMG) {
            const std::string name = std::filesystem::path(filename).stem().string();
            const auto image = cv::imread(MEDIA_DIR + filename);
            const auto circles = savedCircles(name);
            const auto context = prepareEvaluation(circles, filename);
            for (size_t i = 0; i < circles.size(); ++i) {
                const int g = context.matching.predictionToGroundTruth[i];
                if (g < 0 || context.truth[g].number == 0) continue;
                const int number = context.truth[g].number;
                const int trueColour = number <= 8 ? number : number - 8;
                const auto ball = extractBall(image, circles[i]);
                const auto segmentation = segmentBallKMeans(ball, 2);
                const int previous = pastHueColour(segmentation);
                const auto current = scoreBallColours(ball, model);
                const bool wasCorrect = previous == trueColour;
                const bool nowCorrect = current.bestColour == trueColour;
                ++matched; kmeansCorrect += wasCorrect; histogramCorrect += nowCorrect;
                gained += !wasCorrect && nowCorrect; lost += wasCorrect && !nowCorrect;
                cv::Vec3b meanHSV(0, 0, 0);
                for (const auto& region : segmentation.regions) {
                    if (region.percentage < 2 || (region.hsv[2] >= 170 && region.hsv[1] <= 165)) continue;
                    meanHSV = region.hsv;
                    break; // Regions are sorted by area, as in pastHueColour.
                }
                evidence << name << ',' << i+1 << ',' << number << ',' << trueColour << ','
                         << previous << ',' << current.bestColour << ',' << int(meanHSV[0]) << ','
                         << int(meanHSV[1]) << ',' << int(meanHSV[2]) << ',' << current.pixels;
                for (const double score : current.similarity) evidence << ',' << score;
                evidence << '\n';
                if (!wasCorrect && nowCorrect) {
                    const auto prefix = "colour_gain_" + name + "_" + std::to_string(i+1);
                    saveImage(output / (prefix + "_original.png"), ball.image);
                    cv::Mat body = cv::Mat::zeros(ball.image.size(), CV_8UC3);
                    ball.image.copyTo(body, current.bodyMask);
                    saveImage(output / (prefix + "_body.png"), body);
                    const auto histogram = buildColourHistogram(makeColourFeatures(ball, options), current.bodyMask, options);
                    for (int h = 0; h < options.hueBins; ++h)
                        for (int s = 0; s < options.saturationBins; ++s)
                            for (int v = 0; v < options.toneBins; ++v) {
                                const int index[] = {h, s, v};
                                queries << name << ',' << i+1 << ',' << h << ',' << s << ',' << v
                                        << ',' << histogram.at<float>(index) << '\n';
                            }
                    std::cout << name << " candidate " << i+1 << " (true " << number << "): "
                              << bodyColourName(previous) << " -> " << bodyColourName(current.bestColour)
                              << "; mean HSV " << meanHSV << '\n';
                }
            }
        }
        finishOutputFile(evidence);
        finishOutputFile(queries);
        if (matched != 80 || kmeansCorrect != 65 || histogramCorrect != 73)
            throw std::runtime_error("Current colour comparison does not reproduce saved report totals");
        std::cout << "Matched " << matched << "; representative colour " << kmeansCorrect
                  << "; histogram " << histogramCorrect << "; gained " << gained << "; lost " << lost << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
