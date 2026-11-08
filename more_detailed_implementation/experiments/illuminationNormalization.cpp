// Reuse the current final application, rather than the older modular classifiers.
// POOL_BALL_STANDALONE is deliberately not defined for this executable.
#include "../../myApplication.cpp"

namespace lighting_experiment {

struct Correction {
    cv::Mat image;
    cv::Vec3d gains{1, 1, 1};
    int neutralPixels = 0;
    bool applied = false;
};

Correction whiteBalance(const cv::Mat& image) {
    cv::Mat table = cv::Mat::zeros(image.size(), CV_8UC1);
    cv::fillConvexPoly(table, TABLE_CORNERS, cv::Scalar(255));
    Correction result;
    cv::Vec3d sum{0, 0, 0};
    // Fixed before evaluation: bright, unsaturated, approximately neutral pixels.
    // This avoids estimating a grey-world mean from the predominantly green cloth.
    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            if (!table.at<uchar>(y, x)) continue;
            const auto pixel = image.at<cv::Vec3b>(y, x);
            const int high = std::max({pixel[0], pixel[1], pixel[2]});
            const int low = std::min({pixel[0], pixel[1], pixel[2]});
            if (high < 80 || high > 250 || double(high - low) / high > 0.25)
                continue;
            sum += cv::Vec3d(pixel);
            ++result.neutralPixels;
        }
    }
    result.image = image.clone();
    if (result.neutralPixels < 100) return result;

    const cv::Vec3d mean = sum / result.neutralPixels;
    const double target = (mean[0] + mean[1] + mean[2]) / 3.0;
    for (int c = 0; c < 3; ++c)
        result.gains[c] = std::clamp(target / mean[c], 0.5, 2.0);
    for (int y = 0; y < image.rows; ++y)
        for (int x = 0; x < image.cols; ++x)
            for (int c = 0; c < 3; ++c)
                result.image.at<cv::Vec3b>(y, x)[c] = cv::saturate_cast<uchar>(
                    image.at<cv::Vec3b>(y, x)[c] * result.gains[c]);
    result.applied = true;
    return result;
}

Correction claheLightness(const cv::Mat& image) {
    cv::Mat lab;
    cv::cvtColor(image, lab, cv::COLOR_BGR2Lab);
    std::vector<cv::Mat> channels;
    cv::split(lab, channels);
    cv::createCLAHE(2.0, cv::Size(8, 8))->apply(channels[0], channels[0]);
    cv::merge(channels, lab);
    Correction result;
    cv::cvtColor(lab, result.image, cv::COLOR_Lab2BGR);
    result.applied = true;
    return result;
}

// Only the colour branch changes. White-coverage evidence, including the +1
// analysis of originally white crops, is always measured on the raw photograph.
std::vector<int> identities(
    const cv::Mat& raw, const cv::Mat& corrected,
    const std::vector<cv::Vec3f>& circles,
    const std::vector<BallROI>& balls,
    const std::vector<BallTypeResult>& types,
    const std::vector<BallColourScores>& colours,
    const BallColourModel& model) {

    std::vector<Candidate> original;
    const auto centres = transformCentres(circles);
    for (size_t i = 0; i < circles.size(); ++i) {
        Candidate c;
        c.circle = circles[i];
        c.tableCentre = centres[i];
        c.roi = balls[i];
        c.baselineType = types[i];
        c.colours = colours[i];
        identityEvidence(c);
        original.push_back(c);
    }
    auto candidates = original;
    for (size_t i = 0; i < candidates.size(); ++i) {
        if (original[i].baselineType.type != "White") continue;
        auto circle = circles[i];
        circle[2] += 1;
        candidates[i].roi = extractBall(raw, circle);
        candidates[i].baselineType = classifyBallType(
            segmentBallKMeans(candidates[i].roi, 2));
        candidates[i].colours = scoreBallColours(extractBall(corrected, circle), model);
        identityEvidence(candidates[i]);
    }
    const int cue = selectAmbiguousCue(original);
    reserveCueIdentity(candidates, cue);
    auto numbers = assignUnique(candidates);
    if (cue >= 0 && numbers[cue] != 0)
        throw std::runtime_error("Cue reservation failed in lighting experiment");
    return numbers;
}

struct Counts {
    int gt = 0, tp = 0, fp = 0, fn = 0;
    int colourGT = 0, colourMatched = 0, colourCorrect = 0;
    int identityCorrect = 0, unknown = 0, numberedFP = 0;
    int identityFixed = 0, identityLost = 0, colourFixed = 0, colourLost = 0;
    Counts& operator+=(const Counts& o) {
        gt += o.gt; tp += o.tp; fp += o.fp; fn += o.fn;
        colourGT += o.colourGT; colourMatched += o.colourMatched;
        colourCorrect += o.colourCorrect; identityCorrect += o.identityCorrect;
        unknown += o.unknown; numberedFP += o.numberedFP;
        identityFixed += o.identityFixed; identityLost += o.identityLost;
        colourFixed += o.colourFixed; colourLost += o.colourLost;
        return *this;
    }
};

double fraction(int a, int b) { return b ? double(a) / b : 0; }

void writeCounts(std::ostream& out, const Counts& c) {
    out << c.gt << ',' << c.tp << ',' << c.fp << ',' << c.fn << ','
        << c.colourGT << ',' << c.colourMatched << ',' << c.colourCorrect << ','
        << c.identityCorrect << ',' << c.unknown << ',' << c.numberedFP << ','
        << fraction(c.colourCorrect, c.colourMatched) << ','
        << fraction(c.identityCorrect, c.tp) << ','
        << fraction(c.identityCorrect, c.gt) << ','
        << c.identityFixed << ',' << c.identityLost << ','
        << c.colourFixed << ',' << c.colourLost << '\n';
}

int run(const std::filesystem::path& directory) {
    std::filesystem::create_directories(directory);
    const std::array<std::string, 3> names = {"baseline", "white_balance", "clahe_lab_l"};
    auto metrics = std::ofstream(directory / "metrics.csv");
    auto predictions = std::ofstream(directory / "predictions.csv");
    auto changes = std::ofstream(directory / "changes.csv");
    auto corrections = std::ofstream(directory / "corrections.csv");
    if (!metrics || !predictions || !changes || !corrections)
        throw std::runtime_error("Cannot create illumination experiment CSVs");
    metrics << "split,image,method,groundTruth,TP,FP,FN,colourGT,colourMatched,"
               "correctColours,correctIdentities,unknown,numberedFalsePositives,"
               "colourAccuracy,identityAccuracy,endToEndAccuracy,identitiesFixed,"
               "identitiesLost,coloursFixed,coloursLost\n";
    predictions << "image,method,candidate,trueNumber,predictedNumber,bestColour,"
                   "colourCorrect,identityCorrect,status\n";
    changes << "image,method,candidate,trueNumber,baselineNumber,predictedNumber,"
               "baselineColour,predictedColour,identityOutcome,colourOutcome\n";
    corrections << "image,method,neutralPixels,gainB,gainG,gainR,applied\n";
    metrics << std::fixed << std::setprecision(9);

    // The same raw detections and CSV labels select all reference crops.
    const auto reference = cv::imread(MEDIA_DIR + TUNNNING_IMG);
    if (reference.empty()) throw std::runtime_error("Missing reference image");
    const auto referenceCircles = filterPocketDetections(
        detectBalls(maskTable(reference), HoughParams{}));
    const auto referenceContext = prepareEvaluation(referenceCircles, TUNNNING_IMG);
    const std::array<Correction, 3> correctedReference = {
        Correction{reference}, whiteBalance(reference), claheLightness(reference)};
    std::array<BallColourModel, 3> models;
    for (size_t method = 0; method < names.size(); ++method) {
        std::vector<LabelledColourSample> samples;
        for (size_t g = 0; g < referenceContext.truth.size(); ++g) {
            const auto& truth = referenceContext.truth[g];
            if (truth.number == 0) continue;
            const int p = referenceContext.matching.groundTruthToPrediction[g];
            if (p < 0) throw std::runtime_error("Missing numbered reference detection");
            samples.push_back({truth.number,
                extractBall(correctedReference[method].image, referenceCircles[p])});
        }
        models[method] = buildReferenceHistograms(samples);
        cv::FileStorage modelFile((directory / (names[method] + "_model.yml")).string(),
                                  cv::FileStorage::WRITE);
        if (!modelFile.isOpened()) throw std::runtime_error("Cannot save reference model");
        for (size_t c = 0; c < 8; ++c)
            modelFile << ("colour" + std::to_string(c + 1)) << models[method].histograms[c];
    }
    // Baseline references must exactly match the current application's model.
    const auto productionModel = buildColourReferenceModel(reference);
    for (size_t c = 0; c < 8; ++c)
        if (cv::norm(models[0].histograms[c], productionModel.histograms[c], cv::NORM_INF) != 0)
            throw std::runtime_error("Baseline reference-model parity failed");

    std::array<Counts, 3> totals;
    for (const auto& filename : getAllImages()) {
        const auto raw = cv::imread(MEDIA_DIR + filename);
        if (raw.empty()) throw std::runtime_error("Missing image " + filename);
        const auto stem = std::filesystem::path(filename).stem();
        const auto circles = filterPocketDetections(detectBalls(maskTable(raw), HoughParams{}));
        const auto context = prepareEvaluation(circles, filename);
        std::vector<BallROI> balls;
        std::vector<BallTypeResult> types;
        for (const auto& circle : circles) {
            balls.push_back(extractBall(raw, circle));
            types.push_back(classifyBallType(segmentBallKMeans(balls.back(), 2)));
        }
        const std::array<Correction, 3> corrected = {
            Correction{raw}, whiteBalance(raw), claheLightness(raw)};
        std::array<std::vector<BallColourScores>, 3> allColours;
        std::array<std::vector<int>, 3> allNumbers;
        std::vector<ImagePanel> colourTables, identityTables, changedCrops;

        for (size_t method = 0; method < names.size(); ++method) {
            const auto& correction = corrected[method];
            corrections << filename << ',' << names[method] << ',' << correction.neutralPixels
                        << ',' << correction.gains[0] << ',' << correction.gains[1]
                        << ',' << correction.gains[2] << ',' << correction.applied << '\n';
            for (const auto& circle : circles)
                allColours[method].push_back(scoreBallColours(
                    extractBall(correction.image, circle), models[method]));
            allNumbers[method] = identities(raw, correction.image, circles, balls,
                                            types, allColours[method], models[method]);
            const auto imageDirectory = directory / names[method] / stem;
            if (method == 0) {
                const auto production = runIdentityRecognition(raw, balls, circles, types,
                    allColours[0], models[0], imageDirectory, filename, context);
                if (production.numbers != allNumbers[0])
                    throw std::runtime_error("Baseline identity parity failed: " + filename);
            } else {
                std::vector<std::string> finalTypes;
                for (int n : allNumbers[method]) finalTypes.push_back(identityType(n));
                saveFinalResults(raw, imageDirectory, circles, transformCentres(circles),
                                 finalTypes, allNumbers[method]);
            }
            const auto colourEvaluation = evaluateBallColours(allColours[method], context);
            const auto identityEvaluation = evaluateBallIdentities(allNumbers[method], context);
            Counts counts;
            counts.gt = int(context.truth.size()); counts.tp = context.matching.TP;
            counts.fp = context.matching.FP; counts.fn = context.matching.FN;
            counts.colourGT = colourEvaluation.groundTruthCount;
            counts.colourMatched = colourEvaluation.matched;
            counts.colourCorrect = colourEvaluation.correct;
            counts.identityCorrect = identityEvaluation.correct;
            counts.unknown = identityEvaluation.unknown;
            for (size_t i = 0; i < circles.size(); ++i) {
                const int g = context.matching.predictionToGroundTruth[i];
                const int number = allNumbers[method][i];
                const int colour = allColours[method][i].bestColour;
                predictions << filename << ',' << names[method] << ',' << i+1 << ',';
                if (g < 0) {
                    counts.numberedFP += number >= 0;
                    predictions << ',' << number << ',' << colour << ",,,false_positive\n";
                    continue;
                }
                const int truth = context.truth[g].number;
                const int trueColour = truth <= 8 ? truth : truth - 8;
                const bool idCorrect = number == truth;
                const bool baseIDCorrect = allNumbers[0][i] == truth;
                const bool colourCorrect = truth != 0 && colour == trueColour;
                const bool baseColourCorrect = truth != 0 && allColours[0][i].bestColour == trueColour;
                counts.identityFixed += idCorrect && !baseIDCorrect;
                counts.identityLost += !idCorrect && baseIDCorrect;
                counts.colourFixed += colourCorrect && !baseColourCorrect;
                counts.colourLost += !colourCorrect && baseColourCorrect;
                predictions << truth << ',' << number << ',' << colour << ',';
                if (truth != 0) predictions << colourCorrect;
                predictions << ',' << idCorrect << ",matched\n";
                if (method > 0 && (number != allNumbers[0][i] || colour != allColours[0][i].bestColour)) {
                    changes << filename << ',' << names[method] << ',' << i+1 << ',' << truth
                            << ',' << allNumbers[0][i] << ',' << number << ','
                            << allColours[0][i].bestColour << ',' << colour << ','
                            << (idCorrect && !baseIDCorrect ? "corrected" : !idCorrect && baseIDCorrect ? "regressed" : "unchanged_correctness")
                            << ',' << (truth == 0 ? "cue_excluded" : colourCorrect && !baseColourCorrect ? "corrected" : !colourCorrect && baseColourCorrect ? "regressed" : "unchanged_correctness") << '\n';
                }
            }
            metrics << (filename == TUNNNING_IMG ? "reference" : "development")
                    << ',' << filename << ',' << names[method] << ',';
            writeCounts(metrics, counts);
            if (filename != TUNNNING_IMG) totals[method] += counts;
            colourTables.push_back({names[method], drawBallColours(raw, circles, allColours[method])});
            identityTables.push_back({names[method], cv::imread((imageDirectory / "original_numbered.png").string())});
        }
        for (size_t i = 0; i < circles.size(); ++i) {
            const int g = context.matching.predictionToGroundTruth[i];
            const bool changed = allNumbers[0][i] != allNumbers[1][i] || allNumbers[0][i] != allNumbers[2][i]
                || allColours[0][i].bestColour != allColours[1][i].bestColour
                || allColours[0][i].bestColour != allColours[2][i].bestColour;
            if (!changed && (g < 0 || context.truth[g].number != 10)) continue;
            for (size_t m = 0; m < 3; ++m) {
                const auto roi = extractBall(corrected[m].image, circles[i]);
                cv::Mat enlarged;
                cv::resize(roi.image, enlarged, {144, 144}, 0, 0, cv::INTER_NEAREST);
                const auto label = names[m] + " candidate " + std::to_string(i+1)
                    + " true " + (g < 0 ? std::string("FP") : std::to_string(context.truth[g].number))
                    + " ID " + std::to_string(allNumbers[m][i])
                    + " colour " + bodyColourName(allColours[m][i].bestColour);
                changedCrops.push_back({label, enlarged});
            }
        }
        const auto viewDirectory = directory / "comparisons" / stem;
        saveImage(viewDirectory / "colour_tables.png", combineImages(colourTables, 1));
        saveImage(viewDirectory / "identity_tables.png", combineImages(identityTables, 1));
        const auto cropGrid = combineImages(changedCrops, 3);
        if (!cropGrid.empty()) saveImage(viewDirectory / "changed_candidates.png", cropGrid);
    }
    for (size_t m = 0; m < names.size(); ++m) {
        metrics << "development,ALL," << names[m] << ',';
        writeCounts(metrics, totals[m]);
        std::cout << names[m] << ": colour=" << totals[m].colourCorrect << '/'
                  << totals[m].colourMatched << " identities=" << totals[m].identityCorrect
                  << '/' << totals[m].gt << " fixed=" << totals[m].identityFixed
                  << " lost=" << totals[m].identityLost << '\n';
    }
    // Fail loudly if the control no longer reproduces the final report's baseline.
    if (totals[0].tp != 86 || totals[0].fp != 2 || totals[0].fn != 4 ||
        totals[0].colourCorrect != 73 || totals[0].identityCorrect != 80)
        throw std::runtime_error("Baseline has changed; do not compare against old report totals");
    finishOutputFile(metrics); finishOutputFile(predictions);
    finishOutputFile(changes); finishOutputFile(corrections);
    return 0;
}
} // namespace lighting_experiment

int main(int argc, char* argv[]) {
    try {
        if (argc > 2) throw std::invalid_argument("Usage: IlluminationExperiments [output-directory]");
        std::filesystem::current_path(POOL_EXPERIMENT_PROJECT_ROOT);
        const std::filesystem::path output = argc == 2 ? argv[1] :
            "more_detailed_implementation/output/experiments/illumination_normalization";
        return lighting_experiment::run(output);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
