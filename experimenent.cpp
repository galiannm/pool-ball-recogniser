#include "experiment.h"
#include "myApplication.h"

/*
======================
    CCA + K MEANS
======================
*/

// TODO_CHECK
CCAExp experimentCCA(const BallROI& ball, const KMeansResult& seg) {
    CCAExp out;
    out.lightMask=cv::Mat::zeros(ball.image.size(), CV_8U);

    for(const auto& r:seg.regions) {
        if(r.hsv[2]>=170 && r.hsv[1]<=165) {
            cv::bitwise_or(out.lightMask, r.mask, out.lightMask);
        }
    }

    cv::subtract(ball.mask,out.lightMask,out.nonLightMask);
    const double area=std::max(1,cv::countNonZero(ball.mask));

    auto measure=[&](const cv::Mat& mask,int& count,double& largest){
        cv::Mat labels, stats, centroids;
        const int n=cv::connectedComponentsWithStats(mask, labels, stats, centroids, 8);
        count = 0;
        int big = 0;
        for(int i=1;i<n;++i) {
            const int pixels=stats.at<int>(i, cv::CC_STAT_AREA);
            if (100.0*pixels/area >= 2.0) {
                ++count;big=std::max(big, pixels);
            }
        }
        largest=100.0*big/area;
    };

    measure(out.lightMask, out.lightCount, out.largestLightPct);
    measure(out.nonLightMask, out.nonLightCount, out.largestNonLightPct);

    cv::Mat eroded, boundary, coverage;
    cv::erode(ball.mask, eroded, cv::getStructuringElement(cv::MORPH_ELLIPSE, {5,5}));
    cv::subtract(ball.mask, eroded, boundary);
    cv::bitwise_and(boundary, out.lightMask, coverage);
    out.boundaryLightPct=100.0 * cv::countNonZero(coverage) / std::max(1, cv::countNonZero(boundary));
    
    return out;
}

std::string experimentCCAType(const BallTypeResult& base, const CCAExp& spatial) {
    const double nonLight=base.colouredPercent+base.darkPercent;

    if (base.lightPercent>=85 && spatial.largestLightPct>=80 && spatial.boundaryLightPct>=75 && spatial.nonLightCount==0) 
        return "White";
    if (spatial.largestLightPct>=20 && spatial.boundaryLightPct>=20 && nonLight>=10) 
        return "Striped";
    if (nonLight>=70 && spatial.boundaryLightPct<=12 && spatial.nonLightCount==1) 
        return "Solid";
    return base.type;
}

void runK2CCAExperiment() {
    const std::filesystem::path directory = std::filesystem::path(EXPERIMENT_DIR) / "k2_cca";

    std::filesystem::create_directories(directory);

    std::ofstream metrics(directory / "metrics.csv");
    std::ofstream changes(directory / "changes.csv");

    if (!metrics.is_open() || !changes.is_open())
        throw std::runtime_error("Could not create CCA experiment CSVs");

    metrics << "image,matched,baselineCorrect,ccaCorrect,corrected,regressed\n";

    changes << "image,candidate,trueType,baselineType,ccaType,"
               "lightComponents,nonLightComponents,"
               "largestLightPercent,boundaryLightPercent\n";

    int totalMatched = 0;
    int totalBaselineCorrect = 0;
    int totalCCACorrect = 0;
    int totalCorrected = 0;
    int totalRegressed = 0;

    for (const std::string& filename : VAL_IMG) {
        const cv::Mat image = cv::imread(MEDIA_DIR + filename);

        if (image.empty())
            throw std::runtime_error("Could not load: " + filename);

        // Both methods use exactly the same detections and crops.
        const auto raw = detectBalls(maskTable(image), HoughParams{});
        const auto circles = filterPocketDetections(raw);
        const auto context = prepareEvaluation(circles, filename);

        std::vector<BallTypeResult> baselineTypes;
        std::vector<BallTypeResult> ccaTypes;
        std::vector<ImagePanel> panels;

        int corrected = 0;
        int regressed = 0;

        for (size_t i = 0; i < circles.size(); ++i) {
            const BallROI ball = extractBall(image, circles[i]);
            const auto segmentation = segmentBallKMeans(ball, 2);
            const BallTypeResult baseline = classifyBallType(segmentation);
            const CCAExp spatial = experimentCCA(ball, segmentation);

            BallTypeResult cca = baseline;
            cca.type = experimentCCAType(baseline, spatial);

            baselineTypes.push_back(baseline);
            ccaTypes.push_back(cca);

            const int g = context.matching.predictionToGroundTruth[i];

            if (g < 0)
                continue;

            const std::string expected = identityType(context.truth[g].number);

            const bool baselineCorrect = baseline.type == expected;
            const bool ccaCorrect = cca.type == expected;

            corrected += !baselineCorrect && ccaCorrect;
            regressed += baselineCorrect && !ccaCorrect;

            // Record every changed prediction.
            if (baseline.type != cca.type) {
                changes << filename << ',' << i + 1 << ','
                        << expected << ','
                        << baseline.type << ',' << cca.type << ','
                        << spatial.lightCount << ','
                        << spatial.nonLightCount << ','
                        << spatial.largestLightPct << ','
                        << spatial.boundaryLightPct << '\n';
            }

            // Show changes and unresolved errors, rather than every ball.
            if (baseline.type != cca.type || !baselineCorrect) {
                const std::string label = "Candidate " + std::to_string(i + 1);

                panels.push_back({
                    label + " / true: " + expected, ball.image
                });

                panels.push_back({
                    "K2: " + baseline.type, segmentation.segmentedImage
                });

                panels.push_back({
                    "Light / CCA: " + cca.type, spatial.lightMask
                });

                panels.push_back({
                    "Non-light regions", spatial.nonLightMask
                });
            }
        }

        const auto baselineEvaluation = evaluateBallTypes(baselineTypes, context);
        const auto ccaEvaluation = evaluateBallTypes(ccaTypes, context);

        metrics << filename << ','
                << baselineEvaluation.matched << ','
                << baselineEvaluation.correct << ','
                << ccaEvaluation.correct << ','
                << corrected << ',' << regressed << '\n';

        totalMatched += baselineEvaluation.matched;
        totalBaselineCorrect += baselineEvaluation.correct;
        totalCCACorrect += ccaEvaluation.correct;
        totalCorrected += corrected;
        totalRegressed += regressed;

        const auto imageDirectory = directory / std::filesystem::path(filename).stem();

        const cv::Mat grid = combineImages(panels, 4);
        if (!grid.empty())
            saveImage(imageDirectory / "candidate_comparison.png", grid);

        saveImage(
            imageDirectory / "table_comparison.png",
            combineImages({
                {"K=2", drawBallTypes(image, circles, baselineTypes)},
                {"K=2 + CCA", drawBallTypes(image, circles, ccaTypes)}}, 1));
    }

    metrics << "ALL," << totalMatched << ','
            << totalBaselineCorrect << ','
            << totalCCACorrect << ','
            << totalCorrected << ','
            << totalRegressed << '\n';

    finishOutputFile(metrics);
    finishOutputFile(changes);

    std::cout << "\n---------K=2 + CCA Experiment---------\n"
              << "K=2 correct types: "
              << totalBaselineCorrect << "/" << totalMatched << '\n'
              << "K=2 + CCA correct types: "
              << totalCCACorrect << "/" << totalMatched << '\n'
              << "Corrected predictions: " << totalCorrected << '\n'
              << "Previously correct predictions lost: "
              << totalRegressed << '\n';
}

/*
======================
    WATERSHED +
    GUIDED HOUGH
======================
*/

cv::Mat makeExperimentForeground(const cv::Mat& image) {
    cv::Mat hsv, cloth;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);

    // Fixed experimental cloth thresholds.
    cv::inRange(hsv, cv::Scalar(35, 35, 30), cv::Scalar(95, 255, 255), cloth);

    cv::Mat table = cv::Mat::zeros(image.size(), CV_8UC1);
    cv::fillConvexPoly(table, TABLE_CORNERS, cv::Scalar(255));

    // Reduce contributions from the table boundary.
    cv::erode(table, table, cv::getStructuringElement(cv::MORPH_ELLIPSE, {9, 9}));

    cv::Mat foreground;
    cv::bitwise_not(cloth, foreground);
    cv::bitwise_and(foreground, table, foreground);

    cv::morphologyEx(foreground, foreground, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_ELLIPSE, {3, 3}));

    // Keep the foreground restricted to the table after closing.
    cv::bitwise_and(foreground, table, foreground);
    return foreground;
}

GuidedExperimentResult detectGuidedHoughWatershed(
    const cv::Mat& image,
    const std::vector<cv::Vec3f>& baseline) {

    if (image.empty())
        throw std::runtime_error("Empty guided-experiment image");

    GuidedExperimentResult result;
    result.baseline = baseline;
    result.guided = baseline;
    result.watershed = baseline;
    result.foreground = makeExperimentForeground(image);

    const auto centre = [](const cv::Vec3f& circle) {
        return cv::Point(cvRound(circle[0]), cvRound(circle[1]));
    };

    const auto duplicate = [](const cv::Vec3f& candidate,
                              const std::vector<cv::Vec3f>& circles) {
        for (const auto& other : circles) {
            const double distance = std::hypot(candidate[0] - other[0], candidate[1] - other[1]);

            if (distance < 0.8 * std::min(candidate[2], other[2]))
                return true;
        }
        return false;
    };

    // Enlarged disks connect nearby baseline detections into groups.
    cv::Mat groups = cv::Mat::zeros(image.size(), CV_8UC1);

    for (const auto& circle : baseline) {
        cv::circle(groups, centre(circle), cvRound(1.4 * circle[2]), cv::Scalar(255), cv::FILLED);
    }

    cv::Mat labels, stats, centroids;
    const int groupCount = cv::connectedComponentsWithStats(
        groups, labels, stats, centroids);

    for (int group = 1; group < groupCount; ++group) {
        int seedCount = 0;
        double radiusSum = 0;

        for (const auto& circle : baseline) {
            if (labels.at<int>(centre(circle)) == group) {
                ++seedCount;
                radiusSum += circle[2];
            }
        }

        // Only search regions containing at least two nearby detections.
        if (seedCount < 2)
            continue;

        const double expectedRadius = radiusSum / seedCount;
        const int padding = cvCeil(expectedRadius);

        cv::Rect crop(stats.at<int>(group, cv::CC_STAT_LEFT) - padding,
                        stats.at<int>(group, cv::CC_STAT_TOP) - padding,
                        stats.at<int>(group, cv::CC_STAT_WIDTH) + 2 * padding,
                        stats.at<int>(group, cv::CC_STAT_HEIGHT) + 2 * padding);

        crop &= cv::Rect(0, 0, image.cols, image.rows);

        // More permissive than the baseline centre-vote threshold.
        HoughParams localParams;
        localParams.param2 = 22;
        localParams.minDist = 12;

        const auto localCircles = detectBalls(image(crop), localParams);
        std::vector<cv::Vec3f> proposals;

        for (const auto& local : localCircles) {
            const cv::Vec3f candidate(
                local[0] + crop.x, local[1] + crop.y, local[2]);

            const cv::Point point = centre(candidate);

            if (!cv::Rect(0, 0, image.cols, image.rows).contains(point))
                continue;

            if (result.foreground.at<uchar>(point) == 0)
                continue;

            if (filterPocketDetections({candidate}).empty())
                continue;

            if (duplicate(candidate, result.guided))
                continue;

            proposals.push_back(candidate);
            result.guided.push_back(candidate);
        }

        // Baseline circles and new proposals provide watershed markers.
        std::vector<cv::Vec3f> seeds;
        for (const auto& circle : baseline) {
            if (crop.contains(centre(circle))) {
                seeds.push_back(circle);
            }
        }

        const size_t firstProposal = seeds.size();
        seeds.insert(seeds.end(), proposals.begin(), proposals.end());

        cv::Mat foreground = result.foreground(crop).clone();

        // Ensure marker cores belong to the provisional foreground.
        for (const auto& seed : seeds) {
            cv::circle(foreground, centre(seed) - crop.tl(), 3, cv::Scalar(255), cv::FILLED);
        }

        cv::Mat markers = cv::Mat::zeros(crop.size(), CV_32SC1);
        markers.setTo(1, foreground == 0); // Known background.

        for (size_t i = 0; i < seeds.size(); ++i) {
            cv::circle(markers, centre(seeds[i]) - crop.tl(), 2, cv::Scalar(static_cast<int>(i) + 2), cv::FILLED);
        }

        cv::Mat smooth;
        cv::GaussianBlur(image(crop), smooth, {5, 5}, 1.5);
        cv::watershed(smooth, markers);

        // Baseline circles are retained; only new proposals are filtered.
        for (size_t i = firstProposal; i < seeds.size(); ++i) {
            cv::Mat basin = markers == static_cast<int>(i) + 2;
            cv::bitwise_and(basin, foreground, basin);

            const double diskArea = CV_PI * seeds[i][2] * seeds[i][2];
            const double areaRatio = cv::countNonZero(basin) / diskArea;

            if (areaRatio >= 0.15 && areaRatio <= 2.0)
                result.watershed.push_back(seeds[i]);
        }

        // Save diagnostic panels without treating basin colours as identities.
        cv::Mat labels8, basinView;
        markers.convertTo(labels8, CV_8U, 13);
        cv::applyColorMap(labels8, basinView, cv::COLORMAP_TURBO);
        basinView.setTo(cv::Scalar(0, 0, 0), markers <= 1);

        const std::string name = "Region " + std::to_string(group);
        result.diagnostics.push_back({name + ": original", image(crop)});
        result.diagnostics.push_back({name + ": foreground", foreground});
        result.diagnostics.push_back({name + ": basins", basinView});
    }

    return result;
}

void runGuidedHoughWatershedExperiment() {
    const std::filesystem::path directory =
        std::filesystem::path(EXPERIMENT_DIR) / "guided_hough_watershed";

    std::filesystem::create_directories(directory);

    std::ofstream csv(directory / "metrics.csv");
    if (!csv.is_open())
        throw std::runtime_error("Could not create guided experiment CSV");

    csv << "image,method,TP,FP,FN,precision,recall,F1,recovered,lost\n";

    const std::array<std::string, 3> names = {
        "baseline", "guided_hough", "guided_hough_watershed"
    };

    std::array<int, 3> totalTP{}, totalFP{}, totalFN{};
    std::array<int, 3> totalRecovered{}, totalLost{};

    for (const std::string& filename : VAL_IMG) {
        const cv::Mat image = cv::imread(MEDIA_DIR + filename);

        if (image.empty())
            throw std::runtime_error("Could not load: " + filename);

        const auto raw = detectBalls(maskTable(image), HoughParams{});
        const auto baseline = filterPocketDetections(raw);
        const auto result = detectGuidedHoughWatershed(image, baseline);

        const auto context = prepareEvaluation(baseline, filename);

        std::vector<cv::Point2f> trueCentres;
        for (const auto& ball : context.truth)
            trueCentres.push_back(ball.centre);

        const std::array<std::vector<cv::Vec3f>, 3> variants = {
            result.baseline, 
            result.guided, 
            result.watershed
        };

        std::vector<ImagePanel> comparisons;

        for (size_t method = 0; method < variants.size(); ++method) {
            const auto evaluation = evaluateDetections(transformCentres(variants[method]), trueCentres, MATCH_TOLERANCE);

            int recovered = 0;
            int lost = 0;

            for (size_t g = 0; g < context.truth.size(); ++g) {
                const bool before =context.matching.groundTruthToPrediction[g] >= 0;
                const bool after = evaluation.groundTruthToPrediction[g] >= 0;

                recovered += !before && after;
                lost += before && !after;
            }

            csv << filename << ',' << names[method] << ','
                << evaluation.TP << ',' << evaluation.FP << ','
                << evaluation.FN << ',' << evaluation.precision << ','
                << evaluation.recall << ',' << evaluation.f1 << ','
                << recovered << ',' << lost << '\n';

            totalTP[method] += evaluation.TP;
            totalFP[method] += evaluation.FP;
            totalFN[method] += evaluation.FN;
            totalRecovered[method] += recovered;
            totalLost[method] += lost;

            const std::string label =
                names[method] +
                " / TP " + std::to_string(evaluation.TP) +
                " FP " + std::to_string(evaluation.FP) +
                " FN " + std::to_string(evaluation.FN);

            comparisons.push_back({label, drawDetectedBalls(image, variants[method])});
        }

        const auto imageDirectory = directory / std::filesystem::path(filename).stem();

        saveImage(imageDirectory / "comparison.png", combineImages(comparisons, 1));
        saveImage(imageDirectory / "foreground.png", result.foreground);

        const cv::Mat diagnostics = combineImages(result.diagnostics, 3);
        if (!diagnostics.empty())
            saveImage(imageDirectory / "regions.png", diagnostics);
    }

    const auto fraction = [](int numerator, int denominator) {
        return denominator > 0 ? double(numerator) / denominator : 0.0;
    };

    std::cout << "\n---------Guided Hough + Watershed Experiment---------\n";
    for (size_t method = 0; method < names.size(); ++method) {
        const int tp = totalTP[method];
        const int fp = totalFP[method];
        const int fn = totalFN[method];

        csv << "ALL," << names[method] << ','
            << tp << ',' << fp << ',' << fn << ','
            << fraction(tp, tp + fp) << ','
            << fraction(tp, tp + fn) << ','
            << fraction(2 * tp, 2 * tp + fp + fn) << ','
            << totalRecovered[method] << ','
            << totalLost[method] << '\n';

        std::cout << names[method]
                  << ": TP=" << tp << " FP=" << fp << " FN=" << fn
                  << " recovered=" << totalRecovered[method]
                  << " lost=" << totalLost[method] << '\n';
    }
    std::cout << "\n";
    finishOutputFile(csv);
}

void runGuidedRecognitionExperiment() {
    const std::filesystem::path root =
        std::filesystem::path(EXPERIMENT_DIR) / "guided_recognition";

    std::filesystem::create_directories(root);

    // Use one shared colour model for every detection variant.
    const cv::Mat referenceImage = cv::imread(MEDIA_DIR + TUNNNING_IMG);
    const BallColourModel model = buildColourReferenceModel(referenceImage);

    const std::array<std::string, 3> names = {
        "baseline",
        "guided_hough",
        "guided_hough_watershed"
    };

    struct Totals {
        int TP = 0;
        int FP = 0;
        int FN = 0;
        int groundTruth = 0;
        int correct = 0;
        int unknown = 0;
        int numberedFalsePositives = 0;
    };

    std::array<Totals, 3> totals{};

    std::ofstream csv(root / "recognition_comparison.csv");
    if (!csv.is_open())
        throw std::runtime_error("Could not create recognition comparison CSV");

    csv << "image,method,TP,FP,FN,groundTruth,correctIdentities,"
           "unknown,numberedFalsePositives,identityAccuracy,"
           "endToEndAccuracy\n";

    const auto fraction = [](int numerator, int denominator) {
        return denominator > 0 ? double(numerator) / denominator : 0.0;
    };

    const auto writeRow = [&](const std::string& imageName, const std::string& method, const Totals& counts) {
        csv << imageName << ',' << method << ','
            << counts.TP << ','
            << counts.FP << ','
            << counts.FN << ','
            << counts.groundTruth << ','
            << counts.correct << ','
            << counts.unknown << ','
            << counts.numberedFalsePositives << ','
            << fraction(counts.correct, counts.TP) << ','
            << fraction(counts.correct, counts.groundTruth) << '\n';
    };

    for (const std::string& filename : VAL_IMG) {
        const cv::Mat image = cv::imread(MEDIA_DIR + filename);

        if (image.empty())
            throw std::runtime_error("Could not load: " + filename);

        const auto raw = detectBalls(maskTable(image), HoughParams{});
        const auto baseline = filterPocketDetections(raw);
        const auto guided = detectGuidedHoughWatershed(image, baseline);

        const std::array<std::vector<cv::Vec3f>, 3> variants = {
            baseline,
            guided.guided,
            guided.watershed
        };

        for (size_t method = 0; method < variants.size(); ++method) {
            const auto& circles = variants[method];

            const auto directory = root / names[method] / std::filesystem::path(filename).stem();

            // Each variant needs matching for its own detection list.
            const auto context = prepareEvaluation(circles, filename);

            // Identical recognition stages and settings for every variant.
            const auto balls = runExtraction(image, circles, directory);

            const auto types = runTypeRecognition(image, balls, circles, directory, filename, context);

            const auto colours = runColourRecognition(
                image, balls, circles, model,
                directory, filename, context);

            const auto identity = runIdentityRecognition(
                image, balls, circles,
                types.types, colours.colours,
                model, directory, filename, context);

            Totals counts;
            counts.TP = context.matching.TP;
            counts.FP = context.matching.FP;
            counts.FN = context.matching.FN;
            counts.groundTruth = identity.groundTruthCount;
            counts.correct = identity.correct;
            counts.unknown = identity.unknown;

            // Check whether false detections receive actual identities.
            for (size_t i = 0; i < circles.size(); ++i) {
                const bool falseDetection = context.matching.predictionToGroundTruth[i] < 0;

                if (falseDetection && identity.numbers[i] >= 0)
                    ++counts.numberedFalsePositives;
            }

            writeRow(filename, names[method], counts);

            auto& total = totals[method];
            total.TP += counts.TP;
            total.FP += counts.FP;
            total.FN += counts.FN;
            total.groundTruth += counts.groundTruth;
            total.correct += counts.correct;
            total.unknown += counts.unknown;
            total.numberedFalsePositives += counts.numberedFalsePositives;
        }
    }

    std::cout << "\n---------Guided Detection + Recognition---------\n";

    for (size_t method = 0; method < names.size(); ++method) {
        const auto& total = totals[method];

        writeRow("ALL", names[method], total);

        std::cout << names[method]
                  << ": TP=" << total.correct
                  << "/" << total.groundTruth
                  << " (" << 100.0 * fraction(total.correct, total.groundTruth) << "%)"
                  << ", FP=" << total.FP
                  << ", numbered FP="
                  << total.numberedFalsePositives
                  << ", unknown=" << total.unknown << '\n';
    }
    std::cout << "\n";
    finishOutputFile(csv);
}