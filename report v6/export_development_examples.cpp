// Recheck the development steps without writing to application output directories.
#include "../myApplication.cpp"
#include "../experimenent.cpp"

int main() {
    try {
        const std::filesystem::path output = "report v6/assets";
        const auto model = buildColourReferenceModel(cv::imread(MEDIA_DIR + TUNNNING_IMG));
        std::ofstream metrics(output / "development_comparisons.csv");
        std::ofstream decisions(output / "cue_decisions.csv");
        std::ofstream changes(output / "cca_changes.csv");
        metrics << "image,method,TP,FP,FN,correct\n";
        decisions << "image,candidate,trueNumber,unique,paddingOnly,final\n";
        changes << "image,candidate,trueNumber,baselineType,ccaType\n";
        std::map<std::string, std::array<int, 4>> totals;
        auto record = [&](const std::string& image, const std::string& method,
                          const EvaluationResult& evaluation, int correct) {
            metrics << image << ',' << method << ',' << evaluation.TP << ','
                    << evaluation.FP << ',' << evaluation.FN << ',' << correct << '\n';
            auto& sum = totals[method];
            sum[0] += evaluation.TP;
            sum[1] += evaluation.FP;
            sum[2] += evaluation.FN;
            if (correct >= 0) sum[3] += correct;
            else sum[3] = -1; // Identity counts are not evaluated for detector-only rows.
        };

        for (const auto& filename : VAL_IMG) {
            const auto image = cv::imread(MEDIA_DIR + filename);
            const std::string name = std::filesystem::path(filename).stem().string();
            const auto circles = filterPocketDetections(detectBalls(maskTable(image), HoughParams{}));
            const auto context = prepareEvaluation(circles, filename);
            const auto centres = transformCentres(circles);
            std::vector<Candidate> original;
            std::vector<BallTypeResult> types, ccaTypes;
            for (size_t i = 0; i < circles.size(); ++i) {
                Candidate candidate;
                candidate.circle = circles[i];
                candidate.tableCentre = centres[i];
                candidate.roi = extractBall(image, circles[i]);
                const auto segmentation = segmentBallKMeans(candidate.roi, 2);
                candidate.baselineType = classifyBallType(segmentation);
                candidate.colours = scoreBallColours(candidate.roi, model);
                identityEvidence(candidate);
                original.push_back(candidate);
                if (name == "Table3" && (i == 1 || i == 3)) {
                    auto contextCircle = circles[i];
                    contextCircle[2] *= 1.65f;
                    const auto contextCrop = extractBall(drawDetectedBalls(image, {circles[i]}), contextCircle);
                    saveImage(output / ("Table3_candidate_" + std::to_string(i+1) + "_context.png"),
                              contextCrop.image);
                }
                types.push_back(candidate.baselineType);
                const auto spatial = experimentCCA(candidate.roi, segmentation);
                auto cca = candidate.baselineType;
                cca.type = experimentCCAType(candidate.baselineType, spatial);
                ccaTypes.push_back(cca);
                const int g = context.matching.predictionToGroundTruth[i];
                if (g >= 0 && cca.type != candidate.baselineType.type) {
                    changes << name << ',' << i+1 << ',' << context.truth[g].number
                            << ',' << candidate.baselineType.type << ',' << cca.type << '\n';
                }
                if (name == "Table4" && g >= 0 && context.truth[g].number == 1) {
                    saveImage(output / "development_cca_original.png", candidate.roi.image);
                    saveImage(output / "development_cca_light.png", spatial.lightMask);
                }
            }
            record(name, "type_k2", context.matching, evaluateBallTypes(types, context).correct);
            record(name, "type_cca", context.matching, evaluateBallTypes(ccaTypes, context).correct);

            const auto unique = assignUnique(original);
            auto padded = original;
            for (size_t i = 0; i < original.size(); ++i)
                if (original[i].baselineType.type == "White")
                    padded[i] = rawWhitePadding(image, original[i], model, 1);
            const auto paddingOnly = assignUnique(padded);
            reserveCueIdentity(padded, selectAmbiguousCue(original));
            const auto final = assignUnique(padded);
            record(name, "identity_unique", context.matching, evaluateBallIdentities(unique, context).correct);
            record(name, "identity_padding", context.matching, evaluateBallIdentities(paddingOnly, context).correct);
            record(name, "identity_final", context.matching, evaluateBallIdentities(final, context).correct);
            for (size_t i = 0; i < circles.size(); ++i) {
                const int g = context.matching.predictionToGroundTruth[i];
                decisions << name << ',' << i+1 << ',' << (g >= 0 ? context.truth[g].number : -1)
                          << ',' << unique[i] << ',' << paddingOnly[i] << ',' << final[i] << '\n';
            }

            const auto guided = detectGuidedHoughWatershed(image, circles);
            std::vector<cv::Point2f> truth;
            for (const auto& ball : context.truth) truth.push_back(ball.centre);
            for (const auto& variant : std::vector<std::pair<std::string, std::vector<cv::Vec3f>>>{
                    {"global_hough", circles}, {"guided_hough", guided.guided},
                    {"guided_watershed", guided.watershed}}) {
                const auto evaluation = evaluateDetections(transformCentres(variant.second), truth, MATCH_TOLERANCE);
                record(name, variant.first, evaluation, -1);
                if (name == "Table3")
                    saveImage(output / ("development_" + variant.first + ".png"), drawDetectedBalls(image, variant.second));
            }
        }
        for (const auto& [method, sum] : totals) {
            metrics << "TOTAL_TABLE1_6," << method;
            for (const int value : sum) metrics << ',' << value;
            metrics << '\n';
            std::cout << method << ": " << sum[0] << '/' << sum[1] << '/' << sum[2]
                      << " TP/FP/FN; " << sum[3] << " correct\n";
        }
        finishOutputFile(metrics);
        finishOutputFile(decisions);
        finishOutputFile(changes);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
