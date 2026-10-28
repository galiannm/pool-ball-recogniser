#include "circularEdgeSupport.h"
#include "config.h"
#include "detection/ballDetection.h"
#include "evaluation/evaluation.h"
#include "geometry/perspectiveTransform.h"
#include "visualisation/combineImages.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace experiments {
namespace {
struct Record {
    std::string image;
    int candidate, number;
    cv::Vec3f circle;
    CircularEdgeScore score;
};

std::ofstream output(const std::filesystem::path& path) {
    std::ofstream stream(path);
    stream.exceptions(std::ios::badbit | std::ios::failbit);
    stream << std::fixed << std::setprecision(6);
    return stream;
}
void save(const std::filesystem::path& path, const cv::Mat& image) {
    if (!cv::imwrite(path.string(), image))
        throw std::runtime_error("Could not save " + path.string());
}
std::string fixed(double value, int decimals = 1) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(decimals) << value;
    return out.str();
}

cv::Mat closeup(const cv::Mat& image, const CircularEdgeMaps& maps, const Record& record) {
    const auto& c = record.circle;
    const int half = cvCeil(c[2] + 14);
    const cv::Rect crop = cv::Rect(cvRound(c[0]) - half, cvRound(c[1]) - half, 2 * half + 1, 2 * half + 1)
                        & cv::Rect(0, 0, image.cols, image.rows);
    constexpr int scale = 4;
    cv::Mat original, annotated, edge;
    cv::resize(image(crop), original, {}, scale, scale, cv::INTER_NEAREST);
    annotated = original.clone();
    cv::resize(maps.edges(crop), edge, {}, scale, scale, cv::INTER_NEAREST);
    cv::cvtColor(edge, edge, cv::COLOR_GRAY2BGR);
    auto local = [&](cv::Point2f p) {
        return cv::Point(cvRound((p.x - crop.x + 0.5) * scale), cvRound((p.y - crop.y + 0.5) * scale));
    };
    const int n = static_cast<int>(record.score.samples.size());
    for (int i = 0; i < n; ++i) {
        const auto& sample = record.score.samples[i];
        const cv::Scalar colour = !sample.valid ? cv::Scalar(150, 150, 150) :
            sample.supported ? cv::Scalar(40, 240, 40) :
            sample.nearbyEdge ? cv::Scalar(0, 180, 255) : cv::Scalar(60, 60, 255);
        // Short sector centred on the sampled direction; no lines across unsupported gaps.
        std::vector<cv::Point> sector;
        for (int j = -2; j <= 2; ++j) {
            const double a = (i + 0.24 * j) * 2 * CV_PI / n;
            sector.push_back(local({float(c[0] + c[2] * std::cos(a)), float(c[1] + c[2] * std::sin(a))}));
        }
        cv::polylines(annotated, sector, false, colour, 3, cv::LINE_AA);
        cv::polylines(edge, sector, false, colour, 2, cv::LINE_AA);
        if (sample.supported)
            cv::circle(edge, local(sample.edge), 2, {255, 255, 0}, cv::FILLED);
    }
    const std::string label = record.image + " #" + std::to_string(record.candidate) +
        (record.number < 0 ? " FP" : " TP ball " + std::to_string(record.number));
    cv::Mat panels = combineImages({{"Original crop", original}, {"Boundary support", annotated},
                                   {"Canny + supporting pixels (cyan)", edge}}, 3);
    cv::Mat result(panels.rows + 82, panels.cols, CV_8UC3, cv::Scalar(32, 32, 32));
    panels.copyTo(result(cv::Rect(0, 82, panels.cols, panels.rows)));
    cv::putText(result, label + " | support " + fixed(100 * record.score.supportFraction) +
        "% | longest arc " + fixed(record.score.longestArcDegrees, 0) + " deg", {10, 21},
        cv::FONT_HERSHEY_SIMPLEX, 0.48, {255, 255, 255}, 1, cv::LINE_AA);
    cv::putText(result, "Green: aligned edge | Orange: edge, wrong direction | Red: no edge | Grey: outside table",
        {10, 44}, cv::FONT_HERSHEY_SIMPLEX, 0.4, {230, 230, 230}, 1, cv::LINE_AA);
    cv::putText(result, "TP/FP are annotation matches, not a filtering decision. No candidates removed.",
        {10, 66}, cv::FONT_HERSHEY_SIMPLEX, 0.4, {230, 230, 230}, 1, cv::LINE_AA);
    return result;
}

void row(std::ostream& out, const Record& r) {
    out << r.image << ',' << r.candidate << ',' << r.number << ',' << (r.number >= 0) << ','
        << r.circle[0] << ',' << r.circle[1] << ',' << r.circle[2] << ',' << r.score.validSamples << ','
        << r.score.coverage << ',' << r.score.nearbyEdgeFraction << ',' << r.score.supportFraction << ','
        << r.score.longestArcDegrees << ',' << r.score.bridgedArcDegrees << '\n';
}
constexpr auto header = "image,candidate,trueNumber,baselineMatched,x,y,radius,validSamples,coverage,"
                        "nearbyEdgeFraction,radialSupportFraction,longestArcDegrees,bridgedArcDegrees\n";
} // namespace

void runCircularEdgeExperiment(const std::filesystem::path& directory) {
    std::filesystem::create_directories(directory);
    const CircularEdgeOptions options;
    auto settings = output(directory / "settings.csv");
    settings << "angularSamples,tolerancePixels,minimumRadialAlignment,cannyLow,cannyHigh,blurKernel,blurSigma,mode\n"
             << options.angularSamples << ',' << options.tolerance << ',' << options.minimumRadialAlignment << ','
             << options.cannyHigh / 2 << ',' << options.cannyHigh << ",5,1.5,measurement_only\n";
    auto csv = output(directory / "candidate_scores.csv");
    csv << header;
    auto samples = output(directory / "angular_samples.csv");
    samples << "image,candidate,sample,angleDegrees,boundaryX,boundaryY,valid,nearbyEdge,supported,edgeX,edgeY,alignment\n";
    auto metrics = output(directory / "baseline_metrics.csv");
    metrics << "image,groundTruth,detections,TP,FP,FN,precision,recall,F1,meanError\n";
    std::vector<Record> records;
    for (const auto& filename : config::allImages()) {
        const cv::Mat image = cv::imread(config::mediaDirectory + filename);
        if (image.empty()) throw std::runtime_error("Could not load " + filename);
        const auto detection = runDetection(image, HoughParams{});
        cv::Mat mask = cv::Mat::zeros(image.size(), CV_8UC1);
        cv::fillConvexPoly(mask, getTableCorners(), cv::Scalar(255));
        const auto maps = prepareCircularEdgeMaps(image, mask, options);
        // All candidate measurements are computed before annotations are loaded.
        std::vector<CircularEdgeScore> scores;
        for (const auto& circle : detection.filteredCircles)
            scores.push_back(measureCircularEdgeSupport(maps, circle, options));
        const auto truth = loadLabelledGroundTruth(config::groundTruthPath, filename);
        if (truth.empty()) throw std::runtime_error("Missing annotations for " + filename);
        std::vector<cv::Point2f> truthCentres;
        for (const auto& ball : truth) truthCentres.push_back(ball.centre);
        const auto e = evaluateDetections(transformCentres(detection.filteredCircles), truthCentres, config::matchTolerance);
        metrics << filename << ',' << truth.size() << ',' << detection.filteredCircles.size() << ',' << e.TP << ','
            << e.FP << ',' << e.FN << ',' << e.precision << ',' << e.recall << ',' << e.f1 << ',' << e.meanError << '\n';
        const auto imageDirectory = directory / std::filesystem::path(filename).stem();
        std::filesystem::create_directories(imageDirectory);
        cv::Mat overview = image.clone();
        for (size_t i = 0; i < scores.size(); ++i) {
            const int match = e.predictionToGroundTruth[i];
            Record r{filename, int(i + 1), match < 0 ? -1 : truth[match].number,
                     detection.filteredCircles[i], std::move(scores[i])};
            row(csv, r);
            for (size_t a = 0; a < r.score.samples.size(); ++a) {
                const auto& s = r.score.samples[a];
                samples << filename << ',' << r.candidate << ',' << a << ',' << 360.0 * a / options.angularSamples
                    << ',' << s.boundary.x << ',' << s.boundary.y << ',' << s.valid << ',' << s.nearbyEdge << ','
                    << s.supported << ',' << s.edge.x << ',' << s.edge.y << ',' << s.alignment << '\n';
            }
            const auto name = "candidate_" + std::to_string(i + 1) + ".png";
            const auto panel = closeup(image, maps, r);
            save(imageDirectory / name, panel);
            if (match < 0)
                save(directory / (std::filesystem::path(filename).stem().string() + "_false_candidate_" +
                    std::to_string(i + 1) + ".png"), panel);
            const cv::Scalar colour = match < 0 ? cv::Scalar(60, 60, 255) : cv::Scalar(50, 240, 50);
            const cv::Point point(cvRound(r.circle[0]), cvRound(r.circle[1]));
            cv::circle(overview, point, cvRound(r.circle[2]), colour, 2);
            cv::putText(overview, std::to_string(r.candidate) + ":" + fixed(100 * r.score.supportFraction, 0),
                        point + cv::Point(0, -cvRound(r.circle[2]) - 4), cv::FONT_HERSHEY_SIMPLEX, 0.35, colour, 1);
            records.push_back(std::move(r));
        }
        save(imageDirectory / "overview.png", overview);
        save(imageDirectory / "canny.png", maps.edges);
        std::cout << filename << ": measured " << scores.size() << " candidates; " << e.FP << " baseline FP\n";
    }
    std::stable_sort(records.begin(), records.end(), [](const Record& a, const Record& b) {
        return a.score.supportFraction < b.score.supportFraction;
    });
    auto ranked = output(directory / "scores_ranked.csv");
    ranked << header;
    for (const auto& r : records) row(ranked, r);

    auto summary = output(directory / "summary.md");
    summary << "# Circular edge support diagnostic\n\n"
        << "Measurement only: no circle is rejected, moved or resized. Settings are fixed before scoring.\n\n"
        << "72 boundary directions (5 degrees apart); search within 2.5 source-image pixels; "
        << "Canny 37.5/75 after 5x5 Gaussian blur, sigma 1.5; absolute radial gradient alignment >= 0.6. "
        << "Polarity is ignored. Edges come from the original image, and support is restricted to the table.\n\n"
        << "Support is aligned supported directions / valid directions. Coverage is valid / 72. "
        << "Raw longest arc accounts for wrap-around. Bridged arc also fills isolated missing directions; "
        << "it does not change the support fraction. Nearby-edge fraction ignores alignment. "
        << "These scores are geometric evidence, not calibrated probabilities.\n\n"
        << "## False detections and score overlap\n\n"
        << "Counts below compare each false detection to true detections in Table1–6 (AllBalls excluded). "
        << "They diagnose score overlap on already inspected images; they do not select or validate a threshold.\n\n"
        << "| Image | Candidate | Support | Raw arc | Bridged arc | True detections with equal/lower support |\n"
        << "|---|---:|---:|---:|---:|---:|\n";
    int tp = 0, fp = 0;
    for (const auto& r : records)
        if (r.image != config::tuningImage) (r.number >= 0 ? tp : fp)++;
    for (const auto& r : records) {
        if (r.number >= 0 || r.image == config::tuningImage) continue;
        int lower = 0;
        for (const auto& t : records)
            if (t.number >= 0 && t.image != config::tuningImage && t.score.supportFraction <= r.score.supportFraction)
                ++lower;
        summary << "| " << r.image << " | " << r.candidate << " | " << fixed(100 * r.score.supportFraction)
            << "% | " << r.score.longestArcDegrees << " | " << r.score.bridgedArcDegrees << " | " << lower << '/' << tp << " |\n";
    }
    summary << "\nBaseline Table1–6: " << tp << " matched detections, " << fp << " false detections.\n\n"
        << "## Lowest-scoring genuine detections\n\n"
        << "| Image | Candidate | Ball | Support | Coverage | Raw arc |\n|---|---:|---:|---:|---:|---:|\n";
    int listed = 0;
    for (const auto& r : records) {
        if (r.number < 0 || r.image == config::tuningImage) continue;
        summary << "| " << r.image << " | " << r.candidate << " | " << r.number << " | "
            << fixed(100 * r.score.supportFraction) << "% | " << fixed(100 * r.score.coverage)
            << "% | " << r.score.longestArcDegrees << " |\n";
        if (++listed == 12) break;
    }
    summary << "\nOpen the false-candidate PNGs at this directory's top level first. "
        << "Each image folder contains one PNG per candidate, an overview (IDs and support percentages; "
        << "green TP, red FP), and the Canny map. Close-up boundary colours mean green aligned, "
        << "orange nearby but unaligned, red no edge, grey invalid; cyan dots mark supporting edge pixels. "
        << "TP/FP labels come solely from one-to-one annotation matching within 15 table pixels.\n\n"
        << "Touching balls, shadows, number patches and edges shared with neighbours can all affect support. "
        << "Inspect close-ups alongside scores before designing a rejection rule. "
        << "The main detector, recognition and report are unchanged.\n";
    std::cout << "Saved circular edge diagnostics to " << directory << '\n';
}
} // namespace experiments
