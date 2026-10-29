#include "chamferRefinement.h"
#include "identityHeuristics.h"
#include "config.h"
#include "detection/ballDetection.h"
#include "evaluation/evaluation.h"
#include "geometry/perspectiveTransform.h"
#include "io/finalResults.h"
#include "recognition/ballTypeClassifier.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>

namespace experiments {
namespace {
constexpr double pi = 3.14159265358979323846;
// Fixed before reading test annotations. Radius growth is deliberately bounded.
constexpr int angles = 72, sectors = 12, maximumGrowth = 8, maximumShift = 2;
constexpr double distanceCap = 3, supportDistance = 1.5, minimumAlignment = .7071067811865476;
constexpr double minimumCoverage = .70, maximumMeanDistance = 1.0, outerCostSlack = .15;
constexpr int minimumSectors = 9;
const std::array<std::string, 4> methods = {
    "C_unique", "white_pad1_unique", "chamfer_minimum_unique", "chamfer_outer_unique"};

std::ofstream output(const std::filesystem::path& p) {
    std::ofstream s(p); s.exceptions(std::ios::badbit | std::ios::failbit);
    s << std::fixed << std::setprecision(6); return s;
}
std::string typeFor(int n) {return n < 0 ? "Unknown" : n == 0 ? "White" : n <= 8 ? "Solid" : "Striped";}
double ratio(double a, double b) {return b ? a / b : 0;}

struct Edges {cv::Mat image, dx, dy, distance;};
Edges edgeImage(const cv::Mat& image) {
    cv::Mat gray, smooth;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, smooth, {5,5}, 1);
    Edges e;
    cv::Canny(smooth, e.image, 50, 100, 3, true);
    cv::Sobel(smooth, e.dx, CV_32F, 1, 0, 3);
    cv::Sobel(smooth, e.dy, CV_32F, 0, 1, 3);
    cv::Mat inverse; cv::bitwise_not(e.image, inverse);
    cv::distanceTransform(inverse, e.distance, cv::DIST_L2, cv::DIST_MASK_PRECISE);
    return e;
}
struct CircleScore {
    cv::Vec3f circle;
    double cost = distanceCap, coverage = 0, neighbourFraction = 0;
    int coveredSectors = 0;
};
// The distance is to the nearest edge whose gradient agrees with the template
// radius. Absolute dot product accepts both bright-to-dark and dark-to-bright.
CircleScore score(const Edges& e, const cv::Vec3f& c,
                  const std::vector<cv::Vec3f>& neighbours) {
    CircleScore result; result.circle = c;
    std::array<int, sectors> supported{};
    double total = 0; int count = 0, contaminated = 0;
    for (int a = 0; a < angles; ++a) {
        const double theta = 2 * pi * a / angles, ux = std::cos(theta), uy = std::sin(theta);
        const double x = c[0] + c[2] * ux, y = c[1] + c[2] * uy;
        bool insideNeighbour = false;
        for (const auto& other : neighbours)
            if (std::hypot(x - other[0], y - other[1]) < .85 * other[2]) insideNeighbour = true;
        double best = distanceCap;
        if (insideNeighbour) ++contaminated;
        else for (int yy = int(std::floor(y)) - 3; yy <= int(std::ceil(y)) + 3; ++yy) {
            if (yy < 0 || yy >= e.image.rows) continue;
            for (int xx = int(std::floor(x)) - 3; xx <= int(std::ceil(x)) + 3; ++xx) {
                if (xx < 0 || xx >= e.image.cols || !e.image.at<uchar>(yy,xx)) continue;
                const double d = std::hypot(xx - x, yy - y);
                if (d >= best) continue;
                const double gx = e.dx.at<float>(yy,xx), gy = e.dy.at<float>(yy,xx);
                const double magnitude = std::hypot(gx,gy);
                if (magnitude > 0 && std::abs(gx * ux + gy * uy) / magnitude >= minimumAlignment) best = d;
            }
        }
        total += best;
        if (best <= supportDistance) {++count; ++supported[a / (angles / sectors)];}
    }
    result.cost = total / angles;
    result.coverage = double(count) / angles;
    result.neighbourFraction = double(contaminated) / angles;
    result.coveredSectors = int(std::count_if(supported.begin(), supported.end(), [](int n) {return n >= 3;}));
    return result;
}
bool supported(const CircleScore& c) {
    return c.coverage >= minimumCoverage && c.coveredSectors >= minimumSectors &&
        c.cost <= maximumMeanDistance && c.neighbourFraction <= .15;
}
struct Refinement {
    CircleScore original, minimum, outer;
    std::vector<CircleScore> templates;
    bool minimumAccepted = false, outerAccepted = false;
};
Refinement refine(const Edges& e, const cv::Vec3f& original,
                  const std::vector<cv::Vec3f>& neighbours) {
    Refinement r;
    r.original = r.minimum = r.outer = score(e, original, neighbours);
    for (int growth = 0; growth <= maximumGrowth; ++growth)
        for (int dy = -maximumShift; dy <= maximumShift; ++dy)
            for (int dx = -maximumShift; dx <= maximumShift; ++dx) {
                cv::Vec3f circle(original[0] + dx, original[1] + dy, original[2] + growth);
                auto current = score(e, circle, neighbours);
                r.templates.push_back(current);
                // Original wins an exact tie with it. Among moved templates an
                // exact tie retains the first visited template in loop order.
                if (current.cost + 1e-9 < r.minimum.cost) r.minimum = current;
            }
    r.minimumAccepted = supported(r.minimum) && r.minimum.cost + .05 < r.original.cost;
    if (!r.minimumAccepted) r.minimum = r.original;
    // Pure minimum distance can lock onto an inner white-patch boundary. This
    // separate fixed variant chooses the outermost equally well supported arc.
    double lowest = r.original.cost;
    for (const auto& c : r.templates) lowest = std::min(lowest, c.cost);
    for (const auto& c : r.templates) {
        if (!supported(c) || c.cost > lowest + outerCostSlack) continue;
        if (c.circle[2] > r.outer.circle[2] ||
            (c.circle[2] == r.outer.circle[2] && c.cost < r.outer.cost)) r.outer = c;
    }
    r.outerAccepted = r.outer.circle != original && supported(r.outer);
    return r;
}

void syntheticChecks(const std::filesystem::path& dir) {
    const cv::Scalar cloth(65,100,65);
    cv::Mat image(140,140,CV_8UC3,cloth);
    cv::circle(image,{70,70},22,{220,220,220},cv::FILLED,cv::LINE_AA);
    auto r = refine(edgeImage(image), {68,70,20}, {});
    if (!r.minimumAccepted || std::hypot(r.minimum.circle[0]-70,r.minimum.circle[1]-70) > 2 ||
        std::abs(r.minimum.circle[2]-22) > 1.5) throw std::runtime_error("Synthetic circle refinement failed");
    cv::imwrite((dir / "synthetic_circle.png").string(),image);
    cv::Mat flat(100,100,CV_8UC3,cloth);
    auto empty = refine(edgeImage(flat), {50,50,20}, {});
    if (empty.minimumAccepted || empty.outerAccepted) throw std::runtime_error("Flat image should preserve original circle");
    // A second detected ball must not provide evidence inside its own disk.
    auto neighbour = score(edgeImage(image),{70,70,22},{{70,70,30}});
    if (neighbour.coverage != 0 || neighbour.cost != distanceCap)
        throw std::runtime_error("Neighbour exclusion failed");
    auto clipped = refine(edgeImage(image), {1,1,20}, {});
    if (!std::isfinite(clipped.original.cost)) throw std::runtime_error("Clipped boundary score not finite");
    auto checks = output(dir / "synthetic_checks.txt");
    checks << "PASS: shifted/undersized synthetic circle refines to its edge; flat image keeps original; "
        << "neighbour interiors cannot provide edge support; image-border scoring is finite.\n";
}

cv::Mat detailPanel(const cv::Mat& image, const cv::Vec3f& original,
                    const cv::Vec3f& proposed, const std::string& title, int identity) {
    const int extent = int(std::ceil(original[2])) + maximumGrowth + maximumShift + 5;
    const cv::Rect wanted(cvRound(original[0])-extent,cvRound(original[1])-extent,2*extent+1,2*extent+1);
    const cv::Rect rect = wanted & cv::Rect(0,0,image.cols,image.rows);
    cv::Mat crop = image(rect).clone();
    cv::circle(crop,{cvRound(original[0])-rect.x,cvRound(original[1])-rect.y},cvRound(original[2]),{0,255,255},1,cv::LINE_AA);
    cv::circle(crop,{cvRound(proposed[0])-rect.x,cvRound(proposed[1])-rect.y},cvRound(proposed[2]),{255,255,0},1,cv::LINE_AA);
    cv::Mat panel(290,280,CV_8UC3,cv::Scalar(25,25,25)), resized;
    cv::resize(crop,resized,{260,240},0,0,cv::INTER_NEAREST); resized.copyTo(panel(cv::Rect(10,40,260,240)));
    cv::putText(panel,title,{8,17},cv::FONT_HERSHEY_SIMPLEX,.42,{255,255,255},1,cv::LINE_AA);
    cv::putText(panel,"assigned " + (identity == 0 ? std::string("Cue") : std::to_string(identity)),{8,33},cv::FONT_HERSHEY_SIMPLEX,.42,{255,255,255},1,cv::LINE_AA);
    return panel;
}
struct Counts {int gt=0,tp=0,fp=0,fn=0,type=0,number=0; double error=0;};
void writeCounts(std::ostream& s,const Counts& c) {
    s << c.gt << ',' << c.tp << ',' << c.fp << ',' << c.fn << ',' << ratio(c.tp,c.tp+c.fp) << ','
      << ratio(c.tp,c.gt) << ',' << ratio(2*c.tp,2*c.tp+c.fp+c.fn) << ',' << ratio(c.error,c.tp) << ','
      << c.type << ',' << c.number << ',' << ratio(c.number,c.tp);
}
} // namespace

void runChamferRefinementExperiment(const std::filesystem::path& directory) {
    std::filesystem::create_directories(directory);
    syntheticChecks(directory);
    const std::filesystem::path source = "output/experiments/ball_colour_histograms/training/models.yml";
    const auto model = loadFrozenIdentityColourModel(source);
    std::filesystem::copy_file(source,directory / "frozen_models.yml",std::filesystem::copy_options::overwrite_existing);
    auto configuration = output(directory / "configuration.txt");
    configuration << "Recognition-only boundary refinement; original Hough/pocket-filtered detections, radii and table centres remain unchanged in final_results.\n"
        << "Refinement triggers on every original K2 White candidate, including true cue balls. Baseline and all variants use the identical best-effort uniqueness backend and frozen AllBalls HSV model.\n"
        << "Fixed before annotation evaluation: grayscale Gaussian 5x5 sigma1; Canny L2 thresholds50/100; Sobel3; "
        << "centres shift at most2 pixels per axis; radius growth integer0..8; 72 circle samples; "
        << "edge gradient/radius absolute cosine>=cos45degrees; nearest eligible edge distance clipped3 pixels; "
        << "other detected ball interiors at0.85radius excluded.\n"
        << "Support gates: coverage(distance<=1.5)>=70%; >=9/12 sectors each containing>=3/6 supported angles; "
        << "mean clipped distance<=1; <=15% template samples inside neighbour interiors.\n"
        << "Minimum variant: smallest mean distance, accepted only if all support gates pass and cost improves by>.05px.\n"
        << "Outer variant: largest radius meeting support gates within.15px of the absolute best mean cost; same-radius ties choose lowest cost. Original is fallback.\n"
        << "These geometric scores are not probabilities. The oriented score is computed by a bounded local nearest-edge search; distance_transform.png is an unoriented diagnostic.\n"
        << "No GT enters refinement or recognition. Labels loaded only after every prediction/export for an image. Experiments are evaluation-informed exploratory followups, not independent validation.\n";
    configuration.close();
    auto templates = output(directory / "template_scores.csv");
    templates << "image,candidate,x,y,radius,growth,cost,coverage,sectors,neighbourFraction,passesSupportGates\n";
    auto boundaries = output(directory / "boundaries.csv");
    boundaries << "image,candidate,method,triggerWhite,changed,originalX,originalY,originalRadius,cropX,cropY,cropRadius,cost,coverage,sectors\n";
    auto predictions = output(directory / "predictions.csv");
    predictions << "image,method,candidate,trueNumber,predictedNumber,typeCorrect,numberCorrect,bodyPixels,bestColour,lightPercent\n";
    auto metrics = output(directory / "metrics.csv");
    metrics << "split,image,method,groundTruth,TP,FP,FN,precision,recall,F1,meanError,correctTypes,correctNumbers,numberAccuracy\n";
    auto centreMetrics = output(directory / "refined_centre_metrics.csv");
    centreMetrics << "image,method,TP,FP,FN,meanError,note\n";
    auto changes = output(directory / "changes.csv");
    changes << "image,method,candidate,trueNumber,before,after,status\n";
    std::array<Counts,4> totals{};
    std::array<int,4> fixes{}, losses{}, moved{};
    for (const auto& filename : config::allImages()) {
        const std::string stem = std::filesystem::path(filename).stem().string();
        const auto imageDir = directory / stem;
        std::filesystem::create_directories(imageDir / "candidates");
        const cv::Mat image = cv::imread(config::mediaDirectory + filename);
        if (image.empty()) throw std::runtime_error("Could not read " + filename);
        const auto circles = runDetection(image,HoughParams{}).filteredCircles;
        const auto centres = transformCentres(circles);
        const auto edges = edgeImage(image);
        cv::imwrite((imageDir / "edges.png").string(),edges.image);
        cv::Mat distance; edges.distance.convertTo(distance,CV_8U,255/distanceCap);
        cv::imwrite((imageDir / "distance_transform.png").string(),distance);
        std::array<std::vector<cv::Vec3f>,4> cropCircles;
        std::array<std::vector<BallROI>,4> rois;
        std::vector<bool> triggers;
        std::vector<Refinement> refinements;
        for (size_t i=0;i<circles.size();++i) {
            const auto original = extractBall(image,circles[i]);
            const bool white = classifyBallType(segmentBallKMeans(original,2)).type == "White";
            triggers.push_back(white);
            Refinement r; r.original.circle = r.minimum.circle = r.outer.circle = circles[i];
            if (white) {
                auto neighbours = circles; neighbours.erase(neighbours.begin()+i);
                r = refine(edges,circles[i],neighbours);
                for (const auto& t : r.templates)
                    templates << filename << ',' << i+1 << ',' << t.circle[0] << ',' << t.circle[1] << ',' << t.circle[2] << ','
                        << t.circle[2]-circles[i][2] << ',' << t.cost << ',' << t.coverage << ',' << t.coveredSectors << ','
                        << t.neighbourFraction << ',' << supported(t) << '\n';
            }
            refinements.push_back(r);
            auto padded = circles[i]; if (white) padded[2] += 1;
            const std::array<cv::Vec3f,4> selected = {circles[i],padded,r.minimum.circle,r.outer.circle};
            for (size_t m=0;m<methods.size();++m) {
                cropCircles[m].push_back(selected[m]);
                rois[m].push_back(extractBall(image,selected[m]));
                const auto& s = m==2 ? r.minimum : m==3 ? r.outer : r.original;
                boundaries << filename << ',' << i+1 << ',' << methods[m] << ',' << white << ',' << (selected[m]!=circles[i]) << ','
                    << circles[i][0] << ',' << circles[i][1] << ',' << circles[i][2] << ',' << selected[m][0] << ',' << selected[m][1] << ','
                    << selected[m][2] << ',';
                if (white && m!=1) boundaries << s.cost << ',' << s.coverage << ',' << s.coveredSectors;
                else boundaries << ",,";
                boundaries << '\n';
                if (filename!="AllBalls.jpeg") moved[m] += selected[m]!=circles[i];
            }
        }
        std::array<std::vector<int>,4> numbers;
        for (size_t m=0;m<methods.size();++m) {
            numbers[m] = recogniseUniqueBallROIs(rois[m],model);
            std::vector<std::string> types;
            for (int n:numbers[m]) types.push_back(typeFor(n));
            saveFinalResults(image,directory / "final_results" / methods[m] / stem,circles,centres,types,numbers[m]);
        }
        for (size_t i=0;i<circles.size();++i) if (triggers[i]) {
            std::vector<cv::Mat> panels, bodyPanels;
            for (size_t m=0;m<methods.size();++m) {
                panels.push_back(detailPanel(image,circles[i],cropCircles[m][i],methods[m],numbers[m][i]));
                const auto colour=scoreBallColours(rois[m][i],model);
                cv::Mat body=cv::Mat::zeros(rois[m][i].image.size(),CV_8UC3), resized;
                rois[m][i].image.copyTo(body,colour.bodyMask);
                cv::Mat bodyPanel(290,280,CV_8UC3,cv::Scalar(25,25,25));
                cv::resize(body,resized,{260,240},0,0,cv::INTER_NEAREST);
                resized.copyTo(bodyPanel(cv::Rect(10,40,260,240)));
                cv::putText(bodyPanel,"Body pixels " + std::to_string(colour.pixels),{8,17},cv::FONT_HERSHEY_SIMPLEX,.42,{255,255,255},1,cv::LINE_AA);
                cv::putText(bodyPanel,"Best colour " + std::to_string(colour.bestColour),{8,33},cv::FONT_HERSHEY_SIMPLEX,.42,{255,255,255},1,cv::LINE_AA);
                bodyPanels.push_back(bodyPanel);
            }
            cv::Mat top,bottom,sheet; cv::hconcat(panels,top); cv::hconcat(bodyPanels,bottom);
            cv::vconcat(top,bottom,sheet);
            cv::imwrite((imageDir / "candidates" / ("candidate_"+std::to_string(i+1)+".png")).string(),sheet);
        }
        // Evaluation is deliberately downstream of inference and image export.
        const auto truth = loadLabelledGroundTruth(config::groundTruthPath,filename);
        std::vector<cv::Point2f> actualCentres;
        for (const auto& ball:truth) actualCentres.push_back(ball.centre);
        const auto e = evaluateDetections(centres,actualCentres,config::matchTolerance);
        for (size_t m=0;m<methods.size();++m) {
            Counts c{int(truth.size()),e.TP,e.FP,e.FN,0,0,e.meanError*e.TP};
            const auto movedEvaluation = evaluateDetections(transformCentres(cropCircles[m]),actualCentres,config::matchTolerance);
            centreMetrics << filename << ',' << methods[m] << ',' << movedEvaluation.TP << ',' << movedEvaluation.FP << ','
                << movedEvaluation.FN << ',' << movedEvaluation.meanError << ",diagnostic only; original centres used for recognition matching\n";
            for (size_t i=0;i<circles.size();++i) {
                const int match = e.predictionToGroundTruth[i];
                const int actual = match<0 ? -1 : truth[match].number;
                const auto colour = scoreBallColours(rois[m][i],model);
                const auto type = classifyBallType(segmentBallKMeans(rois[m][i],2));
                predictions << filename << ',' << methods[m] << ',' << i+1 << ',' << actual << ',' << numbers[m][i] << ',';
                if (actual>=0) {
                    const bool typeCorrect=typeFor(numbers[m][i])==typeFor(actual), numberCorrect=numbers[m][i]==actual;
                    c.type+=typeCorrect; c.number+=numberCorrect;
                    predictions << typeCorrect << ',' << numberCorrect;
                    if (m>0 && filename!="AllBalls.jpeg") {
                        fixes[m] += numbers[0][i]!=actual && numberCorrect;
                        losses[m] += numbers[0][i]==actual && !numberCorrect;
                    }
                } else predictions << ',';
                predictions << ',' << colour.pixels << ',' << colour.bestColour << ',' << type.lightPercent << '\n';
                if (m>0 && numbers[m][i]!=numbers[0][i]) {
                    const std::string status=actual<0 ? "unmatched" : numbers[m][i]==actual ? "corrected" : numbers[0][i]==actual ? "regressed" : "changed_still_wrong";
                    changes << filename << ',' << methods[m] << ',' << i+1 << ',' << actual << ',' << numbers[0][i] << ',' << numbers[m][i] << ',' << status << '\n';
                }
            }
            metrics << (filename=="AllBalls.jpeg" ? "training_resubstitution" : "evaluation") << ',' << filename << ',' << methods[m] << ',';
            writeCounts(metrics,c); metrics << '\n';
            if (filename!="AllBalls.jpeg") {
                auto& t=totals[m]; t.gt+=c.gt;t.tp+=c.tp;t.fp+=c.fp;t.fn+=c.fn;t.type+=c.type;t.number+=c.number;t.error+=c.error;
            }
        }
        std::cout << "Chamfer refinement: " << filename << " completed\n";
    }
    auto summary = output(directory / "summary.md");
    summary << "# White-candidate chamfer boundary refinement\n\n"
        << "All four profiles use the same frozen AllBalls histogram and best-effort uniqueness assignment. "
        << "Only original K2 White crops are modified. Detection counts, stored Hough circles and table coordinates remain fixed.\n\n"
        << "| Method | Correct identities | Correct types | Corrected vs C | Regressed vs C | Modified crops |\n"
        << "|---|---:|---:|---:|---:|---:|\n";
    for (size_t m=0;m<methods.size();++m) {
        metrics << "evaluation,ALL," << methods[m] << ','; writeCounts(metrics,totals[m]);metrics << '\n';
        summary << "| " << methods[m] << " | " << totals[m].number << '/' << totals[m].tp << " | " << totals[m].type << '/' << totals[m].tp
            << " | " << fixes[m] << " | " << losses[m] << " | " << moved[m] << " |\n";
    }
    summary << "\nYellow outlines show original Hough circles; cyan outlines show recognition crop circles. "
        << "The second row shows eligible coloured-body pixels for each recognition crop. "
        << "The crop comparison sheets contain predictions, not ground-truth labels. Use predictions.csv and changes.csv for automatic correctness.\n\n"
        << "The oriented chamfer score measures edge alignment, not identity confidence. Inner white patches can also have strong circular edges; "
        << "shape evidence alone cannot guarantee the physical ball boundary. Neighbour edges, perspective, shadows and incomplete arcs remain limitations. "
        << "The outer variant adds a fixed largest-supported-radius preference; it is not a parameter search or a validated optimum. "
        << "Refined centre localisation is reported separately in refined_centre_metrics.csv; final_results intentionally preserve original detection geometry.\n\n"
        << "In this run, the outer-supported variant corrects Table1's cue/14 swap. "
        << "The actual adjustments are small: the real cue shifts down1 pixel with radius unchanged24.15, "
        << "reducing eligible body pixels75 to42; ball14 shifts down1 pixel and grows17.55 to18.55, "
        << "yielding15 eligible body pixels. Radius+1 alone yields only8, below the frozen minimum12. "
        << "Thus the improvement involves centre/mask discretisation and the body-pixel cutoff as well as edge evidence; "
        << "it does not establish recovery of an entire physical outer boundary. "
        << "The real cue's green rim can include cloth, which gives the histogram misleading green evidence. "
        << "The minimum-cost variant changes3 crops without correcting identities; the outer preference changes6 and corrects2.\n\n"
        << "Parameters were fixed before evaluation, but this experiment follows observations on Table1–6 and is exploratory. "
        << "No production integration or report modification is performed.\n";
}
} // namespace experiments
