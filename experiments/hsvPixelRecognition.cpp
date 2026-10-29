#include "hsvPixelRecognition.h"
#include "identityHeuristics.h"
#include "config.h"
#include "detection/ballDetection.h"
#include "evaluation/evaluation.h"
#include "geometry/perspectiveTransform.h"
#include "io/finalResults.h"
#include "io/output.h"
#include "visualisation/combineImages.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace experiments {
namespace {
struct Pixels {BallTypeResult type; cv::Mat light,dark,coloured; int total=0;};
Pixels pixelEvidence(const BallROI& roi) {
    if (roi.image.empty() || roi.image.type()!=CV_8UC3 || roi.mask.type()!=CV_8UC1 || roi.image.size()!=roi.mask.size())
        throw std::invalid_argument("Invalid HSV pixel ROI");
    Pixels p; cv::Mat hsv, nonLight, occupied;
    cv::cvtColor(roi.image,hsv,cv::COLOR_BGR2HSV);
    cv::compare(roi.mask,0,occupied,cv::CMP_GT);
    cv::inRange(hsv,cv::Scalar(0,0,170),cv::Scalar(179,165,255),p.light); cv::bitwise_and(p.light,occupied,p.light);
    cv::inRange(hsv,cv::Scalar(0,0,0),cv::Scalar(179,255,90),p.dark); cv::bitwise_and(p.dark,occupied,p.dark);
    cv::bitwise_not(p.light,nonLight); cv::bitwise_and(nonLight,occupied,nonLight);
    p.dark.setTo(0,p.light);
    cv::bitwise_or(p.light,p.dark,p.coloured); cv::bitwise_not(p.coloured,p.coloured);
    cv::bitwise_and(p.coloured,occupied,p.coloured);
    p.total=cv::countNonZero(occupied);
    if (!p.total) return p;
    p.type.lightPercent=100.0*cv::countNonZero(p.light)/p.total;
    p.type.darkPercent=100.0*cv::countNonZero(p.dark)/p.total;
    p.type.colouredPercent=100.0*cv::countNonZero(p.coloured)/p.total;
    if (p.type.lightPercent>=85) p.type.type="White";
    else if (p.type.darkPercent+p.type.colouredPercent>=70) p.type.type="Solid";
    else if (p.type.lightPercent>0 && p.type.darkPercent+p.type.colouredPercent>0) p.type.type="Striped";
    return p;
}
void syntheticChecks() {
    BallROI roi{cv::Mat(10,10,CV_8UC3,cv::Scalar(0,0,0)),cv::Mat(10,10,CV_8UC1,cv::Scalar(255))};
    roi.image.rowRange(0,6).setTo(cv::Scalar(240,240,240));
    roi.image.rowRange(8,10).setTo(cv::Scalar(0,0,240));
    auto p=pixelEvidence(roi);
    if (p.type.type!="Striped" || p.type.lightPercent!=60 || p.type.darkPercent!=20 || p.type.colouredPercent!=20)
        throw std::runtime_error("HSV mask proportions failed");
    cv::Mat intersection,joined;
    cv::bitwise_and(p.light,p.dark,intersection); if (cv::countNonZero(intersection)) throw std::runtime_error("Overlapping HSV masks");
    cv::bitwise_or(p.light,p.dark,joined); cv::bitwise_and(joined,p.coloured,intersection);
    if (cv::countNonZero(intersection)) throw std::runtime_error("Overlapping coloured mask");
    cv::bitwise_or(joined,p.coloured,joined);
    if (cv::countNonZero(joined)!=100) throw std::runtime_error("HSV masks do not partition ROI");
    roi.mask.rowRange(0,6).setTo(0); p=pixelEvidence(roi);
    if (p.total!=40 || p.type.lightPercent!=0 || p.type.type!="Solid") throw std::runtime_error("HSV mask exclusion failed");
    roi.mask.setTo(0); if (pixelEvidence(roi).type.type!="Unknown") throw std::runtime_error("Empty HSV fallback failed");
    roi.mask.setTo(255); roi.image.setTo(cv::Scalar(240,240,240));
    if (pixelEvidence(roi).type.type!="White") throw std::runtime_error("HSV white classification failed");
}
std::ofstream file(const std::filesystem::path& path) {
    std::ofstream out(path); out.exceptions(std::ios::badbit|std::ios::failbit); out<<std::fixed<<std::setprecision(6); return out;
}
std::string typeFor(int n) {return n<0 ? "Unknown" : n==0 ? "White" : n<=8 ? "Solid" : "Striped";}
std::string numberName(int n) {return n<0 ? "Unknown" : n==0 ? "Cue" : std::to_string(n);}
double fraction(double a,double b) {return b ? a/b : 0;}
struct Method {std::string name; bool pixels,padding,unique,size;};
const std::array<Method,8> methods={{{"K2_local",false,false,false,false},{"HSV_local",true,false,false,false},
    {"K2_unique",false,false,true,false},{"HSV_unique",true,false,true,false},
    {"K2_pad1_unique",false,true,true,false},{"HSV_pad1_unique",true,true,true,false},
    {"K2_pad1_largest_cue",false,true,true,true},{"HSV_pad1_largest_cue",true,true,true,true}}};
struct Result {
    std::vector<BallROI> rois; std::vector<BallTypeResult> evidence;
    std::vector<BallColourScores> colours; std::vector<int> numbers,padding;
    std::vector<std::string> types;
    int whites=0,reservedCue=-1;
};
struct Counts {int gt=0,tp=0,fp=0,fn=0,types=0,numbers=0,unknown=0,cues=0,cueTotal=0; double error=0;};
void writeCounts(std::ostream& out,const Counts& c) {
    out<<c.gt<<','<<c.tp<<','<<c.fp<<','<<c.fn<<','<<fraction(c.tp,c.tp+c.fp)<<','<<fraction(c.tp,c.gt)<<','
       <<fraction(2*c.tp,2*c.tp+c.fp+c.fn)<<','<<fraction(c.error,c.tp)<<','<<c.types<<','<<c.numbers<<','
       <<fraction(c.numbers,c.tp)<<','<<c.unknown<<','<<c.cues<<','<<c.cueTotal;
}
}
void runHSVPixelRecognitionExperiment(const std::filesystem::path& directory) {
    syntheticChecks();
    std::filesystem::create_directories(directory);
    const std::filesystem::path source="output/experiments/ball_colour_histograms/training/models.yml";
    const auto model=loadFrozenIdentityColourModel(source);
    std::filesystem::copy_file(source,directory/"frozen_models.yml",std::filesystem::copy_options::overwrite_existing);
    const std::filesystem::path rowSource="output/experiments/row_cue_size/row_model.yml";
    cv::FileStorage row(rowSource.string(),cv::FileStorage::READ);
    if (!row.isOpened() || std::string(row["trainingImage"])!="AllBalls.jpeg") throw std::runtime_error("Run --row-cue-size first");
    const double intercept=double(row["intercept"]),slope=double(row["slope"]);
    if (!std::isfinite(intercept) || !std::isfinite(slope) || intercept<=0 || slope<0) throw std::runtime_error("Invalid row model");
    row.release(); std::filesystem::copy_file(rowSource,directory/"frozen_row_model.yml",std::filesystem::copy_options::overwrite_existing);
    auto configFile=file(directory/"configuration.txt");
    configFile<<"Direct HSV pixel type evidence replaces K2 only in HSV profiles. All thresholds copied unchanged: "
        <<"light V>=170,S<=165; dark V<=90 excluding light; coloured=remaining ROI pixels. "
        <<"Masks disjoint and exhaustive inside the original extraction disk; no erosion for type proportions. "
        <<"Cue light>=85%; solid nonlight>=70%; otherwise stripe when both present. Empty ROI ->Unknown.\n"
        <<"Frozen AllBalls HSV histogram models, min12 body pixels, histogram body erosion3 remain unchanged. "
        <<"Colour classification uses 1-Bhattacharyya distance, not mean backprojection. "
        <<"Unique profiles use unchanged Gaussian type width.10 and best-effort Hungarian assignment.\n"
        <<"Each frontend uses its OWN original White classifications to trigger padding and size reservation. "
        <<"No KMeans call enters HSV type/trigger evidence. Padding1 only on originally White candidates. "
        <<"Largest-cue profiles reserve cue for greatest original radius/expected(row) only with>=2 original White candidates. "
        <<"Row model frozen from --row-cue-size; no refit. Original geometry retained in exports.\n"
        <<"No threshold search or retraining. This follows inspected test failures; exploratory evaluation only.\n";
    auto checks=file(directory/"synthetic_checks.txt"); checks<<"PASS: exact pixel fractions; disjoint exhaustive masks; excluded pixels; empty mask fallback; white classification.\n";
    auto metrics=file(directory/"metrics.csv");
    metrics<<"split,image,method,groundTruth,TP,FP,FN,precision,recall,F1,meanError,correctTypes,correctNumbers,numberAccuracy,unknownPredictions,correctCues,totalCues\n";
    auto predictions=file(directory/"predictions.csv");
    predictions<<"image,method,candidate,trueNumber,number,predictedType,observedType,lightPercent,darkPercent,colouredPercent,bodyPixels,bodyColour,padding,originalWhiteCount,reservedCueCandidate\n";
    auto changes=file(directory/"changes.csv");
    changes<<"image,comparison,candidate,trueNumber,before,after,status\n";
    auto maskCounts=file(directory/"mask_counts.csv");
    maskCounts<<"image,candidate,crop,padding,total,light,dark,coloured,type\n";
    std::array<Counts,8> totals{};
    std::array<int,8> fixes{},losses{};
    for (const auto& filename:config::allImages()) {
        const auto image=cv::imread(config::mediaDirectory+filename);
        if (image.empty()) throw std::runtime_error("Could not load "+filename);
        const auto circles=runDetection(image,HoughParams{}).filteredCircles;
        const auto centres=transformCentres(circles);
        const std::string stem=std::filesystem::path(filename).stem().string();
        std::vector<BallROI> original;
        std::vector<BallTypeResult> k2,pixel;
        for (const auto& c:circles) {
            original.push_back(extractBall(image,c));
            k2.push_back(classifyBallType(segmentBallKMeans(original.back(),2)));
            pixel.push_back(pixelEvidence(original.back()).type);
        }
        std::array<Result,8> results;
        for (size_t m=0;m<methods.size();++m) {
            const auto& method=methods[m]; auto& r=results[m]; const auto& initial=method.pixels ? pixel : k2;
            double greatest=-1;
            for (size_t i=0;i<circles.size();++i) {
                const bool white=initial[i].type=="White";
                r.whites+=white; r.padding.push_back(method.padding && white ? 1 : 0);
                auto cropCircle=circles[i]; cropCircle[2]+=r.padding.back();
                r.rois.push_back(r.padding.back() ? extractBall(image,cropCircle) : original[i]);
                r.evidence.push_back(r.padding.back() ? (method.pixels ? pixelEvidence(r.rois.back()).type : classifyBallType(segmentBallKMeans(r.rois.back(),2))) : initial[i]);
                r.colours.push_back(scoreBallColours(r.rois.back(),model));
                const double relative=circles[i][2]/std::max(1.0,intercept+slope*circles[i][1]);
                if (white && relative>greatest) {greatest=relative; r.reservedCue=int(i);}
            }
            if (!method.size || r.whites<2) r.reservedCue=-1;
            if (method.unique) r.numbers=recogniseUniqueBallEvidence(r.evidence,r.colours,r.reservedCue);
            else for (size_t i=0;i<circles.size();++i) r.numbers.push_back(histogramBallNumber(r.evidence[i].type,r.colours[i].bestColour));
            for (size_t i=0;i<circles.size();++i) r.types.push_back(method.unique ? typeFor(r.numbers[i]) : r.evidence[i].type);
            saveFinalResults(image,directory/"final_results"/method.name/stem,circles,centres,r.types,r.numbers);
        }
        // Ground truth appears only after all inference, trigger choices and exports are complete.
        const auto truth=loadLabelledGroundTruth(config::groundTruthPath,filename);
        if (truth.empty()) throw std::runtime_error("Missing annotations for "+filename);
        std::vector<cv::Point2f> trueCentres; for (const auto& t:truth) trueCentres.push_back(t.centre);
        const auto e=evaluateDetections(centres,trueCentres,config::matchTolerance);
        auto actualFor=[&](size_t i) {const int j=e.predictionToGroundTruth[i]; return j<0 ? -1 : truth[j].number;};
        for (size_t m=0;m<methods.size();++m) {
            const auto& r=results[m]; Counts c{int(truth.size()),e.TP,e.FP,e.FN}; c.error=e.meanError*e.TP;
            for (size_t i=0;i<circles.size();++i) {
                const int actual=actualFor(i),n=r.numbers[i]; const auto& t=r.evidence[i];
                c.unknown+=n<0;
                if (actual>=0) {c.types+=r.types[i]==typeFor(actual); c.numbers+=n==actual; c.cueTotal+=actual==0; c.cues+=actual==0 && n==0;}
                predictions<<filename<<','<<methods[m].name<<','<<i+1<<','<<actual<<','<<n<<','<<r.types[i]<<','<<t.type<<','
                    <<t.lightPercent<<','<<t.darkPercent<<','<<t.colouredPercent<<','<<r.colours[i].pixels<<','<<r.colours[i].bestColour<<','
                    <<r.padding[i]<<','<<r.whites<<','<<(r.reservedCue<0 ? -1 : r.reservedCue+1)<<'\n';
                if (m%2) {
                    const int before=results[m-1].numbers[i];
                    const std::string status=before==n ? "unchanged" : actual<0 ? "unmatched" : n==actual ? "corrected" : before==actual ? "regressed" : "changed_still_wrong";
                    changes<<filename<<','<<methods[m-1].name<<"_to_"<<methods[m].name<<','<<i+1<<','<<actual<<','<<before<<','<<n<<','<<status<<'\n';
                    if (filename!="AllBalls.jpeg") {fixes[m]+=status=="corrected"; losses[m]+=status=="regressed";}
                }
            }
            metrics<<(filename=="AllBalls.jpeg" ? "training_resubstitution" : "evaluation")<<','<<filename<<','<<methods[m].name<<','; writeCounts(metrics,c); metrics<<'\n';
            if (filename!="AllBalls.jpeg") {
                auto& total=totals[m]; total.gt+=c.gt;total.tp+=c.tp;total.fp+=c.fp;total.fn+=c.fn;total.types+=c.types;
                total.numbers+=c.numbers;total.unknown+=c.unknown;total.error+=c.error;total.cues+=c.cues;total.cueTotal+=c.cueTotal;
            }
        }
        for (size_t i=0;i<circles.size();++i) {
            std::vector<ImagePanel> panels;
            for (size_t m=0;m<methods.size();++m) {
                const auto& r=results[m]; cv::Mat tile(320,330,CV_8UC3,cv::Scalar(28,28,28)),crop;
                cv::resize(r.rois[i].image,crop,{220,220},0,0,cv::INTER_NEAREST); crop.copyTo(tile(cv::Rect(55,0,220,220)));
                const int n=r.numbers[i],actual=actualFor(i);
                const auto ink=actual<0 ? cv::Scalar(200,200,200) : actual==n ? cv::Scalar(60,240,60) : cv::Scalar(60,60,255);
                auto text=[&](const std::string& str,int y,cv::Scalar colour) {cv::putText(tile,str,{8,y},cv::FONT_HERSHEY_SIMPLEX,.43,colour,1,cv::LINE_AA);};
                text("Actual "+numberName(actual)+" | assigned "+numberName(n),240,ink);
                text("Observed "+r.evidence[i].type+" | padding "+std::to_string(r.padding[i]),262,{220,220,220});
                std::ostringstream proportions; proportions<<std::fixed<<std::setprecision(1)<<"Light "<<r.evidence[i].lightPercent<<"% dark "<<r.evidence[i].darkPercent<<'%';
                text(proportions.str(),284,{220,220,220});
                text("Histogram body "+std::to_string(r.colours[i].pixels)+" pixels",306,{220,220,220});
                panels.push_back({methods[m].name,tile});
            }
            saveImage(directory/"candidate_comparisons"/stem/("candidate_"+std::to_string(i+1)+".png"),combineImages(panels,2));
            std::vector<ImagePanel> maskPanels;
            for (const int method : {3,5}) {
                const auto& r=results[method]; const auto p=pixelEvidence(r.rois[i]);
                maskCounts<<filename<<','<<i+1<<','<<(method==3 ? "original" : "HSV_white_pad1")<<','<<r.padding[i]<<','<<p.total<<','
                    <<cv::countNonZero(p.light)<<','<<cv::countNonZero(p.dark)<<','<<cv::countNonZero(p.coloured)<<','<<p.type.type<<'\n';
                for (const auto& panel:std::vector<ImagePanel>{{"Crop",r.rois[i].image},{"Light",p.light},{"Dark",p.dark},{"Coloured",p.coloured}}) {
                    cv::Mat resized;cv::resize(panel.image,resized,{200,200},0,0,cv::INTER_NEAREST);
                    maskPanels.push_back({(method==3 ? "Original " : "HSV-triggered +1 ")+panel.label,resized});
                }
            }
            saveImage(directory/"pixel_masks"/stem/("candidate_"+std::to_string(i+1)+".png"),combineImages(maskPanels,4));
        }
        std::cout<<filename<<": K2 / direct HSV pixel comparison complete\n";
    }
    auto summary=file(directory/"summary.md");
    summary<<"# Direct HSV pixel proportions versus K2 type evidence\n\n"
        <<"HSV profiles contain no K-means type/trigger stage. Light/dark/coloured fractions are counted directly using "
        <<"the existing thresholds, without tuning. Frozen AllBalls HSV histogram comparison supplies colour evidence.\n\n"
        <<"| Method | Correct types | Correct identities | Correct cues | Unknown | Fixes / losses vs paired K2 |\n|---|---:|---:|---:|---:|---:|\n";
    for (size_t m=0;m<methods.size();++m) {
        metrics<<"evaluation,ALL,"<<methods[m].name<<',';writeCounts(metrics,totals[m]);metrics<<'\n';
        summary<<"| "<<methods[m].name<<" | "<<totals[m].types<<'/'<<totals[m].tp<<" | "<<totals[m].numbers<<'/'<<totals[m].tp
            <<" | "<<totals[m].cues<<'/'<<totals[m].cueTotal<<" | "<<totals[m].unknown<<" | ";
        if (m%2) summary<<fixes[m]<<" / "<<losses[m]; else summary<<"—";
        summary<<" |\n";
    }
    summary<<"\nLight: V>=170 and S<=165. Dark: V<=90 outside light. Coloured: all remaining pixels. "
        <<"Masks partition the complete extraction disk; type fractions have no erosion. White>=85% light; "
        <<"solid>=70% nonlight; otherwise stripe when both occur. The independent histogram body mask still uses "
        <<"its frozen erosion3/light exclusion/min12 pixels. Histogram scores are 1-Bhattacharyya distance; "
        <<"backprojection images are not the classification score.\n\n"
        <<"Each frontend's original White labels control its own padding and ambiguous-cue trigger. Thus replacing "
        <<"type evidence may also change which crops expand and whether a cue is reserved. Original circles, coordinates "
        <<"and detections remain fixed. The row model is copied unchanged from the previous AllBalls-only fit.\n\n"
        <<"Unique scores use the same soft type intervals/width.10 and Hungarian assignment; largest-cue variants "
        <<"reserve cue for the greatest original radius/expected(row) among original White candidates only when >=2 exist. "
        <<"No threshold sweep, new histogram training or production integration. Table1–6 have been inspected; "
        <<"these are exploratory comparisons. AllBalls is training resubstitution, excluded from totals.\n\n"
        <<"Outputs: final_results/<method>/<image>/; candidate_comparisons/<image>/candidate_N.png; "
        <<"pixel_masks/<image>/candidate_N.png; mask_counts.csv; predictions.csv; metrics.csv; changes.csv; "
        <<"configuration.txt; frozen_models.yml; frozen_row_model.yml; synthetic_checks.txt.\n";
    std::cout<<"Saved direct HSV experiment to "<<directory<<'\n';
}
}
