// Report-only diagnostics: reuse the application methods without running MyApplication().
#include "../myApplication.cpp"

int main() {
    try {
        const std::filesystem::path out = "report v5/assets";
        std::filesystem::create_directories(out);
        const auto model = buildColourReferenceModel(cv::imread(MEDIA_DIR + TUNNNING_IMG));
        std::ofstream evidence(out / "candidate_evidence.csv");
        evidence << "image,candidate,trueNumber,predictedNumber,distance,x,y,radius,type,lightPercent,bestColour,paddedType,paddedLightPercent,paddedBestColour\n";
        std::ofstream scores(out / "colour_scores.csv");
        scores << "image,candidate,trueNumber,yellow,blue,red,purple,orange,green,maroon,black\n";
        for (const auto& filename : getAllImages()) {
            const auto stem = std::filesystem::path(filename).stem().string();
            const auto image = cv::imread(MEDIA_DIR + filename);
            std::vector<cv::Vec3f> circles;
            std::vector<int> numbers;
            std::ifstream file(std::filesystem::path(FINAL_RES_DIR) / stem / "balls.csv");
            if (!file) throw std::runtime_error("Missing saved predictions for " + filename);
            std::string line;
            std::getline(file, line);
            while (std::getline(file, line)) {
                std::stringstream row(line);
                std::vector<std::string> fields;
                std::string field;
                while (std::getline(row, field, ',')) fields.push_back(field);
                numbers.push_back(std::stoi(fields[2]));
                circles.emplace_back(std::stof(fields[3]),std::stof(fields[4]),std::stof(fields[5]));
            }
            const auto context = prepareEvaluation(circles, filename);
            if (stem == "AllBalls") {
                const auto masked = maskTable(image);
                cv::Mat grey, blur;
                cv::cvtColor(masked, grey, cv::COLOR_BGR2GRAY);
                cv::GaussianBlur(grey, blur, {5,5}, 1.5);
                saveImage(out / "detection_grey_blur.png", blur);
            }
            if (stem=="Table3") {
                const auto guided = detectGuidedHoughWatershed(image, circles);
                saveImage(out/"guided_Table3_baseline.png", drawDetectedBalls(image, circles));
                saveImage(out/"guided_Table3_local.png", drawDetectedBalls(image, guided.guided));
                saveImage(out/"guided_Table3_watershed.png", drawDetectedBalls(image, guided.watershed));
                for(size_t d=0;d<guided.diagnostics.size();++d)
                    saveImage(out/("guided_Table3_diagnostic_"+std::to_string(d)+".png"),guided.diagnostics[d].image);
            }
            for (size_t i=0;i<circles.size();++i) {
                const int g = context.matching.predictionToGroundTruth[i];
                const int actual = g < 0 ? -1 : context.truth[g].number;
                const auto roi = extractBall(image, circles[i]);
                const auto segmentation = segmentBallKMeans(roi, 2);
                const auto type = classifyBallType(segmentation);
                const auto colour = scoreBallColours(roi, model);
                auto grownCircle = circles[i]; grownCircle[2] += 1;
                const auto grown = extractBall(image, grownCircle);
                const auto grownSeg = segmentBallKMeans(grown, 2);
                const auto grownType = classifyBallType(grownSeg);
                const auto grownColour = scoreBallColours(grown, model);
                const double distance = g < 0 ? -1 : cv::norm(transformCentres({circles[i]})[0]-context.truth[g].centre);
                evidence << stem << ',' << i+1 << ',' << actual << ',' << numbers[i] << ',' << distance << ',' << circles[i][0] << ',' << circles[i][1] << ',' << circles[i][2] << ',' << type.type << ',' << type.lightPercent << ',' << colour.bestColour << ',' << grownType.type << ',' << grownType.lightPercent << ',' << grownColour.bestColour << '\n';
                scores << stem << ',' << i+1 << ',' << actual;
                for (double s : colour.similarity) scores << ',' << s;
                scores << '\n';
                const bool selected = (stem=="Table2" && i==2) || (stem=="Table3" && i==1) || (stem=="Table4" && i==8) || (stem=="Table2" && (actual==1||actual==9||actual==8)) || (stem=="Table1" && (type.type=="White" || actual==9)) || (stem=="Table6" && (actual==10||actual==3)) || (stem=="Table3" && actual==12);
                if (!selected) continue;
                const auto prefix = stem + "_candidate_" + std::to_string(i+1);
                saveImage(out / (prefix+"_original.png"), roi.image);
                saveImage(out / (prefix+"_mask.png"), roi.mask);
                const auto cca = experimentCCA(roi, segmentation);
                saveImage(out / (prefix+"_cca_light.png"), cca.lightMask);
                saveImage(out / (prefix+"_cca_nonlight.png"), cca.nonLightMask);
                saveImage(out / (prefix+"_kmeans.png"), segmentation.segmentedImage);
                cv::Mat body=cv::Mat::zeros(roi.image.size(),CV_8UC3);
                roi.image.copyTo(body,colour.bodyMask);
                saveImage(out / (prefix+"_body.png"), body);
                saveImage(out / (prefix+"_padded.png"), grown.image);
                saveImage(out / (prefix+"_padded_kmeans.png"), grownSeg.segmentedImage);
                if (stem=="Table6" || stem=="Table1") {
                    cv::Rect crop(cvRound(circles[i][0])-65,cvRound(circles[i][1])-55,130,110);
                    crop &= cv::Rect(0,0,image.cols,image.rows);
                    const auto labelled=cv::imread((std::filesystem::path(FINAL_RES_DIR)/stem/"original_numbered.png").string());
                    saveImage(out/(prefix+"_context.png"),labelled(crop));
                }
            }
        }
        finishOutputFile(evidence); finishOutputFile(scores);
        return 0;
    } catch(const std::exception& e) {std::cerr << e.what() << '\n'; return 1;}
}
