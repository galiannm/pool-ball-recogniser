#pragma once

#include <vector>

namespace experiments {

// All threshold values use OpenCV HSV units (H: 0-179, S/V: 0-255)
// or percentages (0-100). These defaults reproduce the production rules.
struct RecognitionParameters {
    int k = 2;
    bool cloth = false;
    int lightMinV = 170;
    int lightMaxS = 165;
    int darkMaxV = 90; // Reporting only: type decisions use all non-light pixels.
    int whiteMinPercent = 85;
    int solidMinPercent = 70;
    int redUpper = 6;
    int orangeUpper = 15;
    int yellowUpper = 35;
    int greenLower = 45;
    int greenUpper = 90;
    int blueLower = 100;
    int bluePurple = 130;
    int purpleUpper = 165;
    int redWrap = 170;
    int maroonMaxV = 150; // Strict V < threshold, matching the existing code.
    int blackMaxS = 60;   // Strict S < threshold.
    int blackMaxV = 90;   // Inclusive V <= threshold.
    int bodyMinPercent = 2;
};

struct RecognitionSearchGrid {
    std::vector<int> ks = {2, 3, 4, 5, 6};
    std::vector<bool> cloth = {false, true};
    std::vector<int> lightMinV = {150, 170, 190};
    std::vector<int> lightMaxS = {125, 145, 165, 185};
    std::vector<int> whiteMinPercent = {80, 85, 90};
    std::vector<int> solidMinPercent = {60, 65, 70, 75, 80};

    std::vector<int> redUpper = {4, 6, 8};
    std::vector<int> orangeUpper = {12, 15, 18};
    std::vector<int> yellowUpper = {30, 35, 40};
    std::vector<int> redWrap = {168, 170, 172};
    std::vector<int> maroonMaxV = {130, 150, 170};

    std::vector<int> greenLower = {40, 45, 50};
    std::vector<int> greenUpper = {85, 90, 95};
    std::vector<int> blueLower = {95, 100, 105};
    std::vector<int> bluePurple = {125, 130, 135};
    std::vector<int> purpleUpper = {160, 165, 168};

    std::vector<int> blackMaxS = {40, 60, 80};
    std::vector<int> blackMaxV = {70, 90, 110};
    std::vector<int> bodyMinPercent = {1, 2, 5};

    int shortlistPerObjective = 3;
    int colourPasses = 2;
};

} // namespace experiments
