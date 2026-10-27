
#include "evaluation/gridSearch.h"
#include "evaluation/sanityCheck.h"

#include "evaluation/colourAnalysisDemo.h"
#include "evaluation/extractionDemo.h"
#include "evaluation/kMeansDemo.h"

#include "validation/validation.h"
#include "validation/kMeansValidation.h"

#include <exception>
#include <iostream>
#include <sstream>
#include <string>

#include "myApplication.h"

enum class RunMode
{
    Application = 1,
    Baseline,
    GridSearch, 
    Validation,
    SanityCheck,
    ExtractBall,
    ColourAnalysis,
    kMeans,
    kMeansValidation
};

int main()
{
    std::cout << "\n1. Application\n"
              << "2. Baseline\n"
              << "3. Grid search\n"
              << "4. Validation\n"
              << "5. Sanity check\n"
              << "6. Extract balls\n"
              << "7. HLS Colour Analysis\n"
              << "8. kMeans \n"
              << "9. kMeans Validation \n"
              << "0. Exit\n\n";

    int choice = 0;
    while (true)
    {
        std::cout << "Select a mode (0-9): " << std::flush;

        std::string input;
        if (!std::getline(std::cin, input))
            return 0;

        std::istringstream selection(input);
        char extra;
        if (!(selection >> choice) || (selection >> extra) ||
            choice < 0 || choice > 9)
        {
            std::cout << "Invalid selection. Enter a number from 0 to 6.\n";
            continue;
        }

        if (choice == 0)
            return 0;

        break;
    }

    RunMode mode = static_cast<RunMode>(choice);

    try
    {
        switch (mode)
        {
            case RunMode::Application:
                MyApplication();
                break;
    
            case RunMode::Baseline:
                testCurrentParameters();
                break;
    
            case RunMode::GridSearch:
                runGridSearch();
                break;
    
            case RunMode::Validation:
                runValidation();
                break;
            
            case RunMode::SanityCheck:
                runSanityCheck();
                break;
            
            case RunMode::ExtractBall:
                testBallExtraction();
                break;
            
            case RunMode::ColourAnalysis:
                testColourAnalysis();
                break;
            
            case RunMode::kMeans:
                testKMeansSegmentation();
                break;

            case RunMode::kMeansValidation:
                runRecognitionValidation();
                break;
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
