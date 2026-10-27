cmake_minimum_required(VERSION 3.16)
project(PoolBallDetection LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV 4 REQUIRED)

add_executable(PoolBallDetection
    main.cpp
    myApplication.cpp

    detection/ballDetection.cpp
    detection/geometricFilter.cpp

    recognition/extractBall.cpp
    recognition/colourAnalysis.cpp
    recognition/kMeansSegmentation.cpp
    recognition/ballTypeClassifier.cpp
    recognition/connectedComponentAnalysis.cpp

    geometry/perspectiveTransform.cpp

    visualisation/visualisation.cpp
    visualisation/combineImages.cpp

    evaluation/evaluation.cpp
    evaluation/gridSearch.cpp
    evaluation/sanityCheck.cpp
    evaluation/extractionDemo.cpp
    evaluation/colourAnalysisDemo.cpp
    evaluation/kMeansDemo.cpp

    validation/validation.cpp
    validation/kMeansValidation.cpp

    io/output.cpp
)

target_include_directories(PoolBallDetection PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(PoolBallDetection PRIVATE ${OpenCV_LIBS})

# Independent runner: the main application and recognition code stay unchanged.
add_executable(BallExperiments
    experiments/main.cpp
    experiments/radiusPadding.cpp
    evaluation/evaluation.cpp
    detection/ballDetection.cpp
    detection/geometricFilter.cpp
    geometry/perspectiveTransform.cpp
    recognition/extractBall.cpp
    recognition/kMeansSegmentation.cpp
    recognition/ballTypeClassifier.cpp
)
target_include_directories(BallExperiments PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(BallExperiments PRIVATE ${OpenCV_LIBS})
