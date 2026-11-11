cmake_minimum_required(VERSION 3.16)
project(PoolBallDetection LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV 4 REQUIRED)

# Both names build the current application, independently of the archived modules.
foreach(application_target IN ITEMS PoolBallLearning PoolBallSubmission)
    add_executable(${application_target} myApplication.cpp)
    target_compile_definitions(${application_target} PRIVATE POOL_BALL_STANDALONE)
    target_include_directories(${application_target} PRIVATE ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(${application_target} PRIVATE ${OpenCV_LIBS})
endforeach()

# The modular tools have their own CMake project in more_detailed_implementation/.
