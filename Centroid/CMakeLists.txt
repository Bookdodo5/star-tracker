cmake_minimum_required(VERSION 3.20)

project(StarTrackerCentroid LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Centroid extractor: image (PPM) -> star centroids (CSV).
# centroid_extract.cpp performs the detection; centroid_cli.cpp is the command-line runner.
add_executable(centroid_extract centroid_extract.cpp centroid_cli.cpp)
target_include_directories(centroid_extract PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
