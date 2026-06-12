cmake_minimum_required(VERSION 3.20)

project(StarTracker LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(star_tracker opencv.cpp)

add_executable(hls_preprocess_file_test hls_preprocess.cpp test_run_with_file.cpp)
target_include_directories(hls_preprocess_file_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})