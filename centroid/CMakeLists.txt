cmake_minimum_required(VERSION 3.16)
project(StarCentroid CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(centroid_extract
    centroid_extract.cpp
    centroid_cli.cpp
)
