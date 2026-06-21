cmake_minimum_required(VERSION 3.20)

project(StarLive LANGUAGES C CXX)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)

set(ID ${CMAKE_CURRENT_SOURCE_DIR}/../identifier)

# One shared library holding the whole in-process chain: centroid detector (C++) +
# TETRA identifier + baked databases (C). Loaded once by Python via ctypes.
add_library(star_live SHARED
    live_identify.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../Centroid/centroid_extract.cpp
    ${ID}/src/attitude.c
    ${ID}/src/camera_model.c
    ${ID}/src/catalog_db.c
    ${ID}/src/identify_tetra.c
    ${ID}/src/star_math.c
    ${ID}/src/verify.c
    ${ID}/generated/catalog_db_generated.c
    ${ID}/generated/tetra_db_generated.c
)

target_include_directories(star_live PRIVATE ${ID}/include)
target_link_libraries(star_live PRIVATE m)

# Embed the MinGW runtime so ctypes can load the DLL without MinGW on PATH.
target_link_options(star_live PRIVATE -static -static-libgcc -static-libstdc++)
