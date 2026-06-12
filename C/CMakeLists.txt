cmake_minimum_required(VERSION 3.20)

project(StarIdentifierC LANGUAGES C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

add_library(star_identifier
    src/attitude.c
    src/camera_model.c
    src/catalog_db.c
    src/compare.c
    src/identify_pyramid.c
    src/identify_tetra.c
    src/star_math.c
    src/verify.c
)

target_include_directories(star_identifier PUBLIC include)
target_link_libraries(star_identifier PUBLIC m)

add_executable(test_star_identifier tests/test_core.c)
target_link_libraries(test_star_identifier PRIVATE star_identifier)

add_library(star_identifier_generated
    src/attitude.c
    src/camera_model.c
    src/catalog_db.c
    src/compare.c
    src/identify_pyramid.c
    src/identify_tetra.c
    src/star_math.c
    src/verify.c
    generated/catalog_db_generated.c
    generated/pyramid_db_generated.c
    generated/tetra_db_generated.c
)

target_include_directories(star_identifier_generated PUBLIC include)
target_link_libraries(star_identifier_generated PUBLIC m)

add_executable(demo_centroid_compare src/demo_centroid_compare.c)
target_link_libraries(demo_centroid_compare PRIVATE star_identifier_generated)

add_executable(batch_synthetic_compare src/batch_synthetic_compare.c)
target_link_libraries(batch_synthetic_compare PRIVATE star_identifier_generated)
