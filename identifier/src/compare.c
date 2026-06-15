#include "compare.h"
#include "clock_utils.h"
#include "identify_pyramid.h"
#include "identify_tetra.h"
#include <time.h>

/**
 * Runs TETRA and Pyramid independently on the same observed stars.
 */
CompareResult compare_tetra_pyramid(const ObservedStar *observed_stars, uint8_t observed_star_count) {
    CompareResult comparison_result;
    /* TETRA result is recorded before Pyramid runs, so neither can affect the other. */
    clock_t start = clock();
    identify_tetra(observed_stars, observed_star_count, &comparison_result.tetra);
    clock_t after_tetra = clock();
    identify_pyramid(observed_stars, observed_star_count, &comparison_result.pyramid);
    clock_t after_pyramid = clock();
    comparison_result.tetra_us = elapsed_us(start, after_tetra);
    comparison_result.pyramid_us = elapsed_us(after_tetra, after_pyramid);
    return comparison_result;
}
