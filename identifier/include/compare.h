#ifndef COMPARE_H
#define COMPARE_H

#include <stdint.h>
#include "star_types.h"

typedef struct {
    MatchResult tetra;
    MatchResult pyramid;
    uint32_t tetra_us;
    uint32_t pyramid_us;
} CompareResult;

/**
 * Runs TETRA and Pyramid independently on the same observed stars.
 */
CompareResult compare_tetra_pyramid(const ObservedStar *observed_stars, uint8_t observed_star_count);

#endif
