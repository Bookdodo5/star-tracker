#ifndef VERIFY_H
#define VERIFY_H

#include <stdbool.h>
#include "star_math.h"
#include "star_types.h"

#define VERIFY_MIN_INLIERS 6
#define VERIFY_MAX_RESIDUAL_ARCSEC 900u

/**
 * Checks one attitude against the catalog without knowing which algorithm made it.
 */
bool verify_attitude(
    const Mat3f *rotation,
    const ObservedStar *observed_stars,
    uint8_t observed_star_count,
    MatchResult *result
);

#endif
