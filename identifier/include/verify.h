#ifndef VERIFY_H
#define VERIFY_H

#include <stdbool.h>
#include "star_math.h"
#include "star_types.h"

#define VERIFY_MIN_INLIERS 6
#define VERIFY_MAX_RESIDUAL_ARCSEC 900u
/* Mean inlier residual gate. On real images, correct solves cluster well below this
   (measured ~20-80 arcsec) while geometrically-coincidental false matches sit far
   above it (measured ~190-490 arcsec). Gating on the mean — not the score, which
   does not separate the two — stops both algorithms from reporting confident wrong
   attitudes. Synthetic solves have ~0 mean residual and are unaffected. */
#define VERIFY_MAX_MEAN_RESIDUAL_ARCSEC 150u

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
