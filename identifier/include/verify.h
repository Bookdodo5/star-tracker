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
#define VERIFY_MAX_MEAN_RESIDUAL_ARCSEC 120u
/* Minimum inlier fraction (integer percent). A correct attitude has most observed
   stars matching the catalog; false matches barely reach VERIFY_MIN_INLIERS while
   the remaining detections are outliers. Gate: count*100 >= observed*this value.
   35% rejects 6/20 (30%) false positives while passing 8/20 (40%) correct solves
   in noisy frames where non-star detections inflate the denominator. */
#define VERIFY_MIN_INLIER_PCT 35u

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
