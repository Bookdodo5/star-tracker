#ifndef IDENTIFY_TETRA_H
#define IDENTIFY_TETRA_H

#include <stdbool.h>
#include "star_types.h"

/**
 * Identifies one field using only the TETRA database and catalog verification.
 */
bool identify_tetra(const ObservedStar *observed_stars, uint8_t observed_star_count, MatchResult *result);

/**
 * Identifies one field when the seed FOV may be far from the true FOV.
 *
 * The TETRA feature lookup is scale-invariant (edge ratios), so it finds the right
 * tetrad at any FOV. This routine uses a matched tetrad's true catalog angles to
 * recover the focal-length scale, rescales the whole observed set, and re-runs the
 * normal identifier. On success, result->focal_scale = f_true / f_seed so the caller
 * can lock the recovered FOV. Bootstrap-only (more expensive than identify_tetra);
 * later frames should reuse the locked FOV with identify_tetra.
 */
bool identify_tetra_calibrate(const ObservedStar *observed_stars, uint8_t observed_star_count, MatchResult *result);

#endif
