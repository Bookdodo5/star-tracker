#ifndef ATTITUDE_H
#define ATTITUDE_H

#include <stdbool.h>
#include "star_math.h"
#include "star_types.h"

/**
 * Solves catalog-to-observed attitude from the first two non-collinear matches.
 */
bool solve_attitude_triad(
    const uint16_t *hr_ids,
    const uint8_t *obs_ids,
    uint8_t count,
    const ObservedStar *observed_stars,
    Mat3f *rotation
);

#endif
