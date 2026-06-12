#ifndef IDENTIFY_PYRAMID_H
#define IDENTIFY_PYRAMID_H

#include <stdbool.h>
#include "star_types.h"

/**
 * Identifies one field using only the Pyramid pair database and catalog verification.
 */
bool identify_pyramid(const ObservedStar *observed_stars, uint8_t observed_star_count, MatchResult *result);

#endif
