#ifndef IDENTIFY_TETRA_H
#define IDENTIFY_TETRA_H

#include <stdbool.h>
#include "star_types.h"

/**
 * Identifies one field using only the TETRA database and catalog verification.
 */
bool identify_tetra(const ObservedStar *observed_stars, uint8_t observed_star_count, MatchResult *result);

#endif
