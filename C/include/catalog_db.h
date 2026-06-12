#ifndef CATALOG_DB_H
#define CATALOG_DB_H

#include "star_math.h"
#include "star_types.h"

extern const CatalogStar catalog_stars[];
extern const uint16_t catalog_star_count;
extern const uint16_t hr_to_catalog_index[];
extern const uint16_t hr_to_catalog_index_count;

/**
 * Returns a catalog star by HR ID, or NULL when absent.
 */
const CatalogStar *catalog_get(uint16_t hr_id);

/**
 * Returns a catalog star unit vector by HR ID.
 */
bool catalog_vector(uint16_t hr_id, Vec3f *catalog_vector_result);

#endif
