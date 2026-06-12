#include "catalog_db.h"
#include <stddef.h>

/**
 * Looks up a catalog star through the generated HR-to-index table.
 */
const CatalogStar *catalog_get(uint16_t hr_id) {
    if (hr_id >= hr_to_catalog_index_count) {
        return NULL;
    }
    uint16_t catalog_index = hr_to_catalog_index[hr_id];
    if (catalog_index == HR_INVALID || catalog_index >= catalog_star_count) {
        return NULL;
    }
    return &catalog_stars[catalog_index];
}

/**
 * Returns a catalog star vector in float form for geometric calculations.
 */
bool catalog_vector(uint16_t hr_id, Vec3f *catalog_vector_result) {
    const CatalogStar *star = catalog_get(hr_id);
    if (star == NULL) {
        return false;
    }
    *catalog_vector_result = q15_to_vec3f(star->x, star->y, star->z);
    return true;
}
