#ifndef CATALOG_DB_H
#define CATALOG_DB_H

#include "star_math.h"
#include "star_types.h"

/** KD-tree node over catalog unit vectors for O(log N) nearest-star lookup. */
typedef struct {
    float x, y, z;    /* catalog unit vector in float */
    float _pad;        /* explicit pad to align to 20 bytes */
    int32_t left;      /* left child index, -1 if none */
    int32_t right;     /* right child index */
    uint16_t hr;       /* catalog HR number */
    uint8_t axis;      /* split dimension: 0=x, 1=y, 2=z */
    uint8_t _pad2;
} CatalogKdNode;

extern const CatalogStar catalog_stars[];
extern const uint16_t catalog_star_count;
extern const uint16_t hr_to_catalog_index[];
extern const uint16_t hr_to_catalog_index_count;
extern const CatalogKdNode catalog_kd_nodes[];
extern const uint32_t catalog_kd_node_count;

/**
 * Returns a catalog star by HR ID, or NULL when absent.
 */
const CatalogStar *catalog_get(uint16_t hr_id);

/**
 * Returns a catalog star unit vector by HR ID.
 */
bool catalog_vector(uint16_t hr_id, Vec3f *catalog_vector_result);

/**
 * Finds the nearest catalog star to a query unit vector (catalog frame, float).
 * Returns HR_INVALID when no catalog star lies within max_residual_arcsec.
 * Writes the angular residual in arcseconds to residual_out.
 */
uint16_t catalog_kd_nearest(Vec3f query, uint16_t max_residual_arcsec, uint16_t *residual_out);

#endif
