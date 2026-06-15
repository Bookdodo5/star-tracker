#include "catalog_db.h"
#include <math.h>
#include <stddef.h>

#define CATALOG_KD_STACK_MAX 64

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

/**
 * Finds the catalog star nearest to a query unit vector using the prebuilt KD-tree.
 *
 * The query must be in the catalog (celestial) frame — typically the observed unit
 * vector after applying the transpose of the candidate attitude rotation.  Uses
 * squared Euclidean distance as the search metric; for unit vectors the nearest
 * Euclidean neighbor is identical to the nearest angular neighbor.
 *
 * The initial search radius is set to max_residual_arcsec so the tree prunes
 * subtrees that cannot improve the best candidate, keeping the search O(log N)
 * in the common case where only a few catalog stars are within tolerance.
 */
uint16_t catalog_kd_nearest(Vec3f query, uint16_t max_residual_arcsec, uint16_t *residual_out) {
    if (catalog_kd_node_count == 0) {
        return HR_INVALID;
    }

    /* Convert the angular tolerance to a squared chord-length threshold.
       chord = 2*sin(angle/2); for the small angles used here chord ≈ angle_rad. */
    float max_angle_rad = (float)max_residual_arcsec / 206264.806247f;
    float half_chord = sinf(max_angle_rad * 0.5f);
    float best_dist_sq = 4.0f * half_chord * half_chord;
    uint16_t best_hr = HR_INVALID;

    int32_t stack[CATALOG_KD_STACK_MAX];
    uint8_t stack_size = 0;
    stack[stack_size++] = 0;

    while (stack_size > 0) {
        int32_t node_id = stack[--stack_size];
        if (node_id < 0 || (uint32_t)node_id >= catalog_kd_node_count) {
            continue;
        }
        const CatalogKdNode *node = &catalog_kd_nodes[node_id];
        float dx = query.x - node->x;
        float dy = query.y - node->y;
        float dz = query.z - node->z;
        float dist_sq = dx * dx + dy * dy + dz * dz;
        if (dist_sq < best_dist_sq) {
            best_dist_sq = dist_sq;
            best_hr = node->hr;
        }

        float axis_delta;
        if (node->axis == 0) {
            axis_delta = query.x - node->x;
        } else if (node->axis == 1) {
            axis_delta = query.y - node->y;
        } else {
            axis_delta = query.z - node->z;
        }
        int32_t near = axis_delta <= 0.0f ? node->left : node->right;
        int32_t far  = axis_delta <= 0.0f ? node->right : node->left;

        /* Prune the far subtree when the splitting hyperplane is farther than best_dist_sq. */
        if (axis_delta * axis_delta < best_dist_sq && stack_size < CATALOG_KD_STACK_MAX) {
            stack[stack_size++] = far;
        }
        if (stack_size < CATALOG_KD_STACK_MAX) {
            stack[stack_size++] = near;
        }
    }

    if (best_hr == HR_INVALID) {
        return HR_INVALID;
    }
    /* Convert the winning chord distance to arcseconds. */
    float angle_rad = 2.0f * asinf(sqrtf(best_dist_sq) * 0.5f);
    float arcsec = angle_rad * 206264.806247f;
    *residual_out = arcsec >= 65535.0f ? 65535u : (uint16_t)(arcsec + 0.5f);
    return best_hr;
}
