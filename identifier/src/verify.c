#include "verify.h"
#include "catalog_db.h"
#include <math.h>
#include <string.h>

/**
 * Counts pairwise disagreements between observed brightness rank and catalog magnitude rank.
 */
static uint8_t brightness_magnitude_inversions(const ObservedStar *observed_stars, const MatchResult *result) {
    uint8_t inversion_count = 0;
    for (uint8_t first_match_index = 0; first_match_index < result->count; ++first_match_index) {
        const CatalogStar *first_star = catalog_get(result->hr_ids[first_match_index]);
        if (first_star == NULL) {
            continue;
        }
        uint32_t first_brightness = observed_stars[result->obs_ids[first_match_index]].brightness;
        for (uint8_t second_match_index = first_match_index + 1; second_match_index < result->count; ++second_match_index) {
            const CatalogStar *second_star = catalog_get(result->hr_ids[second_match_index]);
            if (second_star == NULL) {
                continue;
            }
            uint32_t second_brightness = observed_stars[result->obs_ids[second_match_index]].brightness;
            uint32_t brighter_margin = first_brightness > second_brightness
                ? first_brightness - second_brightness
                : second_brightness - first_brightness;
            if (brighter_margin < (first_brightness + second_brightness) / 20u) {
                continue;
            }
            if (first_brightness > second_brightness && first_star->mag_q100 > second_star->mag_q100) {
                ++inversion_count;
            }
            if (second_brightness > first_brightness && second_star->mag_q100 > first_star->mag_q100) {
                ++inversion_count;
            }
        }
    }
    return inversion_count;
}

/**
 * Verifies an attitude against catalog geometry without algorithm-specific logic.
 *
 * For each observed star, the candidate attitude is inverted (R^T) to rotate the
 * observed unit vector back into the catalog frame, then a KD-tree search finds
 * the nearest catalog star in O(log N) instead of the O(N) linear scan.  The
 * deduplication check prevents two observed stars from claiming the same catalog
 * star when a correct attitude brings multiple observed vectors near one catalog
 * position (e.g., double stars).
 */
bool verify_attitude(
    const Mat3f *rotation,
    const ObservedStar *observed_stars,
    uint8_t observed_star_count,
    MatchResult *result
) {
    memset(result, 0, sizeof(*result));
    for (uint8_t row_index = 0; row_index < 3; ++row_index) {
        for (uint8_t column_index = 0; column_index < 3; ++column_index) {
            result->catalog_to_observed[row_index][column_index] = rotation->m[row_index][column_index];
        }
    }
    uint32_t residual_sum = 0;
    for (uint8_t observed_index = 0; observed_index < observed_star_count && observed_index < MAX_MATCHES; ++observed_index) {
        float ox = observed_stars[observed_index].x;
        float oy = observed_stars[observed_index].y;
        float oz = observed_stars[observed_index].z;

        /* Rotate observed vector back into the catalog frame using R^T (= R^{-1} for rotations).
           This maps the observed direction to the catalog position it should match. */
        Vec3f cat_query = {
            rotation->m[0][0] * ox + rotation->m[1][0] * oy + rotation->m[2][0] * oz,
            rotation->m[0][1] * ox + rotation->m[1][1] * oy + rotation->m[2][1] * oz,
            rotation->m[0][2] * ox + rotation->m[1][2] * oy + rotation->m[2][2] * oz,
        };

        uint16_t best_residual;
        uint16_t best_hr = catalog_kd_nearest(cat_query, VERIFY_MAX_RESIDUAL_ARCSEC, &best_residual);
        if (best_hr == HR_INVALID) {
            continue;
        }

        /* Reject if another observed star already claimed this catalog star. */
        bool already_matched = false;
        for (uint8_t match_index = 0; match_index < result->count; ++match_index) {
            if (result->hr_ids[match_index] == best_hr) {
                already_matched = true;
                break;
            }
        }
        if (already_matched) {
            continue;
        }

        result->hr_ids[result->count] = best_hr;
        result->obs_ids[result->count] = observed_index;
        result->residual_arcsec[result->count] = best_residual;
        residual_sum += best_residual;
        if (best_residual > result->max_residual_arcsec) {
            result->max_residual_arcsec = best_residual;
        }
        ++result->count;
    }

    if (result->count > 0) {
        result->mean_residual_arcsec = (uint16_t)(residual_sum / result->count);
    }
    uint8_t inversion_count = brightness_magnitude_inversions(observed_stars, result);
    /* Inversions are a score penalty, not a hard gate; uncalibrated real images
       always have brightness/magnitude rank mismatches even on correct matches. */
    /* Require both enough inliers and a low mean residual: a geometrically wrong
       attitude can still gather VERIFY_MIN_INLIERS coincidental matches under the
       generous per-star residual cap, but only a correct attitude keeps the mean
       residual small. This rejects confident false positives on real images. */
    result->success = result->count >= VERIFY_MIN_INLIERS &&
                      result->mean_residual_arcsec <= VERIFY_MAX_MEAN_RESIDUAL_ARCSEC;
    result->score =
        (int32_t)result->count * 10000 -
        (int32_t)result->mean_residual_arcsec * 10 -
        result->max_residual_arcsec -
        (int32_t)inversion_count * 2000;
    return result->success;
}
