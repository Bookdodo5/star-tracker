#include "verify.h"
#include "catalog_db.h"
#include <math.h>
#include <string.h>

/**
 * Converts radians to arcseconds with uint16 saturation.
 */
static uint16_t rad_to_arcsec(float radians) {
    float arcsec = radians * 206264.806247f;
    if (arcsec <= 0.0f) return 0;
    if (arcsec >= 65535.0f) return 65535u;
    return (uint16_t)(arcsec + 0.5f);
}

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
        Vec3f observed_vector = {
            observed_stars[observed_index].x,
            observed_stars[observed_index].y,
            observed_stars[observed_index].z,
        };
        uint16_t best_hr = HR_INVALID;
        uint16_t best_residual = 65535u;

        /* Each observed star independently chooses the nearest rotated catalog star. */
        for (uint16_t catalog_index = 0; catalog_index < catalog_star_count; ++catalog_index) {
            bool hr_already_matched = false;
            for (uint8_t match_index = 0; match_index < result->count; ++match_index) {
                if (result->hr_ids[match_index] == catalog_stars[catalog_index].hr) {
                    hr_already_matched = true;
                    break;
                }
            }
            if (hr_already_matched) {
                continue;
            }
            Vec3f catalog_vector_value = q15_to_vec3f(
                catalog_stars[catalog_index].x,
                catalog_stars[catalog_index].y,
                catalog_stars[catalog_index].z
            );
            Vec3f predicted_vector = mat3_mul_vec3(rotation, catalog_vector_value);
            uint16_t residual = rad_to_arcsec(angular_distance_rad(predicted_vector, observed_vector));
            if (residual < best_residual) {
                best_residual = residual;
                best_hr = catalog_stars[catalog_index].hr;
            }
        }

        /* Only residual inliers are exposed as verified matches. */
        if (best_hr != HR_INVALID && best_residual <= VERIFY_MAX_RESIDUAL_ARCSEC) {
            result->hr_ids[result->count] = best_hr;
            result->obs_ids[result->count] = observed_index;
            result->residual_arcsec[result->count] = best_residual;
            residual_sum += best_residual;
            if (best_residual > result->max_residual_arcsec) {
                result->max_residual_arcsec = best_residual;
            }
            ++result->count;
        }
    }

    if (result->count > 0) {
        result->mean_residual_arcsec = (uint16_t)(residual_sum / result->count);
    }
    uint8_t inversion_count = brightness_magnitude_inversions(observed_stars, result);
    result->success = result->count >= VERIFY_MIN_INLIERS && inversion_count == 0u;
    result->score =
        (int32_t)result->count * 10000 -
        (int32_t)result->mean_residual_arcsec * 10 -
        result->max_residual_arcsec -
        (int32_t)inversion_count * 2000;
    return result->success;
}
