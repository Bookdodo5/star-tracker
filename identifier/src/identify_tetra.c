#include "identify_tetra.h"
#include "attitude.h"
#include "catalog_db.h"
#include "clock_utils.h"
#include "star_math.h"
#include "tetra_db.h"
#include "verify.h"
#include <math.h>
#include <string.h>

#define TETRA_MAX_QUERY_STARS 16
#define TETRA_TOP_CANDIDATES 8
/** Exit once a match this good is found; avoids exhausting all 210 4-tuples. */
#define TETRA_EARLY_EXIT_SCORE 70000
#define KD_STACK_MAX 96
#define KD_NODE_VISIT_LIMIT 32768
#define TETRA_PATTERN_MAX_RESIDUAL_RAD 0.010471976f
/** FOV self-calibration: focal-correction iterations and convergence/sanity bounds. */
#define CALIB_MAX_ITERS 6
#define CALIB_CONVERGE_TOL 0.001f
#define CALIB_MIN_RATIO 0.05f
#define CALIB_MAX_RATIO 20.0f
/** Wall-clock budget per calibrate call. The bootstrap does O(C(query,4)*candidates) FULL
    re-identifies, and a single inner identify_tetra on a dense wrong-FOV frame can take ~0.5s,
    so a non-solving frame runs for minutes. Checked per candidate (not per seed) since inner
    cost varies ~1000x. Past this, bail so the caller moves to the next frame. ~2s keeps genuine
    solves (which early-exit well under it) while capping a hopeless frame. */
#define CALIB_BUDGET_US 2000000u

typedef struct {
    uint32_t node_id;
    uint32_t error;
} TetraCandidate;

static const uint8_t TETRA_PERMUTATIONS[24][4] = {
    {0, 1, 2, 3}, {0, 1, 3, 2}, {0, 2, 1, 3}, {0, 2, 3, 1},
    {0, 3, 1, 2}, {0, 3, 2, 1}, {1, 0, 2, 3}, {1, 0, 3, 2},
    {1, 2, 0, 3}, {1, 2, 3, 0}, {1, 3, 0, 2}, {1, 3, 2, 0},
    {2, 0, 1, 3}, {2, 0, 3, 1}, {2, 1, 0, 3}, {2, 1, 3, 0},
    {2, 3, 0, 1}, {2, 3, 1, 0}, {3, 0, 1, 2}, {3, 0, 2, 1},
    {3, 1, 0, 2}, {3, 1, 2, 0}, {3, 2, 0, 1}, {3, 2, 1, 0},
};

/**
 * Returns the absolute value of a signed 32-bit integer as uint32.
 */
static uint32_t abs_i32(int32_t value) {
    return (uint32_t)(value < 0 ? -value : value);
}

/**
 * Computes L1 distance between two quantized five-dimensional TETRA features.
 */
static uint32_t feature_error(const uint16_t query_feature[5], const uint16_t database_feature[5]) {
    uint32_t error = 0;
    for (uint8_t feature_index = 0; feature_index < 5; ++feature_index) {
        error += abs_i32((int32_t)query_feature[feature_index] - (int32_t)database_feature[feature_index]);
    }
    return error;
}

/**
 * Inserts a KD-tree candidate while keeping the candidate list sorted by error.
 */
static void add_candidate(TetraCandidate *items, uint8_t *count, uint32_t node_id, uint32_t error) {
    uint8_t pos = *count;
    if (pos < TETRA_TOP_CANDIDATES) {
        ++(*count);
    } else if (error >= items[pos - 1].error) {
        return;
    } else {
        pos = TETRA_TOP_CANDIDATES - 1;
    }
    while (pos > 0 && items[pos - 1].error > error) {
        items[pos] = items[pos - 1];
        --pos;
    }
    items[pos] = (TetraCandidate){node_id, error};
}

/**
 * Searches the array KD-tree and returns the nearest TETRA feature candidates.
 */
static uint8_t tetra_kd_search_topk(const uint16_t query_feature[5], TetraCandidate *nearest_candidates) {
    int32_t stack[KD_STACK_MAX];
    uint8_t stack_size = 0;
    uint8_t nearest_candidate_count = 0;
    uint32_t best_error = 0xFFFFFFFFu;
    uint32_t visited_node_count = 0;

    if (tetra_kd_node_count == 0) {
        return 0;
    }
    stack[stack_size++] = 0;

    while (stack_size > 0) {
        if (++visited_node_count > KD_NODE_VISIT_LIMIT) {
            break;
        }
        int32_t node_id = stack[--stack_size];
        if (node_id == KD_NULL || (uint32_t)node_id >= tetra_kd_node_count) {
            continue;
        }

        const TetraKdNode *node = &tetra_kd_nodes[node_id];
        uint32_t error = feature_error(query_feature, node->f);
        add_candidate(nearest_candidates, &nearest_candidate_count, (uint32_t)node_id, error);
        if (nearest_candidate_count == TETRA_TOP_CANDIDATES) {
            best_error = nearest_candidates[nearest_candidate_count - 1].error;
        }

        int32_t split_axis_delta = (int32_t)query_feature[node->axis] - (int32_t)node->f[node->axis];
        int32_t near_node_id = split_axis_delta < 0 ? node->left : node->right;
        int32_t far_node_id = split_axis_delta < 0 ? node->right : node->left;

        /* The far branch can only improve the result if its axis distance is small enough. */
        if (abs_i32(split_axis_delta) <= best_error && stack_size < KD_STACK_MAX) {
            stack[stack_size++] = far_node_id;
        }
        if (stack_size < KD_STACK_MAX) {
            stack[stack_size++] = near_node_id;
        }
    }
    return nearest_candidate_count;
}

/**
 * Builds the normalized five-edge feature used for TETRA nearest-neighbor lookup.
 *
 * Mirrors the Python DB builder exactly: compute all six angular distances in radians,
 * sort, then normalize each by the longest edge before quantizing to uint16.  Using
 * float ratios instead of pre-quantizing each edge to the π scale eliminates the
 * ~31-unit systematic L1 error that would otherwise cause KD-tree misses.
 */
static bool make_tetra_feature(const ObservedStar *observed_stars, const uint8_t observed_ids[4], uint16_t feature[5]) {
    float edges_rad[6];
    uint8_t edge_count = 0;
    for (uint8_t first_pattern_index = 0; first_pattern_index < 4; ++first_pattern_index) {
        for (uint8_t second_pattern_index = first_pattern_index + 1; second_pattern_index < 4; ++second_pattern_index) {
            Vec3f first_vector = {
                observed_stars[observed_ids[first_pattern_index]].x,
                observed_stars[observed_ids[first_pattern_index]].y,
                observed_stars[observed_ids[first_pattern_index]].z,
            };
            Vec3f second_vector = {
                observed_stars[observed_ids[second_pattern_index]].x,
                observed_stars[observed_ids[second_pattern_index]].y,
                observed_stars[observed_ids[second_pattern_index]].z,
            };
            edges_rad[edge_count++] = angular_distance_rad(first_vector, second_vector);
        }
    }
    /* Sort ascending so edges_rad[5] is the longest edge. */
    for (uint8_t outer = 0; outer < 6; ++outer) {
        for (uint8_t inner = outer + 1; inner < 6; ++inner) {
            if (edges_rad[inner] < edges_rad[outer]) {
                float tmp = edges_rad[outer];
                edges_rad[outer] = edges_rad[inner];
                edges_rad[inner] = tmp;
            }
        }
    }
    if (edges_rad[5] == 0.0f) {
        return false;
    }
    /* TETRA feature: five shortest edges as ratios of the longest, quantized to uint16. */
    for (uint8_t feature_index = 0; feature_index < 5; ++feature_index) {
        float ratio = edges_rad[feature_index] / edges_rad[5];
        feature[feature_index] = (uint16_t)(ratio * 65535.0f + 0.5f);
    }
    return true;
}

/**
 * Keeps the better verified result without using any Pyramid information.
 */
static void keep_better(MatchResult *best, const MatchResult *candidate) {
    if (candidate->success && (!best->success || candidate->score > best->score)) {
        *best = *candidate;
    }
}

/**
 * Rejects impossible TETRA mappings before the expensive full-catalog verifier.
 */
static bool pattern_residual_is_small(
    const Mat3f *rotation,
    const uint16_t *mapped_hr_ids,
    const uint8_t *observed_ids,
    const ObservedStar *observed_stars
) {
    for (uint8_t pattern_index = 0; pattern_index < 4; ++pattern_index) {
        Vec3f catalog_vector_value;
        if (!catalog_vector(mapped_hr_ids[pattern_index], &catalog_vector_value)) {
            return false;
        }
        Vec3f predicted_observed_vector = mat3_mul_vec3(rotation, catalog_vector_value);
        Vec3f observed_vector = {
            observed_stars[observed_ids[pattern_index]].x,
            observed_stars[observed_ids[pattern_index]].y,
            observed_stars[observed_ids[pattern_index]].z,
        };
        if (angular_distance_rad(predicted_observed_vector, observed_vector) > TETRA_PATTERN_MAX_RESIDUAL_RAD) {
            return false;
        }
    }
    return true;
}

/**
 * Identifies a field using TETRA feature lookup and catalog-only verification.
 */
bool identify_tetra(const ObservedStar *observed_stars, uint8_t observed_star_count, MatchResult *result) {
    memset(result, 0, sizeof(*result));
    result->focal_scale = 1.0f;
    clock_t identify_start = clock();
    uint32_t verify_us_accum = 0;
    uint8_t query_star_count = observed_star_count < TETRA_MAX_QUERY_STARS ? observed_star_count : TETRA_MAX_QUERY_STARS;
    if (query_star_count < 4) {
        return false;
    }

    for (uint8_t first_observed_index = 0; first_observed_index < query_star_count - 3; ++first_observed_index) {
        for (uint8_t second_observed_index = first_observed_index + 1; second_observed_index < query_star_count - 2; ++second_observed_index) {
            for (uint8_t third_observed_index = second_observed_index + 1; third_observed_index < query_star_count - 1; ++third_observed_index) {
                for (uint8_t fourth_observed_index = third_observed_index + 1; fourth_observed_index < query_star_count; ++fourth_observed_index) {
                    uint8_t observed_ids[4] = {
                        first_observed_index,
                        second_observed_index,
                        third_observed_index,
                        fourth_observed_index,
                    };
                    uint16_t feature[5];
                    if (!make_tetra_feature(observed_stars, observed_ids, feature)) {
                        continue;
                    }

                    TetraCandidate candidates[TETRA_TOP_CANDIDATES];
                    uint8_t candidate_count = tetra_kd_search_topk(feature, candidates);
                    for (uint8_t candidate_index = 0; candidate_index < candidate_count; ++candidate_index) {
                        const TetraKdNode *node = &tetra_kd_nodes[candidates[candidate_index].node_id];
                        for (uint8_t permutation_index = 0; permutation_index < 24; ++permutation_index) {
                            uint16_t mapped_hr_ids[4] = {
                                node->hr[TETRA_PERMUTATIONS[permutation_index][0]],
                                node->hr[TETRA_PERMUTATIONS[permutation_index][1]],
                                node->hr[TETRA_PERMUTATIONS[permutation_index][2]],
                                node->hr[TETRA_PERMUTATIONS[permutation_index][3]],
                            };
                            Mat3f rotation;
                            MatchResult verified;
                            /* TETRA sorted-edge features do not preserve star order, so every HR permutation is tested. */
                            if (solve_attitude_triad(mapped_hr_ids, observed_ids, 4, observed_stars, &rotation) &&
                                pattern_residual_is_small(&rotation, mapped_hr_ids, observed_ids, observed_stars)) {
                                clock_t verify_start = clock();
                                /* Patterns are built from the brightest query stars, but verification
                                   counts inliers across ALL observed stars: real images carry more
                                   true catalog stars than fit in the matching pool, and a wider inlier
                                   count both rescues correct solves and outscores false positives. */
                                bool verified_ok = verify_attitude(&rotation, observed_stars, observed_star_count, &verified);
                                verify_us_accum += elapsed_us(verify_start, clock());
                                if (verified_ok) {
                                    keep_better(result, &verified);
                                    /* High-confidence match: no need to search further. */
                                    if (result->score >= TETRA_EARLY_EXIT_SCORE) {
                                        goto tetra_done;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
tetra_done:
    {
        uint32_t total_us = elapsed_us(identify_start, clock());
        result->verify_us = verify_us_accum;
        result->db_us = total_us > verify_us_accum ? total_us - verify_us_accum : 0;
    }
    return result->success;
}

/**
 * Rescales one observed unit vector as if the camera focal length changed by `ratio`.
 *
 * An observed vector v came from pixel offsets (dx, dy) as (dx/f, dy/f, 1) normalized,
 * so its tangent-plane offset is (v.x/v.z, v.y/v.z) = (dx/f, dy/f). Changing focal to
 * f' = f*ratio scales that offset by 1/ratio while the pixel offsets stay fixed. z>0
 * always (boresight forward), so the division is safe.
 */
static ObservedStar rescale_observed(ObservedStar star, float ratio) {
    float denom = star.z * ratio;
    Vec3f raw = {star.x / denom, star.y / denom, 1.0f};
    Vec3f unit = normalize3(raw);
    return (ObservedStar){unit.x, unit.y, unit.z, star.brightness};
}

/**
 * Computes the six sorted pairwise angular distances among four direction vectors.
 */
static void sorted_edges_rad(const Vec3f vectors[4], float edges_out[6]) {
    uint8_t edge_count = 0;
    for (uint8_t first = 0; first < 4; ++first) {
        for (uint8_t second = first + 1; second < 4; ++second) {
            edges_out[edge_count++] = angular_distance_rad(vectors[first], vectors[second]);
        }
    }
    for (uint8_t outer = 0; outer < 6; ++outer) {
        for (uint8_t inner = outer + 1; inner < 6; ++inner) {
            if (edges_out[inner] < edges_out[outer]) {
                float tmp = edges_out[outer];
                edges_out[outer] = edges_out[inner];
                edges_out[inner] = tmp;
            }
        }
    }
}

/**
 * Estimates f_true / f_seed from a candidate tetrad's observed vs. true catalog angles.
 *
 * Sorted edges are permutation-independent, so the focal correction needs no knowledge
 * of the star-to-HR correspondence. Because pixel->angle is gnomonic (nonlinear over the
 * field), one correction is not exact from an arbitrary seed, so the ratio is iterated to
 * convergence. Returns false if any HR has no catalog vector, an edge is degenerate, or
 * the recovered ratio leaves the sane band.
 */
static bool estimate_focal_ratio(
    const ObservedStar *observed_stars,
    const uint8_t observed_ids[4],
    const uint16_t hr_ids[4],
    float *out_ratio
) {
    Vec3f catalog_vectors[4];
    for (uint8_t index = 0; index < 4; ++index) {
        if (!catalog_vector(hr_ids[index], &catalog_vectors[index])) {
            return false;
        }
    }
    float true_edges[6];
    sorted_edges_rad(catalog_vectors, true_edges);
    if (true_edges[0] <= 0.0f) {
        return false;
    }

    float ratio = 1.0f;
    for (uint8_t iteration = 0; iteration < CALIB_MAX_ITERS; ++iteration) {
        Vec3f scaled[4];
        for (uint8_t index = 0; index < 4; ++index) {
            ObservedStar rescaled = rescale_observed(observed_stars[observed_ids[index]], ratio);
            scaled[index] = (Vec3f){rescaled.x, rescaled.y, rescaled.z};
        }
        float observed_edges[6];
        sorted_edges_rad(scaled, observed_edges);

        float ratio_sum = 0.0f;
        for (uint8_t edge = 0; edge < 6; ++edge) {
            ratio_sum += observed_edges[edge] / true_edges[edge];
        }
        float correction = ratio_sum / 6.0f;
        ratio *= correction;
        if (ratio < CALIB_MIN_RATIO || ratio > CALIB_MAX_RATIO || !isfinite(ratio)) {
            return false;
        }
        if (fabsf(correction - 1.0f) < CALIB_CONVERGE_TOL) {
            break;
        }
    }
    *out_ratio = ratio;
    return true;
}

/**
 * Identifies a field whose seed FOV may be far from the true FOV (see header).
 */
bool identify_tetra_calibrate(const ObservedStar *observed_stars, uint8_t observed_star_count, MatchResult *result) {
    memset(result, 0, sizeof(*result));
    result->focal_scale = 1.0f;
    uint8_t query_star_count = observed_star_count < TETRA_MAX_QUERY_STARS ? observed_star_count : TETRA_MAX_QUERY_STARS;
    if (query_star_count < 4 || observed_star_count > MAX_OBS_STARS) {
        return false;
    }
    clock_t calib_start = clock();
    MatchResult best;
    bool found_best = false;
    memset(&best, 0, sizeof(best));

    /* ponytail: O(C(query,4) * candidates) full re-identifies; bootstrap only runs until FOV locks.
       Keep the best verified candidate instead of the first one, because false FOVs can pass a
       weak early tetrad before the real FOV candidate is reached. */
    for (uint8_t first = 0; first < query_star_count - 3; ++first) {
        for (uint8_t second = first + 1; second < query_star_count - 2; ++second) {
            for (uint8_t third = second + 1; third < query_star_count - 1; ++third) {
                for (uint8_t fourth = third + 1; fourth < query_star_count; ++fourth) {
                    uint8_t observed_ids[4] = {first, second, third, fourth};
                    uint16_t feature[5];
                    if (!make_tetra_feature(observed_stars, observed_ids, feature)) {
                        continue;
                    }
                    TetraCandidate candidates[TETRA_TOP_CANDIDATES];
                    uint8_t candidate_count = tetra_kd_search_topk(feature, candidates);
                    for (uint8_t candidate_index = 0; candidate_index < candidate_count; ++candidate_index) {
                        if (elapsed_us(calib_start, clock()) > CALIB_BUDGET_US) {
                            if (found_best) {
                                *result = best;
                                return true;
                            }
                            return false;  // budget exhausted: give up on this frame, caller retries the next
                        }
                        const TetraKdNode *node = &tetra_kd_nodes[candidates[candidate_index].node_id];
                        float ratio;
                        if (!estimate_focal_ratio(observed_stars, observed_ids, node->hr, &ratio)) {
                            continue;
                        }
                        ObservedStar rescaled[MAX_OBS_STARS];
                        for (uint8_t star_index = 0; star_index < observed_star_count; ++star_index) {
                            rescaled[star_index] = rescale_observed(observed_stars[star_index], ratio);
                        }
                        MatchResult attempt;
                        if (identify_tetra(rescaled, observed_star_count, &attempt) && attempt.success) {
                            attempt.focal_scale = ratio;
                            if (!found_best || attempt.score > best.score) {
                                best = attempt;
                                found_best = true;
                            }
                        }
                    }
                }
            }
        }
    }
    if (!found_best) {
        return false;
    }
    *result = best;
    return true;
}
