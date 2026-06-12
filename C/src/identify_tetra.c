#include "identify_tetra.h"
#include "attitude.h"
#include "catalog_db.h"
#include "star_math.h"
#include "tetra_db.h"
#include "verify.h"
#include <string.h>

#define TETRA_MAX_QUERY_STARS 8
#define TETRA_TOP_CANDIDATES 32
#define KD_STACK_MAX 96
#define KD_NODE_VISIT_LIMIT 32768
#define TETRA_PATTERN_MAX_RESIDUAL_RAD 0.010471976f

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
 * Computes one quantized angular edge between two observed stars.
 */
static uint16_t edge_code(const ObservedStar *first_observed_star, const ObservedStar *second_observed_star) {
    Vec3f first_vector = {first_observed_star->x, first_observed_star->y, first_observed_star->z};
    Vec3f second_vector = {second_observed_star->x, second_observed_star->y, second_observed_star->z};
    return angle_to_code(angular_distance_rad(first_vector, second_vector), 3.14159265358979323846f);
}

/**
 * Builds the normalized five-edge feature used for TETRA nearest-neighbor lookup.
 */
static bool make_tetra_feature(const ObservedStar *observed_stars, const uint8_t observed_ids[4], uint16_t feature[5]) {
    uint16_t edges[6];
    uint8_t edge_count = 0;
    for (uint8_t first_pattern_index = 0; first_pattern_index < 4; ++first_pattern_index) {
        for (uint8_t second_pattern_index = first_pattern_index + 1; second_pattern_index < 4; ++second_pattern_index) {
            edges[edge_count++] = edge_code(
                &observed_stars[observed_ids[first_pattern_index]],
                &observed_stars[observed_ids[second_pattern_index]]
            );
        }
    }
    sort_u16(edges, 6);
    if (edges[5] == 0) {
        return false;
    }
    /* TETRA stores the five shortest edges normalized by the longest edge. */
    for (uint8_t feature_index = 0; feature_index < 5; ++feature_index) {
        feature[feature_index] = (uint16_t)(((uint32_t)edges[feature_index] * 65535u) / edges[5]);
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
                                pattern_residual_is_small(&rotation, mapped_hr_ids, observed_ids, observed_stars) &&
                                verify_attitude(&rotation, observed_stars, query_star_count, &verified)) {
                                keep_better(result, &verified);
                            }
                        }
                    }
                }
            }
        }
    }
    return result->success;
}
