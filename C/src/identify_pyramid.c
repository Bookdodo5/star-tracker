#include "identify_pyramid.h"
#include "attitude.h"
#include "catalog_db.h"
#include "pyramid_db.h"
#include "star_math.h"
#include "verify.h"
#include <string.h>

#define PYRAMID_MAX_QUERY_STARS 10
#define PYRAMID_MAX_SEED_CANDIDATES 45
#define PYRAMID_MAX_GROW_CANDIDATES 45
#define PYRAMID_MAX_INTERSECTION_CANDIDATES 180
#define PYRAMID_MAX_BRANCHES 1200
#define PYRAMID_SEED_TOL_CODE 328u
#define PYRAMID_GROW_TOL_CODE 590u
#define PYRAMID_PATTERN_MAX_RESIDUAL_RAD 0.010471976f

typedef struct {
    uint16_t hr_a;
    uint16_t hr_b;
    uint16_t error;
    uint16_t vote_error;
    uint8_t votes;
} PairCandidate;

typedef struct {
    MatchResult best;
    uint16_t branches;
} PyramidState;

static uint8_t next_catalog_candidates(
    const uint16_t angle_codes[MAX_OBS_STARS][MAX_OBS_STARS],
    uint8_t next_obs,
    const uint8_t *obs_ids,
    const uint16_t *hr_ids,
    uint8_t size,
    uint16_t *catalog_candidate_hr_ids
);

/**
 * Computes absolute difference between two quantized separation codes.
 */
static uint16_t abs_diff_u16(uint16_t left_value, uint16_t right_value) {
    return left_value > right_value ? (uint16_t)(left_value - right_value) : (uint16_t)(right_value - left_value);
}

/**
 * Finds the first pair row with sep_code >= code.
 */
static uint32_t lower_bound_sep(uint16_t target_sep_code) {
    uint32_t search_start = 0;
    uint32_t search_end = pyramid_pair_count;
    while (search_start < search_end) {
        uint32_t middle_index = search_start + (search_end - search_start) / 2;
        if (pyramid_pairs_by_sep[middle_index].sep_code < target_sep_code) {
            search_start = middle_index + 1;
        } else {
            search_end = middle_index;
        }
    }
    return search_start;
}

/**
 * Finds the first pair row with sep_code > code.
 */
static uint32_t upper_bound_sep(uint16_t target_sep_code) {
    uint32_t search_start = 0;
    uint32_t search_end = pyramid_pair_count;
    while (search_start < search_end) {
        uint32_t middle_index = search_start + (search_end - search_start) / 2;
        if (pyramid_pairs_by_sep[middle_index].sep_code <= target_sep_code) {
            search_start = middle_index + 1;
        } else {
            search_end = middle_index;
        }
    }
    return search_start;
}

/**
 * Finds the first neighbor row in one HR adjacency range with sep_code >= code.
 */
static uint32_t lower_bound_neighbor_sep(uint32_t search_start, uint32_t search_end, uint16_t target_sep_code) {
    while (search_start < search_end) {
        uint32_t middle_index = search_start + (search_end - search_start) / 2;
        if (pyramid_neighbors_by_hr[middle_index].sep_code < target_sep_code) {
            search_start = middle_index + 1;
        } else {
            search_end = middle_index;
        }
    }
    return search_start;
}

/**
 * Finds the first neighbor row in one HR adjacency range with sep_code > code.
 */
static uint32_t upper_bound_neighbor_sep(uint32_t search_start, uint32_t search_end, uint16_t target_sep_code) {
    while (search_start < search_end) {
        uint32_t middle_index = search_start + (search_end - search_start) / 2;
        if (pyramid_neighbors_by_hr[middle_index].sep_code <= target_sep_code) {
            search_start = middle_index + 1;
        } else {
            search_end = middle_index;
        }
    }
    return search_start;
}

/**
 * Returns nearest catalog pair candidates for one observed separation.
 */
static void insert_pair_candidate(
    PairCandidate *nearest_pair_candidates,
    uint8_t *nearest_pair_count,
    PairCandidate candidate
) {
    uint32_t candidate_score = (uint32_t)candidate.error + (uint32_t)candidate.vote_error - (uint32_t)candidate.votes * 256u;
    uint8_t insert_position = *nearest_pair_count;
    if (insert_position < PYRAMID_MAX_SEED_CANDIDATES) {
        ++(*nearest_pair_count);
    } else {
        uint32_t worst_score =
            (uint32_t)nearest_pair_candidates[insert_position - 1].error +
            (uint32_t)nearest_pair_candidates[insert_position - 1].vote_error -
            (uint32_t)nearest_pair_candidates[insert_position - 1].votes * 256u;
        if (candidate_score >= worst_score) {
            return;
        }
        insert_position = PYRAMID_MAX_SEED_CANDIDATES - 1;
    }
    while (insert_position > 0) {
        uint32_t previous_score =
            (uint32_t)nearest_pair_candidates[insert_position - 1].error +
            (uint32_t)nearest_pair_candidates[insert_position - 1].vote_error -
            (uint32_t)nearest_pair_candidates[insert_position - 1].votes * 256u;
        if (previous_score <= candidate_score) {
            break;
        }
        nearest_pair_candidates[insert_position] = nearest_pair_candidates[insert_position - 1];
        --insert_position;
    }
    nearest_pair_candidates[insert_position] = candidate;
}

/**
 * Scores whether one seed candidate can geometrically connect to other observed stars.
 */
static PairCandidate score_seed_candidate(
    uint16_t seed_hr_a,
    uint16_t seed_hr_b,
    uint16_t separation_error,
    const uint16_t angle_codes[MAX_OBS_STARS][MAX_OBS_STARS],
    uint8_t first_observed_index,
    uint8_t second_observed_index,
    uint8_t observed_star_count
) {
    PairCandidate candidate = {seed_hr_a, seed_hr_b, separation_error, 0, 0};
    for (uint8_t other_observed_index = 0; other_observed_index < observed_star_count; ++other_observed_index) {
        if (other_observed_index == first_observed_index || other_observed_index == second_observed_index) {
            continue;
        }
        uint16_t candidate_hrs[PYRAMID_MAX_GROW_CANDIDATES];
        uint8_t obs_ids[2] = {first_observed_index, second_observed_index};
        uint16_t hr_ids[2] = {seed_hr_a, seed_hr_b};
        uint8_t count = next_catalog_candidates(angle_codes, other_observed_index, obs_ids, hr_ids, 2, candidate_hrs);
        if (count > 0) {
            ++candidate.votes;
        } else {
            candidate.vote_error = (uint16_t)(candidate.vote_error + PYRAMID_GROW_TOL_CODE);
        }
    }
    return candidate;
}

/**
 * Returns nearest catalog pair candidates for one observed separation.
 */
static uint8_t query_pairs(
    const uint16_t angle_codes[MAX_OBS_STARS][MAX_OBS_STARS],
    uint8_t first_observed_index,
    uint8_t second_observed_index,
    uint8_t observed_star_count,
    PairCandidate *nearest_pair_candidates
) {
    uint16_t observed_sep_code_value = angle_codes[first_observed_index][second_observed_index];
    uint16_t min_sep_code = observed_sep_code_value > PYRAMID_SEED_TOL_CODE
        ? (uint16_t)(observed_sep_code_value - PYRAMID_SEED_TOL_CODE)
        : 0;
    uint16_t max_sep_code = observed_sep_code_value < 65535u - PYRAMID_SEED_TOL_CODE
        ? (uint16_t)(observed_sep_code_value + PYRAMID_SEED_TOL_CODE)
        : 65535u;
    uint32_t first_candidate_index = lower_bound_sep(min_sep_code);
    uint32_t after_last_candidate_index = upper_bound_sep(max_sep_code);
    uint8_t nearest_pair_count = 0;

    /* The database is sorted by sep_code, so only the tolerance window is scanned. */
    for (uint32_t pair_index = first_candidate_index; pair_index < pyramid_pair_count && pair_index < after_last_candidate_index; ++pair_index) {
        uint16_t separation_error = abs_diff_u16(observed_sep_code_value, pyramid_pairs_by_sep[pair_index].sep_code);
        PairCandidate candidate = score_seed_candidate(
            pyramid_pairs_by_sep[pair_index].hr_a,
            pyramid_pairs_by_sep[pair_index].hr_b,
            separation_error,
            angle_codes,
            first_observed_index,
            second_observed_index,
            observed_star_count
        );
        insert_pair_candidate(nearest_pair_candidates, &nearest_pair_count, candidate);
    }
    return nearest_pair_count;
}

/**
 * Quantizes observed angular separation into the Pyramid pair-code scale.
 */
static uint16_t observed_sep_code(const ObservedStar *first_observed_star, const ObservedStar *second_observed_star) {
    Vec3f first_vector = {first_observed_star->x, first_observed_star->y, first_observed_star->z};
    Vec3f second_vector = {second_observed_star->x, second_observed_star->y, second_observed_star->z};
    return angle_to_code(angular_distance_rad(first_vector, second_vector), pyramid_max_sep_rad);
}

/**
 * Computes catalog angular separation for pattern-growth consistency checks.
 */
static uint16_t catalog_sep_code(uint16_t left_hr, uint16_t right_hr) {
    Vec3f left;
    Vec3f right;
    if (!catalog_vector(left_hr, &left) || !catalog_vector(right_hr, &right)) {
        return 65535u;
    }
    return angle_to_code(angular_distance_rad(left, right), pyramid_max_sep_rad);
}

/**
 * Rejects impossible Pyramid mappings before the expensive full-catalog verifier.
 */
static bool pattern_residual_is_small(
    const Mat3f *rotation,
    const uint16_t *hr_ids,
    const uint8_t *obs_ids,
    const ObservedStar *observed_stars
) {
    for (uint8_t pattern_index = 0; pattern_index < PYRAMID_SIZE; ++pattern_index) {
        Vec3f catalog_vector_value;
        if (!catalog_vector(hr_ids[pattern_index], &catalog_vector_value)) {
            return false;
        }
        Vec3f predicted_observed_vector = mat3_mul_vec3(rotation, catalog_vector_value);
        Vec3f observed_vector = {
            observed_stars[obs_ids[pattern_index]].x,
            observed_stars[obs_ids[pattern_index]].y,
            observed_stars[obs_ids[pattern_index]].z,
        };
        if (angular_distance_rad(predicted_observed_vector, observed_vector) > PYRAMID_PATTERN_MAX_RESIDUAL_RAD) {
            return false;
        }
    }
    return true;
}

/**
 * Finds catalog stars that are geometrically consistent with the partial Pyramid branch.
 */
static uint8_t next_catalog_candidates(
    const uint16_t angle_codes[MAX_OBS_STARS][MAX_OBS_STARS],
    uint8_t next_obs,
    const uint8_t *obs_ids,
    const uint16_t *hr_ids,
    uint8_t size,
    uint16_t *catalog_candidate_hr_ids
) {
    uint16_t candidate_pool[PYRAMID_MAX_INTERSECTION_CANDIDATES];
    uint32_t candidate_errors[PYRAMID_MAX_INTERSECTION_CANDIDATES];
    uint16_t first_observed_sep = angle_codes[next_obs][obs_ids[0]];
    uint16_t min_sep_code = first_observed_sep > PYRAMID_GROW_TOL_CODE
        ? (uint16_t)(first_observed_sep - PYRAMID_GROW_TOL_CODE)
        : 0;
    uint16_t max_sep_code = first_observed_sep < 65535u - PYRAMID_GROW_TOL_CODE
        ? (uint16_t)(first_observed_sep + PYRAMID_GROW_TOL_CODE)
        : 65535u;
    if (hr_ids[0] + 1u >= pyramid_neighbor_start_count) {
        return 0;
    }
    uint32_t neighbor_start = pyramid_neighbor_starts[hr_ids[0]];
    uint32_t neighbor_end = pyramid_neighbor_starts[hr_ids[0] + 1u];
    uint32_t first_pair_index = lower_bound_neighbor_sep(neighbor_start, neighbor_end, min_sep_code);
    uint32_t after_last_pair_index = upper_bound_neighbor_sep(neighbor_start, neighbor_end, max_sep_code);
    uint8_t pool_count = 0;

    /* Full HR adjacency gives grow candidates without scanning the entire catalog. */
    for (uint32_t pair_index = first_pair_index; pair_index < after_last_pair_index; ++pair_index) {
        uint16_t candidate_hr_id = pyramid_neighbors_by_hr[pair_index].hr_id;

        bool used = false;
        for (uint8_t branch_index = 0; branch_index < size; ++branch_index) {
            if (candidate_hr_id == hr_ids[branch_index]) {
                used = true;
                break;
            }
        }
        if (used) {
            continue;
        }

        uint32_t geometry_error = abs_diff_u16(pyramid_neighbors_by_hr[pair_index].sep_code, first_observed_sep);
        uint8_t insert_position = pool_count;
        if (insert_position < PYRAMID_MAX_INTERSECTION_CANDIDATES) {
            ++pool_count;
        } else if (geometry_error >= candidate_errors[insert_position - 1]) {
            continue;
        } else {
            insert_position = PYRAMID_MAX_INTERSECTION_CANDIDATES - 1;
        }
        while (insert_position > 0 && candidate_errors[insert_position - 1] > geometry_error) {
            candidate_errors[insert_position] = candidate_errors[insert_position - 1];
            candidate_pool[insert_position] = candidate_pool[insert_position - 1];
            --insert_position;
        }
        candidate_errors[insert_position] = geometry_error;
        candidate_pool[insert_position] = candidate_hr_id;
    }

    uint8_t candidate_count = 0;
    uint32_t errors[PYRAMID_MAX_GROW_CANDIDATES];
    for (uint8_t pool_index = 0; pool_index < pool_count; ++pool_index) {
        uint32_t geometry_error = candidate_errors[pool_index];
        for (uint8_t branch_index = 1; branch_index < size; ++branch_index) {
            uint16_t catalog_separation = catalog_sep_code(candidate_pool[pool_index], hr_ids[branch_index]);
            geometry_error += abs_diff_u16(catalog_separation, angle_codes[next_obs][obs_ids[branch_index]]);
        }
        if (geometry_error > (uint32_t)PYRAMID_GROW_TOL_CODE * size) {
            continue;
        }

        /* Keep only the best growth candidates to avoid branch explosion. */
        uint8_t insert_position = candidate_count;
        if (insert_position < PYRAMID_MAX_GROW_CANDIDATES) {
            ++candidate_count;
        } else if (geometry_error >= errors[insert_position - 1]) {
            continue;
        } else {
            insert_position = PYRAMID_MAX_GROW_CANDIDATES - 1;
        }
        while (insert_position > 0 && errors[insert_position - 1] > geometry_error) {
            errors[insert_position] = errors[insert_position - 1];
            catalog_candidate_hr_ids[insert_position] = catalog_candidate_hr_ids[insert_position - 1];
            --insert_position;
        }
        errors[insert_position] = geometry_error;
        catalog_candidate_hr_ids[insert_position] = candidate_pool[pool_index];
    }

    return candidate_count;
}

/**
 * Solves attitude for one complete Pyramid pattern and verifies it against the catalog.
 */
static void evaluate_pattern(
    const ObservedStar *observed_stars,
    uint8_t observed_star_count,
    const uint8_t *obs_ids,
    const uint16_t *hr_ids,
    PyramidState *state
) {
    for (uint8_t first_pattern_index = 0; first_pattern_index < PYRAMID_SIZE - 1; ++first_pattern_index) {
        for (uint8_t second_pattern_index = first_pattern_index + 1; second_pattern_index < PYRAMID_SIZE; ++second_pattern_index) {
            uint16_t pair_hr_ids[2] = {hr_ids[first_pattern_index], hr_ids[second_pattern_index]};
            uint8_t pair_obs_ids[2] = {obs_ids[first_pattern_index], obs_ids[second_pattern_index]};
            Mat3f rotation;
            MatchResult verified;
            /* Verification is shared and algorithm-neutral; it does not compare against TETRA. */
            if (solve_attitude_triad(pair_hr_ids, pair_obs_ids, 2, observed_stars, &rotation) &&
                pattern_residual_is_small(&rotation, hr_ids, obs_ids, observed_stars) &&
                verify_attitude(&rotation, observed_stars, observed_star_count, &verified) &&
                (!state->best.success || verified.score > state->best.score)) {
                state->best = verified;
            }
        }
    }
}

/**
 * Recursively grows a seed pair into a four-star Pyramid pattern.
 */
static void grow(
    const ObservedStar *observed_stars,
    uint8_t observed_star_count,
    const uint16_t angle_codes[MAX_OBS_STARS][MAX_OBS_STARS],
    uint8_t *obs_ids,
    uint16_t *hr_ids,
    uint8_t size,
    uint8_t next_obs,
    PyramidState *state
) {
    if (++state->branches > PYRAMID_MAX_BRANCHES) {
        return;
    }
    if (size == PYRAMID_SIZE) {
        evaluate_pattern(observed_stars, observed_star_count, obs_ids, hr_ids, state);
        return;
    }
    if (next_obs >= observed_star_count) {
        return;
    }

    /* Skip observed stars that already belong to this branch. */
    while (next_obs < observed_star_count) {
        bool used_obs = false;
        for (uint8_t branch_index = 0; branch_index < size; ++branch_index) {
            if (obs_ids[branch_index] == next_obs) {
                used_obs = true;
                break;
            }
        }
        if (!used_obs) {
            break;
        }
        ++next_obs;
    }
    if (next_obs >= observed_star_count) {
        return;
    }

    uint16_t candidates[PYRAMID_MAX_GROW_CANDIDATES];
    uint8_t count = next_catalog_candidates(angle_codes, next_obs, obs_ids, hr_ids, size, candidates);
    for (uint8_t candidate_index = 0; candidate_index < count; ++candidate_index) {
        obs_ids[size] = next_obs;
        hr_ids[size] = candidates[candidate_index];
        grow(
            observed_stars,
            observed_star_count,
            angle_codes,
            obs_ids,
            hr_ids,
            (uint8_t)(size + 1),
            (uint8_t)(next_obs + 1),
            state
        );
        if (state->branches > PYRAMID_MAX_BRANCHES) {
            return;
        }
    }
}

/**
 * Identifies a field using Pyramid pair lookup and catalog-only verification.
 */
bool identify_pyramid(const ObservedStar *observed_stars, uint8_t observed_star_count, MatchResult *result) {
    memset(result, 0, sizeof(*result));
    uint8_t query_star_count = observed_star_count < PYRAMID_MAX_QUERY_STARS ? observed_star_count : PYRAMID_MAX_QUERY_STARS;
    if (query_star_count < PYRAMID_SIZE) {
        return false;
    }

    uint16_t angle_codes[MAX_OBS_STARS][MAX_OBS_STARS] = {{0}};
    /* Precompute observed pair separations once because every branch reuses them. */
    for (uint8_t first_observed_index = 0; first_observed_index < query_star_count; ++first_observed_index) {
        for (uint8_t second_observed_index = first_observed_index + 1; second_observed_index < query_star_count; ++second_observed_index) {
            angle_codes[first_observed_index][second_observed_index] =
                observed_sep_code(&observed_stars[first_observed_index], &observed_stars[second_observed_index]);
            angle_codes[second_observed_index][first_observed_index] =
                angle_codes[first_observed_index][second_observed_index];
        }
    }

    PyramidState state;
    memset(&state, 0, sizeof(state));
    for (uint8_t first_seed_observed_index = 0; first_seed_observed_index < query_star_count - 1; ++first_seed_observed_index) {
        for (uint8_t second_seed_observed_index = first_seed_observed_index + 1; second_seed_observed_index < query_star_count; ++second_seed_observed_index) {
            PairCandidate candidates[PYRAMID_MAX_SEED_CANDIDATES];
            uint8_t seed_candidate_count = query_pairs(
                angle_codes,
                first_seed_observed_index,
                second_seed_observed_index,
                query_star_count,
                candidates
            );
            for (uint8_t seed_candidate_index = 0; seed_candidate_index < seed_candidate_count; ++seed_candidate_index) {
                uint8_t obs_ids[PYRAMID_SIZE] = {first_seed_observed_index, second_seed_observed_index, 0, 0};
                uint16_t hr_ids[PYRAMID_SIZE] = {
                    candidates[seed_candidate_index].hr_a,
                    candidates[seed_candidate_index].hr_b,
                    0,
                    0,
                };
                /* Pair orientation is ambiguous, so Pyramid tests both directions. */
                grow(observed_stars, query_star_count, angle_codes, obs_ids, hr_ids, 2, 0, &state);
                hr_ids[0] = candidates[seed_candidate_index].hr_b;
                hr_ids[1] = candidates[seed_candidate_index].hr_a;
                grow(observed_stars, query_star_count, angle_codes, obs_ids, hr_ids, 2, 0, &state);
                if (state.branches > PYRAMID_MAX_BRANCHES) {
                    *result = state.best;
                    return result->success;
                }
            }
        }
    }

    *result = state.best;
    return result->success;
}
