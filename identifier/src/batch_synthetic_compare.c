#include "catalog_db.h"
#include "clock_utils.h"
#include "compare.h"
#include "pyramid_db.h"
#include "star_math.h"
#include "tetra_db.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/**
 * Returns the in-memory footprint of the TETRA KD-tree database in bytes.
 */
static uint64_t tetra_db_bytes(void) {
    return (uint64_t)tetra_kd_node_count * sizeof(TetraKdNode);
}

/**
 * Returns the in-memory footprint of the Pyramid pair/neighbor database in bytes.
 */
static uint64_t pyramid_db_bytes(void) {
    return (uint64_t)pyramid_pair_count * sizeof(PairRow)
        + (uint64_t)pyramid_neighbor_count * sizeof(PairNeighbor)
        + (uint64_t)pyramid_neighbor_start_count * sizeof(uint32_t);
}

/**
 * Stores one visible catalog star while building a synthetic field.
 */
typedef struct {
    uint16_t hr_id;
    int16_t magnitude_q100;
    Vec3f vector;
} SyntheticVisibleStar;

/**
 * Writes one semicolon-separated HR list into a CSV field.
 */
static void write_hr_list(FILE *csv_file, const uint16_t *hr_ids, uint8_t hr_count) {
    for (uint8_t hr_index = 0; hr_index < hr_count; ++hr_index) {
        if (hr_index > 0) {
            fputc(';', csv_file);
        }
        fprintf(csv_file, "%u", hr_ids[hr_index]);
    }
}

/**
 * Converts degrees to radians.
 */
static float deg_to_rad(float degrees) {
    return degrees * 3.14159265358979323846f / 180.0f;
}

/**
 * Inserts a visible star while keeping the brightest stars first.
 */
static void keep_brightest(
    SyntheticVisibleStar *visible_stars,
    uint8_t *visible_star_count,
    uint8_t max_visible_stars,
    SyntheticVisibleStar candidate_star
) {
    uint8_t insert_position = *visible_star_count;
    if (insert_position < max_visible_stars) {
        ++(*visible_star_count);
    } else if (candidate_star.magnitude_q100 >= visible_stars[insert_position - 1].magnitude_q100) {
        return;
    } else {
        insert_position = (uint8_t)(max_visible_stars - 1);
    }
    while (insert_position > 0 && visible_stars[insert_position - 1].magnitude_q100 > candidate_star.magnitude_q100) {
        visible_stars[insert_position] = visible_stars[insert_position - 1];
        --insert_position;
    }
    visible_stars[insert_position] = candidate_star;
}

/**
 * Builds one synthetic observed field centered on a catalog star.
 */
static uint8_t build_synthetic_field(
    uint16_t center_catalog_index,
    float fov_degrees,
    uint8_t max_query_stars,
    ObservedStar *observed_stars,
    uint16_t *visible_hr_ids,
    uint8_t *visible_hr_count
) {
    Vec3f center_vector = q15_to_vec3f(
        catalog_stars[center_catalog_index].x,
        catalog_stars[center_catalog_index].y,
        catalog_stars[center_catalog_index].z
    );
    float max_radius_radians = deg_to_rad(fov_degrees * 0.5f);
    SyntheticVisibleStar visible_stars[MAX_OBS_STARS];
    uint8_t query_star_count = 0;
    *visible_hr_count = 0;

    for (uint16_t catalog_index = 0; catalog_index < catalog_star_count; ++catalog_index) {
        Vec3f catalog_vector_value = q15_to_vec3f(
            catalog_stars[catalog_index].x,
            catalog_stars[catalog_index].y,
            catalog_stars[catalog_index].z
        );
        if (angular_distance_rad(center_vector, catalog_vector_value) > max_radius_radians) {
            continue;
        }
        keep_brightest(
            visible_stars,
            &query_star_count,
            max_query_stars,
            (SyntheticVisibleStar){
                catalog_stars[catalog_index].hr,
                catalog_stars[catalog_index].mag_q100,
                catalog_vector_value,
            }
        );
    }

    uint8_t observed_star_count = query_star_count < max_query_stars ? query_star_count : max_query_stars;
    *visible_hr_count = observed_star_count;
    for (uint8_t observed_index = 0; observed_index < observed_star_count; ++observed_index) {
        visible_hr_ids[observed_index] = visible_stars[observed_index].hr_id;
        /* Brightness inversely proportional to magnitude: brighter stars get higher values.
           Range ~1..851 so the 5% margin filter in verify_attitude fires for stars that
           differ by more than ~0.5 magnitudes, enabling false-positive rejection. */
        int32_t brightness_value = 651 - (int32_t)visible_stars[observed_index].magnitude_q100;
        if (brightness_value < 1) brightness_value = 1;
        observed_stars[observed_index] = (ObservedStar){
            visible_stars[observed_index].vector.x,
            visible_stars[observed_index].vector.y,
            visible_stars[observed_index].vector.z,
            (uint32_t)brightness_value,
        };
    }
    return observed_star_count;
}

/**
 * Returns true when one HR ID is in the expected visible field.
 */
static bool hr_is_visible(uint16_t hr_id, const uint16_t *visible_hr_ids, uint8_t visible_hr_count) {
    for (uint8_t visible_index = 0; visible_index < visible_hr_count; ++visible_index) {
        if (visible_hr_ids[visible_index] == hr_id) {
            return true;
        }
    }
    return false;
}

/**
 * Scores a match result against the synthetic field's known visible HR IDs.
 */
static bool result_is_correct(const MatchResult *match_result, const uint16_t *visible_hr_ids, uint8_t visible_hr_count) {
    if (!match_result->success || match_result->count == 0) {
        return false;
    }
    uint8_t visible_hits = 0;
    for (uint8_t match_index = 0; match_index < match_result->count; ++match_index) {
        if (hr_is_visible(match_result->hr_ids[match_index], visible_hr_ids, visible_hr_count)) {
            ++visible_hits;
        }
    }
    return visible_hits >= 4 && visible_hits * 10u >= match_result->count * 7u;
}

/**
 * Runs a repeatable synthetic batch and prints accuracy/runtime for both algorithms.
 */
int main(int argc, char **argv) {
    if (argc != 4 && argc != 5) {
        fprintf(stderr, "Usage: batch_synthetic_compare <samples> <fov_deg> <max_query_stars> [output.csv]\n");
        return 2;
    }

    int requested_samples = atoi(argv[1]);
    float fov_degrees = (float)atof(argv[2]);
    uint8_t max_query_stars = (uint8_t)atoi(argv[3]);
    if (requested_samples <= 0 || max_query_stars > MAX_OBS_STARS || max_query_stars < 4) {
        fprintf(stderr, "Invalid arguments.\n");
        return 2;
    }

    FILE *csv_file = NULL;
    if (argc == 5) {
        csv_file = fopen(argv[4], "w");
        if (csv_file == NULL) {
            fprintf(stderr, "Could not write CSV: %s\n", argv[4]);
            return 2;
        }
        fprintf(csv_file, "sample_index,valid,observed_stars,expected_hrs,tetra_success,tetra_correct,tetra_time_us,tetra_hrs,pyramid_success,pyramid_correct,pyramid_time_us,pyramid_hrs\n");
    }

    /* benchmark_latest.csv always receives one row per test image with per-step timing. */
    FILE *benchmark_file = fopen("outputs/benchmark_latest.csv", "w");
    if (benchmark_file != NULL) {
        fprintf(benchmark_file,
            "sample_index,observed_stars,"
            "tetra_correct,tetra_camera_us,tetra_db_us,tetra_verify_us,tetra_total_us,"
            "pyramid_correct,pyramid_camera_us,pyramid_db_us,pyramid_verify_us,pyramid_total_us\n");
    }

    int valid_fields = 0;
    int tetra_correct = 0;
    int pyramid_correct = 0;
    uint64_t tetra_total_us = 0;
    uint64_t pyramid_total_us = 0;
    /* Synthetic fields build observed vectors directly from the catalog, so there is
       no pixel-to-vector camera step here; camera_us is reported as 0. */
    uint64_t tetra_db_total_us = 0;
    uint64_t tetra_verify_total_us = 0;
    uint64_t pyramid_db_total_us = 0;
    uint64_t pyramid_verify_total_us = 0;
    clock_t batch_start = clock();

    for (int sample_index = 0; sample_index < requested_samples; ++sample_index) {
        uint16_t center_catalog_index = (uint16_t)((sample_index * 97u + 211u) % catalog_star_count);
        ObservedStar observed_stars[MAX_OBS_STARS];
        uint16_t visible_hr_ids[MAX_MATCHES];
        uint8_t visible_hr_count = 0;
        uint8_t observed_star_count = build_synthetic_field(
            center_catalog_index,
            fov_degrees,
            max_query_stars,
            observed_stars,
            visible_hr_ids,
            &visible_hr_count
        );
        if (observed_star_count < 4) {
            continue;
        }

        CompareResult comparison_result = compare_tetra_pyramid(observed_stars, observed_star_count);
        tetra_total_us += comparison_result.tetra_us;
        pyramid_total_us += comparison_result.pyramid_us;
        tetra_db_total_us += comparison_result.tetra.db_us;
        tetra_verify_total_us += comparison_result.tetra.verify_us;
        pyramid_db_total_us += comparison_result.pyramid.db_us;
        pyramid_verify_total_us += comparison_result.pyramid.verify_us;
        bool tetra_result_correct = result_is_correct(&comparison_result.tetra, visible_hr_ids, visible_hr_count);
        bool pyramid_result_correct = result_is_correct(&comparison_result.pyramid, visible_hr_ids, visible_hr_count);
        tetra_correct += tetra_result_correct ? 1 : 0;
        pyramid_correct += pyramid_result_correct ? 1 : 0;
        ++valid_fields;

        if (benchmark_file != NULL) {
            fprintf(benchmark_file,
                "%d,%u,%s,0,%lu,%lu,%lu,%s,0,%lu,%lu,%lu\n",
                sample_index,
                observed_star_count,
                tetra_result_correct ? "true" : "false",
                (unsigned long)comparison_result.tetra.db_us,
                (unsigned long)comparison_result.tetra.verify_us,
                (unsigned long)comparison_result.tetra_us,
                pyramid_result_correct ? "true" : "false",
                (unsigned long)comparison_result.pyramid.db_us,
                (unsigned long)comparison_result.pyramid.verify_us,
                (unsigned long)comparison_result.pyramid_us
            );
        }

        if (csv_file != NULL) {
            fprintf(
                csv_file,
                "%d,true,%u,\"",
                sample_index,
                observed_star_count
            );
            write_hr_list(csv_file, visible_hr_ids, visible_hr_count);
            fprintf(
                csv_file,
                "\",%s,%s,%lu,\"",
                comparison_result.tetra.success ? "true" : "false",
                tetra_result_correct ? "true" : "false",
                (unsigned long)comparison_result.tetra_us
            );
            write_hr_list(csv_file, comparison_result.tetra.hr_ids, comparison_result.tetra.count);
            fprintf(
                csv_file,
                "\",%s,%s,%lu,\"",
                comparison_result.pyramid.success ? "true" : "false",
                pyramid_result_correct ? "true" : "false",
                (unsigned long)comparison_result.pyramid_us
            );
            write_hr_list(csv_file, comparison_result.pyramid.hr_ids, comparison_result.pyramid.count);
            fprintf(csv_file, "\"\n");
        }

        if ((sample_index + 1) == requested_samples || (sample_index + 1) % 10 == 0) {
            clock_t now = clock();
            uint32_t elapsed_us_value = elapsed_us(batch_start, now);
            float completed_fraction = (float)(sample_index + 1) / (float)requested_samples;
            float eta_seconds = completed_fraction > 0.0f
                ? ((float)elapsed_us_value / 1000000.0f) * (1.0f - completed_fraction) / completed_fraction
                : 0.0f;
            printf(
                "progress %d/%d valid=%d elapsed=%.2fs eta=%.2fs\n",
                sample_index + 1,
                requested_samples,
                valid_fields,
                (float)elapsed_us_value / 1000000.0f,
                eta_seconds
            );
            fflush(stdout);
        }
    }

    printf("valid_fields=%d\n", valid_fields);
    printf(
        "TETRA accuracy_pct=%.2f avg_time_us=%llu\n",
        valid_fields > 0 ? (double)tetra_correct * 100.0 / (double)valid_fields : 0.0,
        valid_fields > 0 ? (unsigned long long)(tetra_total_us / (uint64_t)valid_fields) : 0ull
    );
    printf(
        "Pyramid accuracy_pct=%.2f avg_time_us=%llu\n",
        valid_fields > 0 ? (double)pyramid_correct * 100.0 / (double)valid_fields : 0.0,
        valid_fields > 0 ? (unsigned long long)(pyramid_total_us / (uint64_t)valid_fields) : 0ull
    );
    if (csv_file != NULL) {
        fclose(csv_file);
        printf("wrote_csv=%s\n", argv[4]);
    }

    /* Human-readable benchmark summary table. */
    double tetra_accuracy = valid_fields > 0 ? (double)tetra_correct * 100.0 / (double)valid_fields : 0.0;
    double pyramid_accuracy = valid_fields > 0 ? (double)pyramid_correct * 100.0 / (double)valid_fields : 0.0;
    double tetra_mean_ms = valid_fields > 0 ? (double)tetra_total_us / (double)valid_fields / 1000.0 : 0.0;
    double pyramid_mean_ms = valid_fields > 0 ? (double)pyramid_total_us / (double)valid_fields / 1000.0 : 0.0;
    double tetra_db_mb = (double)tetra_db_bytes() / (1024.0 * 1024.0);
    double pyramid_db_mb = (double)pyramid_db_bytes() / (1024.0 * 1024.0);
    double tetra_db_mean_ms = valid_fields > 0 ? (double)tetra_db_total_us / (double)valid_fields / 1000.0 : 0.0;
    double tetra_verify_mean_ms = valid_fields > 0 ? (double)tetra_verify_total_us / (double)valid_fields / 1000.0 : 0.0;
    double pyramid_db_mean_ms = valid_fields > 0 ? (double)pyramid_db_total_us / (double)valid_fields / 1000.0 : 0.0;
    double pyramid_verify_mean_ms = valid_fields > 0 ? (double)pyramid_verify_total_us / (double)valid_fields / 1000.0 : 0.0;

    printf("\n");
    printf("%-10s %9s %9s %9s %9s %9s\n", "Algorithm", "Accuracy%", "Mean_ms", "DB_ms", "Verify_ms", "DB_MB");
    printf("%-10s %9.2f %9.2f %9.2f %9.2f %9.2f\n",
        "TETRA", tetra_accuracy, tetra_mean_ms, tetra_db_mean_ms, tetra_verify_mean_ms, tetra_db_mb);
    printf("%-10s %9.2f %9.2f %9.2f %9.2f %9.2f\n",
        "Pyramid", pyramid_accuracy, pyramid_mean_ms, pyramid_db_mean_ms, pyramid_verify_mean_ms, pyramid_db_mb);

    if (benchmark_file != NULL) {
        fclose(benchmark_file);
        printf("wrote_benchmark=outputs/benchmark_latest.csv\n");
    }
    return 0;
}
