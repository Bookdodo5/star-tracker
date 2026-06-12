#include "camera_model.h"
#include "clock_utils.h"
#include "identify_pyramid.h"
#include "identify_tetra.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/**
 * Wraps an angle in degrees to [0, 360).
 */
static float wrap_degrees(float angle_degrees) {
    while (angle_degrees < 0.0f) {
        angle_degrees += 360.0f;
    }
    while (angle_degrees >= 360.0f) {
        angle_degrees -= 360.0f;
    }
    return angle_degrees;
}

/**
 * Prints the catalog-to-camera attitude as matrix and boresight RA/DEC/roll.
 */
static void print_attitude(const char *name, const MatchResult *match_result) {
    if (!match_result->success) {
        printf("%s attitude unavailable\n", name);
        return;
    }

    const float radians_to_degrees = 57.29577951308232f;
    float boresight_x = match_result->catalog_to_observed[2][0];
    float boresight_y = match_result->catalog_to_observed[2][1];
    float boresight_z = match_result->catalog_to_observed[2][2];
    float boresight_norm = sqrtf(boresight_x * boresight_x + boresight_y * boresight_y + boresight_z * boresight_z);
    if (boresight_norm <= 0.0f) {
        printf("%s attitude invalid\n", name);
        return;
    }
    boresight_x /= boresight_norm;
    boresight_y /= boresight_norm;
    boresight_z /= boresight_norm;

    float boresight_ra_degrees = wrap_degrees(atan2f(boresight_y, boresight_x) * radians_to_degrees);
    float boresight_dec_degrees = asinf(boresight_z) * radians_to_degrees;
    float boresight_ra_radians = boresight_ra_degrees / radians_to_degrees;
    float boresight_dec_radians = boresight_dec_degrees / radians_to_degrees;

    float east_x = -sinf(boresight_ra_radians);
    float east_y = cosf(boresight_ra_radians);
    float east_z = 0.0f;
    float north_x = -sinf(boresight_dec_radians) * cosf(boresight_ra_radians);
    float north_y = -sinf(boresight_dec_radians) * sinf(boresight_ra_radians);
    float north_z = cosf(boresight_dec_radians);

    /* Image-up is negative camera Y because pixel Y grows downward. */
    float image_up_x = -match_result->catalog_to_observed[1][0];
    float image_up_y = -match_result->catalog_to_observed[1][1];
    float image_up_z = -match_result->catalog_to_observed[1][2];
    float roll_degrees = atan2f(
        image_up_x * east_x + image_up_y * east_y + image_up_z * east_z,
        image_up_x * north_x + image_up_y * north_y + image_up_z * north_z
    ) * radians_to_degrees;

    printf(
        "%s attitude_ra_deg=%.6f attitude_dec_deg=%.6f attitude_roll_deg=%.6f\n",
        name,
        boresight_ra_degrees,
        boresight_dec_degrees,
        roll_degrees
    );
    printf(
        "%s rotation_catalog_to_camera=[[%.8f,%.8f,%.8f],[%.8f,%.8f,%.8f],[%.8f,%.8f,%.8f]]\n",
        name,
        match_result->catalog_to_observed[0][0],
        match_result->catalog_to_observed[0][1],
        match_result->catalog_to_observed[0][2],
        match_result->catalog_to_observed[1][0],
        match_result->catalog_to_observed[1][1],
        match_result->catalog_to_observed[1][2],
        match_result->catalog_to_observed[2][0],
        match_result->catalog_to_observed[2][1],
        match_result->catalog_to_observed[2][2]
    );
}

/**
 * Builds a simple pinhole camera model from image size and horizontal FOV.
 */
static CameraModel camera_from_horizontal_fov(int image_width, int image_height, float horizontal_fov_degrees) {
    float horizontal_fov_radians = horizontal_fov_degrees * 3.14159265358979323846f / 180.0f;
    float focal_length_pixels = ((float)image_width * 0.5f) / tanf(horizontal_fov_radians * 0.5f);
    return (CameraModel){
        focal_length_pixels,
        focal_length_pixels,
        ((float)image_width - 1.0f) * 0.5f,
        ((float)image_height - 1.0f) * 0.5f,
    };
}

/**
 * Reads Centroid/stars.csv rows into DetectedStar records.
 */
static uint8_t read_centroid_csv(const char *path, DetectedStar *detected_stars, uint8_t max_detected_stars) {
    FILE *file = fopen(path, "r");
    if (file == NULL) {
        fprintf(stderr, "Could not open centroid CSV: %s\n", path);
        return 0;
    }

    char line[256];
    uint8_t detected_star_count = 0;
    fgets(line, sizeof(line), file);
    while (detected_star_count < max_detected_stars && fgets(line, sizeof(line), file) != NULL) {
        unsigned int csv_index;
        unsigned int pixel_x;
        unsigned int pixel_y;
        unsigned long brightness;
        if (sscanf(line, "%u,%u,%u,%lu", &csv_index, &pixel_x, &pixel_y, &brightness) == 4) {
            detected_stars[detected_star_count++] = (DetectedStar){
                (uint16_t)pixel_x,
                (uint16_t)pixel_y,
                (uint32_t)brightness,
            };
        }
    }

    fclose(file);
    return detected_star_count;
}

/**
 * Prints one algorithm result in a compact, comparable format.
 */
static void print_match_result(const char *name, const MatchResult *match_result, uint32_t elapsed_us) {
    printf(
        "%s success=%s matches=%u mean_residual_arcsec=%u max_residual_arcsec=%u score=%ld time_us=%lu\n",
        name,
        match_result->success ? "true" : "false",
        match_result->count,
        match_result->mean_residual_arcsec,
        match_result->max_residual_arcsec,
        (long)match_result->score,
        (unsigned long)elapsed_us
    );

    printf("%s HR IDs:", name);
    for (uint8_t match_index = 0; match_index < match_result->count; ++match_index) {
        printf(" %u", match_result->hr_ids[match_index]);
    }
    printf("\n");

    print_attitude(name, match_result);

    printf("%s matches_csv: obs_id,hr_id,residual_arcsec\n", name);
    for (uint8_t match_index = 0; match_index < match_result->count; ++match_index) {
        printf(
            "%s match,%u,%u,%u\n",
            name,
            match_result->obs_ids[match_index],
            match_result->hr_ids[match_index],
            match_result->residual_arcsec[match_index]
        );
    }
}

/**
 * Runs independent TETRA and Pyramid identification from a Centroid CSV file.
 */
int main(int argc, char **argv) {
    if (argc != 5) {
        fprintf(stderr, "Usage: demo_centroid_compare <stars.csv> <image_width> <image_height> <horizontal_fov_deg>\n");
        return 2;
    }

    const char *centroid_csv_path = argv[1];
    int image_width = atoi(argv[2]);
    int image_height = atoi(argv[3]);
    float horizontal_fov_degrees = (float)atof(argv[4]);

    DetectedStar detected_stars[MAX_OBS_STARS];
    ObservedStar observed_stars[MAX_OBS_STARS];
    CameraModel camera_model = camera_from_horizontal_fov(image_width, image_height, horizontal_fov_degrees);
    uint8_t detected_star_count = read_centroid_csv(centroid_csv_path, detected_stars, MAX_OBS_STARS);
    uint8_t observed_star_count = convert_detected_stars(
        detected_stars,
        detected_star_count,
        &camera_model,
        observed_stars,
        MAX_OBS_STARS
    );

    printf("Detected stars: %u\n", detected_star_count);
    printf("Observed vectors: %u\n", observed_star_count);
    printf(
        "Camera fx=%.3f fy=%.3f cx=%.3f cy=%.3f\n",
        camera_model.fx,
        camera_model.fy,
        camera_model.cx,
        camera_model.cy
    );

    MatchResult tetra_result;
    MatchResult pyramid_result;

    printf("Running TETRA...\n");
    fflush(stdout);
    clock_t tetra_start = clock();
    identify_tetra(observed_stars, observed_star_count, &tetra_result);
    clock_t tetra_end = clock();
    print_match_result("TETRA", &tetra_result, elapsed_us(tetra_start, tetra_end));
    fflush(stdout);

    printf("Running Pyramid...\n");
    fflush(stdout);
    clock_t pyramid_start = clock();
    identify_pyramid(observed_stars, observed_star_count, &pyramid_result);
    clock_t pyramid_end = clock();
    print_match_result("Pyramid", &pyramid_result, elapsed_us(pyramid_start, pyramid_end));
    return 0;
}
