/**
 * In-process star identification: one frame buffer in, attitude out.
 *
 * This is the portable core for real-time use and hardware porting. Python (or a
 * sensor driver) hands it an RGB frame; it runs the exact centroid -> camera model
 * -> TETRA chain the file-based exes run, but with no process spawn and no disk I/O.
 * `identify_frame` is the single entry point a firmware loop would call.
 */
#include <cstdint>
#include <cmath>
#include <vector>

extern "C" {
#include "star_types.h"
#include "camera_model.h"
#include "identify_tetra.h"
}

/** Centroid detector, compiled from Centroid/centroid_extract.cpp (C linkage). */
extern "C" void extract_centroids(const uint8_t *rgb_in, uint8_t *dog_out, uint8_t *morph_out,
                                  // morph_passes appended at end; see centroid_extract.cpp
                                  uint16_t *star_x, uint16_t *star_y, uint64_t *star_brightness,
                                  int *star_count, int width, int height, int morph_passes);

/** Matches the centroid detector's static buffer ceiling (1920x1080). */
static const int LIVE_MAX_PIXELS = 1920 * 1080;
static const float RAD_TO_DEG = 57.29577951308232f;

/** Wraps degrees to [0, 360). */
static float wrap360(float angle) {
    while (angle < 0.0f) angle += 360.0f;
    while (angle >= 360.0f) angle -= 360.0f;
    return angle;
}

/** Builds a pinhole camera from image size and horizontal FOV (same as identify_from_centroids). */
static CameraModel camera_from_fov(int width, int height, float fov_deg) {
    float fov_rad = fov_deg * 3.14159265358979323846f / 180.0f;
    float focal = ((float)width * 0.5f) / tanf(fov_rad * 0.5f);
    return (CameraModel){focal, focal, ((float)width - 1.0f) * 0.5f, ((float)height - 1.0f) * 0.5f};
}

/** Converts a catalog->camera rotation to boresight RA/DEC and roll in degrees. */
static bool attitude_to_radecroll(const float rotation[3][3], double *ra, double *dec, double *roll) {
    float bx = rotation[2][0], by = rotation[2][1], bz = rotation[2][2];
    float norm = sqrtf(bx * bx + by * by + bz * bz);
    if (norm <= 0.0f) return false;
    bx /= norm; by /= norm; bz /= norm;

    float ra_deg = wrap360(atan2f(by, bx) * RAD_TO_DEG);
    float dec_deg = asinf(bz) * RAD_TO_DEG;
    float ra_rad = ra_deg / RAD_TO_DEG, dec_rad = dec_deg / RAD_TO_DEG;

    float ex = -sinf(ra_rad), ey = cosf(ra_rad), ez = 0.0f;
    float nx = -sinf(dec_rad) * cosf(ra_rad), ny = -sinf(dec_rad) * sinf(ra_rad), nz = cosf(dec_rad);
    float ux = rotation[1][0], uy = rotation[1][1], uz = rotation[1][2];
    float roll_deg = atan2f(ux * ex + uy * ey + uz * ez, ux * nx + uy * ny + uz * nz) * RAD_TO_DEG;

    *ra = ra_deg; *dec = dec_deg; *roll = roll_deg;
    return true;
}

/** Converts a catalog->camera rotation matrix to a unit quaternion (w,x,y,z). */
static void rotation_to_quaternion(const float m[3][3], double *qw, double *qx, double *qy, double *qz) {
    float trace = m[0][0] + m[1][1] + m[2][2];
    if (trace > 0.0f) {
        float s = sqrtf(trace + 1.0f) * 2.0f;
        *qw = 0.25 * s;
        *qx = (m[2][1] - m[1][2]) / s;
        *qy = (m[0][2] - m[2][0]) / s;
        *qz = (m[1][0] - m[0][1]) / s;
    } else if (m[0][0] > m[1][1] && m[0][0] > m[2][2]) {
        float s = sqrtf(1.0f + m[0][0] - m[1][1] - m[2][2]) * 2.0f;
        *qw = (m[2][1] - m[1][2]) / s;
        *qx = 0.25 * s;
        *qy = (m[0][1] + m[1][0]) / s;
        *qz = (m[0][2] + m[2][0]) / s;
    } else if (m[1][1] > m[2][2]) {
        float s = sqrtf(1.0f + m[1][1] - m[0][0] - m[2][2]) * 2.0f;
        *qw = (m[0][2] - m[2][0]) / s;
        *qx = (m[0][1] + m[1][0]) / s;
        *qy = 0.25 * s;
        *qz = (m[1][2] + m[2][1]) / s;
    } else {
        float s = sqrtf(1.0f + m[2][2] - m[0][0] - m[1][1]) * 2.0f;
        *qw = (m[1][0] - m[0][1]) / s;
        *qx = (m[0][2] + m[2][0]) / s;
        *qy = (m[1][2] + m[2][1]) / s;
        *qz = 0.25 * s;
    }
}

/**
 * Runs only the centroid detector on one RGB frame and copies the detected pixel
 * centroids out. Returns the number written (capped at max_out). Same detector the
 * identify_* entry points use, so the overlay shows exactly what the solver sees.
 */
extern "C" __declspec(dllexport)
int detect_centroids(const uint8_t *rgb, int width, int height, int morph_passes,
                     uint16_t *out_x, uint16_t *out_y, int max_out) {
    if (width <= 0 || height <= 0 || width * height > LIVE_MAX_PIXELS) return 0;

    static std::vector<uint8_t> dog, morph;
    dog.assign((size_t)width * height, 0);
    morph.assign((size_t)width * height, 0);

    uint16_t star_x[20], star_y[20];
    uint64_t star_brightness[20];
    int star_count = 0;
    extract_centroids(rgb, dog.data(), morph.data(), star_x, star_y, star_brightness,
                      &star_count, width, height, morph_passes);

    int n = star_count < 20 ? star_count : 20;
    if (n > max_out) n = max_out;
    for (int i = 0; i < n; ++i) { out_x[i] = star_x[i]; out_y[i] = star_y[i]; }
    return n;
}

/**
 * Identifies an attitude directly from observed unit vectors (no image, no centroiding).
 * Used by the synthetic accuracy harness: it builds observed vectors from the catalog at
 * known attitudes and checks the solve. Vectors must be brightest-first (xyz triples).
 * Returns 1 and fills RA/DEC/roll/quaternion(w,x,y,z) on a solve, 0 otherwise.
 */
extern "C" __declspec(dllexport)
int identify_vectors(const float *xyz, int n, double *out_ra, double *out_dec, double *out_roll,
                     double *out_qw, double *out_qx, double *out_qy, double *out_qz) {
    if (n < 4) return 0;
    int m = n < MAX_OBS_STARS ? n : MAX_OBS_STARS;
    ObservedStar observed[MAX_OBS_STARS];
    for (int i = 0; i < m; ++i) {
        observed[i] = (ObservedStar){xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2], (uint32_t)(m - i)};
    }
    MatchResult result;
    identify_tetra(observed, (uint8_t)m, &result);
    if (!result.success) return 0;
    if (!attitude_to_radecroll(result.catalog_to_observed, out_ra, out_dec, out_roll)) return 0;
    rotation_to_quaternion(result.catalog_to_observed, out_qw, out_qx, out_qy, out_qz);
    return 1;
}

/**
 * Identifies one RGB frame. Returns 1 and fills RA/DEC/roll/quaternion(w,x,y,z) on
 * a solve, 0 if no attitude was found, -2 if the frame is too large for the centroid
 * buffers. Scratch buffers are static (single-threaded), so no per-frame allocation churn.
 *
 * morph_passes tunes the centroid morphological open: 1 = satellite default (3x3
 * open), 0 = camera (skip the open so 1-2 px stars survive), N = repeat. See
 * extract_centroids in centroid_extract.cpp.
 */
extern "C" __declspec(dllexport)
int identify_frame(const uint8_t *rgb, int width, int height, float fov_deg,
                   double *out_ra, double *out_dec, double *out_roll,
                   double *out_qw, double *out_qx, double *out_qy, double *out_qz, int morph_passes) {
    if (width <= 0 || height <= 0 || width * height > LIVE_MAX_PIXELS) return -2;

    static std::vector<uint8_t> dog, morph;
    dog.assign((size_t)width * height, 0);
    morph.assign((size_t)width * height, 0);

    uint16_t star_x[20], star_y[20];
    uint64_t star_brightness[20];
    int star_count = 0;
    extract_centroids(rgb, dog.data(), morph.data(), star_x, star_y, star_brightness,
                      &star_count, width, height, morph_passes);
    if (star_count <= 0) return 0;

    DetectedStar detected[20];
    int n = star_count < 20 ? star_count : 20;
    for (int i = 0; i < n; ++i) {
        detected[i] = (DetectedStar){star_x[i], star_y[i], (uint32_t)star_brightness[i]};
    }

    CameraModel camera = camera_from_fov(width, height, fov_deg);
    ObservedStar observed[MAX_OBS_STARS];
    uint8_t observed_count = convert_detected_stars(detected, (uint8_t)n, &camera, observed, MAX_OBS_STARS);

    MatchResult result;
    identify_tetra(observed, observed_count, &result);
    if (!result.success) return 0;
    if (!attitude_to_radecroll(result.catalog_to_observed, out_ra, out_dec, out_roll)) return 0;
    rotation_to_quaternion(result.catalog_to_observed, out_qw, out_qx, out_qy, out_qz);
    return 1;
}

/**
 * Like identify_frame, but seed_fov_deg may be far from the true FOV. On a solve, the
 * recovered horizontal FOV is written to *out_fov_deg so the caller can lock it and use
 * identify_frame (cheap) on every later frame. Returns 1 on solve, 0 if none, -2 if the
 * frame is too large. Bootstrap-only: more expensive than identify_frame.
 */
extern "C" __declspec(dllexport)
int identify_frame_calibrate(const uint8_t *rgb, int width, int height, float seed_fov_deg,
                             double *out_ra, double *out_dec, double *out_roll,
                             double *out_qw, double *out_qx, double *out_qy, double *out_qz,
                             double *out_fov_deg, int morph_passes) {
    if (width <= 0 || height <= 0 || width * height > LIVE_MAX_PIXELS) return -2;

    static std::vector<uint8_t> dog, morph;
    dog.assign((size_t)width * height, 0);
    morph.assign((size_t)width * height, 0);

    uint16_t star_x[20], star_y[20];
    uint64_t star_brightness[20];
    int star_count = 0;
    extract_centroids(rgb, dog.data(), morph.data(), star_x, star_y, star_brightness,
                      &star_count, width, height, morph_passes);
    if (star_count <= 0) return 0;

    DetectedStar detected[20];
    int n = star_count < 20 ? star_count : 20;
    for (int i = 0; i < n; ++i) {
        detected[i] = (DetectedStar){star_x[i], star_y[i], (uint32_t)star_brightness[i]};
    }

    CameraModel camera = camera_from_fov(width, height, seed_fov_deg);
    ObservedStar observed[MAX_OBS_STARS];
    uint8_t observed_count = convert_detected_stars(detected, (uint8_t)n, &camera, observed, MAX_OBS_STARS);

    MatchResult result;
    if (!identify_tetra_calibrate(observed, observed_count, &result) || !result.success) return 0;
    if (!attitude_to_radecroll(result.catalog_to_observed, out_ra, out_dec, out_roll)) return 0;
    rotation_to_quaternion(result.catalog_to_observed, out_qw, out_qx, out_qy, out_qz);

    /* Recover the true FOV from the focal scale: f_true = f_seed * focal_scale. */
    float recovered_focal = camera.fx * result.focal_scale;
    *out_fov_deg = 2.0 * atan(((double)width * 0.5) / (double)recovered_focal) * (double)RAD_TO_DEG;
    return 1;
}
