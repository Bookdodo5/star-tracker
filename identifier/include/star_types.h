#ifndef STAR_TYPES_H
#define STAR_TYPES_H

#include <stdbool.h>
#include <stdint.h>

#define STAR_Q15 32767
#define HR_INVALID 0xFFFFu
/* Observed star pool: real images carry non-catalog blobs (galaxies, blooming,
   sub-catalog stars), so the pool is larger than the per-field catalog star count
   to ensure enough true catalog stars survive for verification. */
#define MAX_OBS_STARS 20
#define MAX_MATCHES 20
#define PYRAMID_SIZE 4

/**
 * Centroid output before camera calibration.
 */
typedef struct {
    uint16_t x;
    uint16_t y;
    uint32_t brightness;
} DetectedStar;

/**
 * Observed star direction after pixel-to-vector conversion.
 */
typedef struct {
    float x;
    float y;
    float z;
    uint32_t brightness;
} ObservedStar;

/**
 * Compact catalog star stored as a unit vector and magnitude.
 */
typedef struct {
    uint16_t hr;
    int16_t x;
    int16_t y;
    int16_t z;
    int16_t mag_q100;
} CatalogStar;

/**
 * Independent result returned by one identification algorithm.
 */
typedef struct {
    uint16_t hr_ids[MAX_MATCHES];
    uint8_t obs_ids[MAX_MATCHES];
    uint16_t residual_arcsec[MAX_MATCHES];
    float catalog_to_observed[3][3];
    uint8_t count;
    uint16_t mean_residual_arcsec;
    uint16_t max_residual_arcsec;
    int32_t score;
    bool success;
    /* Per-step timing populated by the identifier (microseconds, clock() resolution).
       db_us covers candidate generation + database search; verify_us covers the
       cumulative time spent in verify_attitude. Camera conversion is timed by the
       caller, not here. */
    uint32_t db_us;
    uint32_t verify_us;
} MatchResult;

#endif
