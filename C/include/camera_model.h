#ifndef CAMERA_MODEL_H
#define CAMERA_MODEL_H

#include "star_types.h"

typedef struct {
    float fx;
    float fy;
    float cx;
    float cy;
} CameraModel;

/**
 * Converts one centroid pixel to a camera-frame unit vector.
 */
ObservedStar pixel_to_unit_vector(const DetectedStar *star, const CameraModel *camera);

/**
 * Converts and truncates detected stars for identification.
 */
uint8_t convert_detected_stars(
    const DetectedStar *input,
    uint8_t input_count,
    const CameraModel *camera,
    ObservedStar *output,
    uint8_t max_output
);

#endif
