#include "camera_model.h"
#include "star_math.h"

/**
 * Converts one centroid from pixel coordinates into a normalized camera vector.
 */
ObservedStar pixel_to_unit_vector(const DetectedStar *star, const CameraModel *camera) {
    /* Pixel coordinates are centered and scaled by focal length before normalization. */
    Vec3f raw = {
        ((float)star->x - camera->cx) / camera->fx,
        ((float)star->y - camera->cy) / camera->fy,
        1.0f,
    };
    Vec3f unit = normalize3(raw);
    return (ObservedStar){unit.x, unit.y, unit.z, star->brightness};
}

/**
 * Converts up to max_output detected centroids into observed unit vectors.
 */
uint8_t convert_detected_stars(
    const DetectedStar *input,
    uint8_t input_count,
    const CameraModel *camera,
    ObservedStar *output,
    uint8_t max_output
) {
    uint8_t count = input_count < max_output ? input_count : max_output;
    for (uint8_t star_index = 0; star_index < count; ++star_index) {
        output[star_index] = pixel_to_unit_vector(&input[star_index], camera);
    }
    return count;
}
