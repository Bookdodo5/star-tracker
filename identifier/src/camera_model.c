#include "camera_model.h"
#include "star_math.h"

/**
 * Converts one centroid from pixel coordinates into a normalized camera vector.
 */
ObservedStar pixel_to_unit_vector(const DetectedStar *star, const CameraModel *camera) {
    /* Pixel coordinates are centered and scaled by focal length before normalization. */
    /* Real sky images follow the physical astronomical convention: north up, east
       left (looking outward at the celestial sphere). Pixel X increases to the
       right (west) and pixel Y increases downward (south), so both axes are
       negated to recover a right-handed (east, north, boresight) camera frame
       aligned with catalog coordinates. Without this the observed frame is a
       mirror of the catalog and solve_attitude_triad cannot recover a rotation. */
    Vec3f raw = {
        (camera->cx - (float)star->x) / camera->fx,
        (camera->cy - (float)star->y) / camera->fy,
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
