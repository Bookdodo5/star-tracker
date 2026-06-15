#include "attitude.h"
#include "catalog_db.h"
#include <math.h>

/**
 * Builds a right-handed orthonormal frame from two non-collinear vectors.
 */
static bool make_frame(Vec3f primary_vector, Vec3f secondary_vector, Mat3f *frame) {
    Vec3f frame_x_axis = normalize3(primary_vector);
    Vec3f frame_y_axis = normalize3(cross3(primary_vector, secondary_vector));
    if (dot3(frame_y_axis, frame_y_axis) < 1e-6f) {
        return false;
    }
    Vec3f frame_z_axis = cross3(frame_x_axis, frame_y_axis);
    frame->m[0][0] = frame_x_axis.x; frame->m[0][1] = frame_y_axis.x; frame->m[0][2] = frame_z_axis.x;
    frame->m[1][0] = frame_x_axis.y; frame->m[1][1] = frame_y_axis.y; frame->m[1][2] = frame_z_axis.y;
    frame->m[2][0] = frame_x_axis.z; frame->m[2][1] = frame_y_axis.z; frame->m[2][2] = frame_z_axis.z;
    return true;
}

/**
 * Multiplies matrix a by transpose(matrix b), used by TRIAD frame alignment.
 */
static Mat3f mul_transpose_right(const Mat3f *left_matrix, const Mat3f *right_matrix) {
    Mat3f product = {{{0}}};
    for (uint8_t row_index = 0; row_index < 3; ++row_index) {
        for (uint8_t column_index = 0; column_index < 3; ++column_index) {
            for (uint8_t axis_index = 0; axis_index < 3; ++axis_index) {
                product.m[row_index][column_index] +=
                    left_matrix->m[row_index][axis_index] * right_matrix->m[column_index][axis_index];
            }
        }
    }
    return product;
}

/**
 * Estimates catalog-to-observed attitude using the TRIAD method.
 */
bool solve_attitude_triad(
    const uint16_t *hr_ids,
    const uint8_t *obs_ids,
    uint8_t count,
    const ObservedStar *observed_stars,
    Mat3f *rotation
) {
    if (count < 2) {
        return false;
    }

    Vec3f catalog_a;
    Vec3f catalog_b;
    if (!catalog_vector(hr_ids[0], &catalog_a) || !catalog_vector(hr_ids[1], &catalog_b)) {
        return false;
    }

    Vec3f observed_primary_vector = {
        observed_stars[obs_ids[0]].x,
        observed_stars[obs_ids[0]].y,
        observed_stars[obs_ids[0]].z,
    };
    Vec3f observed_secondary_vector = {
        observed_stars[obs_ids[1]].x,
        observed_stars[obs_ids[1]].y,
        observed_stars[obs_ids[1]].z,
    };
    Mat3f catalog_frame;
    Mat3f observed_frame;
    if (!make_frame(catalog_a, catalog_b, &catalog_frame) ||
        !make_frame(observed_primary_vector, observed_secondary_vector, &observed_frame)) {
        return false;
    }

    *rotation = mul_transpose_right(&observed_frame, &catalog_frame);
    return true;
}
