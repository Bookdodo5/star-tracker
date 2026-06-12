#include "star_math.h"
#include <math.h>

/**
 * Converts a fixed-point catalog vector component triplet into float form.
 */
Vec3f q15_to_vec3f(int16_t x, int16_t y, int16_t z) {
    const float scale = 1.0f / (float)STAR_Q15;
    return (Vec3f){(float)x * scale, (float)y * scale, (float)z * scale};
}

/**
 * Computes the dot product of two 3D vectors.
 */
float dot3(Vec3f left_vector, Vec3f right_vector) {
    return left_vector.x * right_vector.x + left_vector.y * right_vector.y + left_vector.z * right_vector.z;
}

/**
 * Computes the cross product of two 3D vectors.
 */
Vec3f cross3(Vec3f left_vector, Vec3f right_vector) {
    return (Vec3f){
        left_vector.y * right_vector.z - left_vector.z * right_vector.y,
        left_vector.z * right_vector.x - left_vector.x * right_vector.z,
        left_vector.x * right_vector.y - left_vector.y * right_vector.x,
    };
}

/**
 * Returns a unit-length vector, falling back to optical axis for zero input.
 */
Vec3f normalize3(Vec3f vector) {
    float norm = sqrtf(dot3(vector, vector));
    if (norm <= 0.0f) {
        return (Vec3f){0.0f, 0.0f, 1.0f};
    }
    return (Vec3f){vector.x / norm, vector.y / norm, vector.z / norm};
}

/**
 * Multiplies a 3x3 matrix by a vector.
 */
Vec3f mat3_mul_vec3(const Mat3f *matrix, Vec3f vector) {
    return (Vec3f){
        matrix->m[0][0] * vector.x + matrix->m[0][1] * vector.y + matrix->m[0][2] * vector.z,
        matrix->m[1][0] * vector.x + matrix->m[1][1] * vector.y + matrix->m[1][2] * vector.z,
        matrix->m[2][0] * vector.x + matrix->m[2][1] * vector.y + matrix->m[2][2] * vector.z,
    };
}

/**
 * Computes angular distance in radians using normalized vector dot product.
 */
float angular_distance_rad(Vec3f first_vector, Vec3f second_vector) {
    float cosine = dot3(normalize3(first_vector), normalize3(second_vector));
    if (cosine > 1.0f) cosine = 1.0f;
    if (cosine < -1.0f) cosine = -1.0f;
    return acosf(cosine);
}

/**
 * Quantizes an angle into the uint16 range used by pair databases.
 */
uint16_t angle_to_code(float angle_rad, float max_angle_rad) {
    float scaled = angle_rad / max_angle_rad * 65535.0f;
    if (scaled <= 0.0f) return 0;
    if (scaled >= 65535.0f) return 65535u;
    return (uint16_t)(scaled + 0.5f);
}

/**
 * Sorts a small uint16 array in-place with insertion sort.
 */
void sort_u16(uint16_t *values, uint8_t count) {
    for (uint8_t unsorted_index = 1; unsorted_index < count; ++unsorted_index) {
        uint16_t value = values[unsorted_index];
        uint8_t insert_index = unsorted_index;
        while (insert_index > 0 && values[insert_index - 1] > value) {
            values[insert_index] = values[insert_index - 1];
            --insert_index;
        }
        values[insert_index] = value;
    }
}
