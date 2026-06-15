#ifndef STAR_MATH_H
#define STAR_MATH_H

#include <stdint.h>
#include "star_types.h"

typedef struct {
    float x;
    float y;
    float z;
} Vec3f;

typedef struct {
    float m[3][3];
} Mat3f;

/**
 * Converts Q15 catalog coordinates to a float vector.
 */
Vec3f q15_to_vec3f(int16_t x, int16_t y, int16_t z);

float dot3(Vec3f left_vector, Vec3f right_vector);
Vec3f cross3(Vec3f left_vector, Vec3f right_vector);
Vec3f normalize3(Vec3f vector);
Vec3f mat3_mul_vec3(const Mat3f *matrix, Vec3f vector);
float angular_distance_rad(Vec3f first_vector, Vec3f second_vector);
uint16_t angle_to_code(float angle_rad, float max_angle_rad);
void sort_u16(uint16_t *values, uint8_t count);

#endif
