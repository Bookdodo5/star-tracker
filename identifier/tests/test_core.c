#include "attitude.h"
#include "camera_model.h"
#include "catalog_db.h"
#include "identify_tetra.h"
#include "star_math.h"
#include "tetra_db.h"
#include "verify.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>

const CatalogStar catalog_stars[] = {
    {1, 32767, 0, 0, 100},
    {2, 0, 32767, 0, 100},
    {3, 0, 0, 32767, 100},
    {4, -32767, 0, 0, 100},
};

const uint16_t catalog_star_count = 4;
const uint16_t hr_to_catalog_index_count = 5;
const uint16_t hr_to_catalog_index[] = {
    HR_INVALID,
    0,
    1,
    2,
    3,
};

/* Minimal 4-node KD-tree over the 4 mock catalog stars.
   Build order (axis=x for all nodes, sorted by x ascending):
     node 0: (0,0,1) hr=3  left=1 right=3
     node 1: (0,1,0) hr=2  left=2 right=-1
     node 2: (-1,0,0) hr=4  left=-1 right=-1
     node 3: (1,0,0) hr=1  left=-1 right=-1   */
const CatalogKdNode catalog_kd_nodes[] = {
    { 0.0f,  0.0f,  1.0f, 0.0f,  1,  3, 3, 0, 0},
    { 0.0f,  1.0f,  0.0f, 0.0f,  2, -1, 2, 0, 0},
    {-1.0f,  0.0f,  0.0f, 0.0f, -1, -1, 4, 0, 0},
    { 1.0f,  0.0f,  0.0f, 0.0f, -1, -1, 1, 0, 0},
};
const uint32_t catalog_kd_node_count = 4;

const TetraKdNode tetra_kd_nodes[] = {
    {{32767, 32767, 32767, 32767, 32767}, {1, 2, 3, 4}, KD_NULL, KD_NULL, 0},
};

const uint32_t tetra_kd_node_count = 1;

/**
 * Verifies that the camera center maps to the optical axis.
 */
static void test_camera_center(void) {
    CameraModel camera = {100.0f, 100.0f, 50.0f, 50.0f};
    DetectedStar detected = {50, 50, 1000};
    ObservedStar observed = pixel_to_unit_vector(&detected, &camera);
    assert(fabsf(observed.x) < 1e-6f);
    assert(fabsf(observed.y) < 1e-6f);
    assert(fabsf(observed.z - 1.0f) < 1e-6f);
}

/**
 * Verifies a known 90-degree angular distance.
 */
static void test_angle(void) {
    Vec3f x = {1.0f, 0.0f, 0.0f};
    Vec3f y = {0.0f, 1.0f, 0.0f};
    float angle = angular_distance_rad(x, y);
    assert(fabsf(angle - 1.57079632679f) < 1e-5f);
}

/**
 * Verifies that identity attitude matches all mock catalog stars.
 */
static void test_verify_identity(void) {
    ObservedStar observed_stars[] = {
        {1.0f, 0.0f, 0.0f, 100},
        {0.0f, 1.0f, 0.0f, 100},
        {0.0f, 0.0f, 1.0f, 100},
        {-1.0f, 0.0f, 0.0f, 100},
    };
    Mat3f identity = {{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
    MatchResult result;
    assert(verify_attitude(&identity, observed_stars, 4, &result));
    assert(result.count == 4);
    assert(result.mean_residual_arcsec == 0);
}

/**
 * Verifies that TETRA identifies the mock field.
 */
static void test_tetra_identify(void) {
    ObservedStar observed_stars[] = {
        {1.0f, 0.0f, 0.0f, 100},
        {0.0f, 1.0f, 0.0f, 100},
        {0.0f, 0.0f, 1.0f, 100},
        {-1.0f, 0.0f, 0.0f, 100},
    };
    MatchResult result;
    identify_tetra(observed_stars, 4, &result);
    assert(result.success);
    assert(result.count == 4);
}

/**
 * Runs all C unit tests for the identification core.
 */
int main(void) {
    test_camera_center();
    test_angle();
    test_verify_identity();
    test_tetra_identify();
    puts("C star identifier tests passed");
    return 0;
}
