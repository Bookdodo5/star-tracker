#ifndef TETRA_DB_H
#define TETRA_DB_H

#include <stdint.h>

#define TETRA_DIM 5
#define KD_NULL (-1)

typedef struct {
    uint16_t f[5];
    uint16_t hr[4];
    int32_t left;
    int32_t right;
    uint8_t axis;
} TetraKdNode;

extern const TetraKdNode tetra_kd_nodes[];
extern const uint32_t tetra_kd_node_count;

#endif
