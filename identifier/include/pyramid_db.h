#ifndef PYRAMID_DB_H
#define PYRAMID_DB_H

#include <stdint.h>

typedef struct {
    uint16_t hr_a;
    uint16_t hr_b;
    uint16_t sep_code;
} PairRow;

typedef struct {
    uint16_t hr_id;
    uint16_t sep_code;
} PairNeighbor;

extern const PairRow pyramid_pairs_by_sep[];
extern const uint32_t pyramid_pair_count;
extern const PairNeighbor pyramid_neighbors_by_hr[];
extern const uint32_t pyramid_neighbor_count;
extern const uint32_t pyramid_neighbor_starts[];
extern const uint32_t pyramid_neighbor_start_count;
extern const float pyramid_max_sep_rad;

#endif
