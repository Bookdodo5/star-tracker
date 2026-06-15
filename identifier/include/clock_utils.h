#ifndef CLOCK_UTILS_H
#define CLOCK_UTILS_H

#include <stdint.h>
#include <time.h>

/**
 * Converts a pair of clock ticks into microseconds.
 */
static inline uint32_t elapsed_us(clock_t start, clock_t end) {
    return (uint32_t)(((double)(end - start) * 1000000.0) / (double)CLOCKS_PER_SEC);
}

#endif
