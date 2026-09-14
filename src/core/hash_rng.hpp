#pragma once

#include "cuda_compat.hpp"
#include "vec3.hpp"

// Deterministic xorshift-based sample in [0,1)^3, keyed only by `seed`.
// Used where a scene needs reproducible pseudo-randomness without cuRAND
// state -- notably the 1000-sphere cluster in the final scene.
RT_HD inline vec3 random_in_unit_cube(int seed) {
    // deterministic, no curand
    unsigned int s = 1103515245u * (seed + 1) + 12345u;
    auto next01 = [&]() {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        return (s & 0xFFFFFF) * (1.0f / 16777216.0f); // [0,1)
    };
    return vec3(next01(), next01(), next01());
}

