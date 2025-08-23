#pragma once
#include <cuda_runtime.h>
#include "vec3.cuh"

// Tiny integer hash utilities (device-only)
__device__ __forceinline__ unsigned int wanghash(unsigned int x) {
    x = (x ^ 61u) ^ (x >> 16);
    x *= 9u;
    x = x ^ (x >> 4);
    x *= 0x27d4eb2du;
    x = x ^ (x >> 15);
    return x;
}
__device__ __forceinline__ unsigned int mix3(int x, int y, int z) {
    return (unsigned int)(x) * 73856093u ^ (unsigned int)(y) * 19349663u ^ (unsigned int)(z) * 83492791u;
}
// map 32-bit int to [-1,1]
__device__ __forceinline__ float u2m11(unsigned int h) {
    // keep upper 24 bits for better distribution
    return (float)((h >> 8) & 0x00FFFFFF) * (1.0f / 8388607.5f) - 1.0f; // [-1,1]
}

struct perlin {
    // Book’s smoothstep (3t^2 - 2t^3)
    __device__ static float smooth(float t) { return t*t*(3.0f - 2.0f*t); }

    // Pseudo-random unit vector for lattice point (xi, yi, zi)
    __device__ static vec3 grad(int xi, int yi, int zi) {
        unsigned int h = wanghash(mix3(xi, yi, zi));
        vec3 v(u2m11(h), u2m11(wanghash(h)), u2m11(wanghash(h ^ 0x9e3779b9u)));
        return unit_vector(v);
    }

    __device__ static float perlin_interp(const vec3 c[2][2][2], float u, float v, float w) {
        float uu = smooth(u), vv = smooth(v), ww = smooth(w);
        float accum = 0.0f;
        #pragma unroll
        for (int i = 0; i < 2; ++i)
        #pragma unroll
        for (int j = 0; j < 2; ++j)
        #pragma unroll
        for (int k = 0; k < 2; ++k) {
            vec3 weight(u - i, v - j, w - k);
            float s = (i ? uu : (1.0f - uu))
                    * (j ? vv : (1.0f - vv))
                    * (k ? ww : (1.0f - ww));
            accum += s * dot(c[i][j][k], weight);
        }
        return accum;
    }

    __device__ static float noise(const vec3& p) {
        float u = p.x() - floorf(p.x());
        float v = p.y() - floorf(p.y());
        float w = p.z() - floorf(p.z());
        int i = (int)floorf(p.x());
        int j = (int)floorf(p.y());
        int k = (int)floorf(p.z());

        vec3 c[2][2][2];
        #pragma unroll
        for (int di = 0; di < 2; ++di)
        #pragma unroll
        for (int dj = 0; dj < 2; ++dj)
        #pragma unroll
        for (int dk = 0; dk < 2; ++dk) {
            c[di][dj][dk] = grad(i + di, j + dj, k + dk);
        }
        return perlin_interp(c, u, v, w);
    }

    __device__ static float turb(const vec3& p, int depth) {
        float accum = 0.0f;
        vec3  temp = p;
        float weight = 1.0f;
        for (int i = 0; i < depth; ++i) {
            accum  += weight * noise(temp);
            weight *= 0.5f;
            temp   *= 2.0f;
        }
        return fabsf(accum);
    }
};
