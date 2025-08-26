#pragma once
#include <float.h>
#include <math.h>
#include "hittable.cuh"
#include "material.cuh"
#include "texture.cuh"
#include "aabb.cuh"

// tiny xorshift RNG to avoid needing curand in the object itself
__device__ inline float rng01_from_seed(unsigned int seed) {
    seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
    // 24-bit mantissa scale -> [0,1)
    return (seed & 0xFFFFFF) * (1.0f / 16777216.0f);
}

class constant_medium : public hittable {
public:
    hittable*  boundary;       // owned
    float      neg_inv_density;
    material*  phase_function; // owned

    __device__ HKind kind() const override { return HK_Composite; }

    __device__ constant_medium(hittable* b, float density, texture* tex)
        : boundary(b), neg_inv_density(-1.0f / density), phase_function(new isotropic(tex)) {}

    __device__ constant_medium(hittable* b, float density, const vec3& albedo)
        : boundary(b), neg_inv_density(-1.0f / density), phase_function(new isotropic(new solid_color(albedo))) {}

    __device__ ~constant_medium() override {
        delete boundary;
        delete phase_function;
    }

    // Preferred path: uses the caller’s RNG
    __device__ bool hit(const ray& r, float tmin, float tmax,
                        hit_record& rec, curandState* rng) const override {
        hit_record rec1, rec2;
        if (!boundary->hit(r, -FLT_MAX,  FLT_MAX, rec1, rng)) return false;
        if (!boundary->hit(r, rec1.t + 1e-4f, FLT_MAX, rec2, rng)) return false;

        if (rec1.t < tmin) rec1.t = tmin;
        if (rec2.t > tmax) rec2.t = tmax;
        if (rec1.t >= rec2.t) return false;
        if (rec1.t < 0)       rec1.t = 0;

        const float ray_len = r.direction().length();
        if (ray_len <= 0.0f || !isfinite(ray_len)) return false;

        const float distance_inside = (rec2.t - rec1.t) * ray_len;

        // Exponential free-flight sampling
        float U = fmaxf(1e-6f, curand_uniform(rng));
        const float hit_distance = neg_inv_density * logf(U);   // (-1/d) * log U

        if (hit_distance > distance_inside) return false;

        rec.t      = rec1.t + hit_distance / ray_len;
        rec.p      = r.point_at_parameter(rec.t);
        rec.normal = vec3(1,0,0);    // arbitrary in volumes
        rec.u = rec.v = 0.0f;
        rec.mat_ptr = phase_function;  // IMPORTANT: match your field name
        return true;
    }

    // Fallback: keep a deterministic path if no RNG was provided
    __device__ bool hit(const ray& r, float tmin, float tmax,
                        hit_record& rec) const override {
        curandState fake;
        unsigned int seed = 1337u
            ^ __float_as_uint(r.origin().x())
            ^ __float_as_uint(r.origin().y()*3.1f)
            ^ __float_as_uint(r.direction().z()*5.7f);
        curand_init(seed, 0, 0, &fake);
        return hit(r, tmin, tmax, rec, &fake);
    }


    __device__ aabb bounding_box() const override { return boundary->bounding_box(); }
};
