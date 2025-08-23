#ifndef HITTABLE_CUH
#define HITTABLE_CUH

#include "aabb.cuh"
#include "ray.cuh"

class material;

struct hit_record
{
    float t;
    vec3 p;
    vec3 normal;
    material *mat_ptr;
    double u;
    double v;
};

class hittable
{
    public:
        __device__ virtual ~hittable() = default;
        __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const = 0;
        __device__ virtual aabb bounding_box() const = 0;
};

#endif