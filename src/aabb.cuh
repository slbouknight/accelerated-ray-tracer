#ifndef AABB_CUH
#define AABB_CUH

#include <float.h>
#include "vec3.cuh"
#include "ray.cuh"

class aabb
{
    public:
        vec3 minimum;
        vec3 maximum;

        __host__ __device__ aabb() : minimum(vec3( FLT_MAX,  FLT_MAX,  FLT_MAX)),
            maximum(vec3(-FLT_MAX, -FLT_MAX, -FLT_MAX)) {}

        __host__ __device__ aabb(const vec3& a, const vec3& b) 
        {
            minimum = vec3(fminf(a.x(), b.x()), fminf(a.y(), b.y()), fminf(a.z(), b.z()));
            maximum = vec3(fmaxf(a.x(), b.x()), fmaxf(a.y(), b.y()), fmaxf(a.z(), b.z()));
        }

        __host__ __device__ const vec3& min() const { return minimum; }
        __host__ __device__ const vec3& max() const { return maximum; }

        // Pad box to avoid zero-width slabs
        __host__ __device__ aabb pad(float delta) const 
        {
            vec3 d(delta, delta, delta);
            return aabb(minimum - d, maximum + d);
        }

        // Surrounding box helper
        __host__ __device__ static aabb surrounding_box(const aabb& box0, const aabb& box1) 
        {
            vec3 small(fminf(box0.minimum.x(), box1.minimum.x()),
                    fminf(box0.minimum.y(), box1.minimum.y()),
                    fminf(box0.minimum.z(), box1.minimum.z()));
            vec3 big  (fmaxf(box0.maximum.x(), box1.maximum.x()),
                    fmaxf(box0.maximum.y(), box1.maximum.y()),
                    fmaxf(box0.maximum.z(), box1.maximum.z()));
            return aabb(small, big);
        }

        __device__ bool hit(const ray& r, float tmin, float tmax) const 
        {
        for (int a = 0; a < 3; a++) {
            float invD = 1.0f / r.direction()[a];
            float t0 = (min()[a] - r.origin()[a]) * invD;
            float t1 = (max()[a] - r.origin()[a]) * invD;
            if (invD < 0.0f) {
                float tmp = t0;
                t0 = t1;
                t1 = tmp;
            }
            tmin = t0 > tmin ? t0 : tmin;
            tmax = t1 < tmax ? t1 : tmax;
            if (tmax <= tmin) return false;
        }
        return true;
    }
};

__host__ __device__ static aabb empty() 
{
    return aabb(vec3( FLT_MAX,  FLT_MAX,  FLT_MAX),
                vec3(-FLT_MAX, -FLT_MAX, -FLT_MAX));
}
__host__ __device__ static aabb universe() 
{
    return aabb(vec3(-FLT_MAX, -FLT_MAX, -FLT_MAX),
                vec3( FLT_MAX,  FLT_MAX,  FLT_MAX));
}

// Shift an AABB by an offset (component-wise).
__host__ __device__ inline aabb operator+(const aabb& box, const vec3& offset) 
{
    return aabb(box.minimum + offset, box.maximum + offset);
}


#endif