#ifndef SPHEREH
#define SPHEREH

#include "hittable.h"
#include "material.h"
#include <math.h>

class sphere : public hittable {
public:
    __device__ sphere() {}

    // Static sphere -> zero velocity
    __device__ sphere(vec3 cen, float r, material* m)
        : center(ray(cen, vec3(0,0,0), 0.0)), radius(r), mat_ptr(m)
    {
        vec3 rvec(radius, radius, radius);
        bbox = aabb(cen - rvec, cen + rvec);
    }

    // Moving sphere -> velocity = cen2 - cen1 (parameterized over [0,1])
    __device__ sphere(vec3 cen1, vec3 cen2, float r, material* m)
        : center(ray(cen1, cen2 - cen1, 0.0)), radius(r), mat_ptr(m)
    {
        vec3 rvec(radius, radius, radius);
        vec3 c0 = center.point_at_parameter(0.0);
        vec3 c1 = center.point_at_parameter(1.0);
        aabb box0(c0 - rvec, c0 + rvec);
        aabb box1(c1 - rvec, c1 + rvec);
        bbox = aabb::surrounding_box(box0, box1);
    }

    __device__ aabb bounding_box() const override { return bbox; }

    __device__ bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override 
    {
        // evaluate center at the ray's shutter time
        vec3 current_center = center.point_at_parameter(r.time());

        vec3 oc = r.origin() - current_center;
        float a = dot(r.direction(), r.direction());
        float b = dot(oc, r.direction());                     // half‑b
        float c = dot(oc, oc) - radius * radius;
        float disc = b*b - a*c;
        if (disc <= 0.0f) return false;

        float s = sqrtf(disc);

        // smaller root
        float t = (-b - s) / a;
        if (t > t_min && t < t_max) 
        {
            rec.t = t;
            rec.p = r.point_at_parameter(t);
            rec.normal = (rec.p - current_center) / radius;
            rec.mat_ptr = mat_ptr;
            return true;
        }

        // larger root
        t = (-b + s) / a;
        if (t > t_min && t < t_max) 
        {
            rec.t = t;
            rec.p = r.point_at_parameter(t);
            rec.normal = (rec.p - current_center) / radius;
            rec.mat_ptr = mat_ptr;
            return true;
        }
        return false;
    }

    // members (POD style)
    ray center;          // encodes c(t) = c0 + t*(c1-c0), t in [0,1]
    float radius;
    material* mat_ptr;
    aabb bbox;
};

#endif