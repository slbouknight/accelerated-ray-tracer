#ifndef SPHEREH
#define SPHEREH

#include "hittable.h"
#include "material.h"

class sphere : public hittable
{
    public:
        __device__ sphere() {}

        // Static sphere -> zero velocity
    __device__ sphere(vec3 cen, float r, material* m)
        : center(ray(cen, vec3(0,0,0), 0.0)), radius(r), mat_ptr(m) {}

    // Moving sphere -> velocity = cen2 - cen1
    __device__ sphere(vec3 cen1, vec3 cen2, float r, material* m)
        : center(ray(cen1, cen2 - cen1, 0.0)), radius(r), mat_ptr(m) {}

        __device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;
        ray center;
        float radius;
        material *mat_ptr;
};

__device__ bool sphere::hit(const ray& r, float t_min, float t_max, hit_record& rec) const
{
    vec3 current_center = center.point_at_parameter(r.time());
    vec3 oc = r.origin() - current_center;
    float a = dot(r.direction(), r.direction());
    float b = dot(oc, r.direction());
    float c = dot(oc, oc) - radius*radius;
    float discriminant = b*b -a*c;

    if (discriminant > 0)
    {
        // Check negative root
        float temp = (-b - sqrt(discriminant)) / a;
        if (temp < t_max && temp > t_min)
        {
            rec.t = temp;
            rec.p = r.point_at_parameter(rec.t);
            rec.normal = (rec.p - current_center) / radius;
            rec.mat_ptr = mat_ptr;
            return true;
        }

        // Check positive root
        temp = (-b + sqrt(discriminant)) / a;
        if (temp < t_max && temp > t_min)
        {
            rec.t = temp;
            rec.p = r.point_at_parameter(rec.t);
            rec.normal = (rec.p - current_center) / radius;
            rec.mat_ptr = mat_ptr;
            return true;
        }
    }
    return false;
}

#endif