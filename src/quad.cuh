#pragma once
#include "hittable.cuh"
#include "aabb.cuh"
#include "ray.cuh"
#include "material.cuh"
#include "vec3.cuh"

class quad : public hittable {
public:
    vec3 Q, u, v;      // origin + edge vectors
    vec3 w;            // for interior test
    vec3 normal;       // unit normal
    float D;           // plane constant
    aabb  bbox;
    material* mat_ptr;

    __device__ quad(const vec3& Q_, const vec3& u_, const vec3& v_, material* m)
        : Q(Q_), u(u_), v(v_), mat_ptr(m)
    {
        vec3 n = cross(u, v);
        normal = unit_vector(n);
        D      = dot(normal, Q);
        w      = n / dot(n, n);
        set_bounding_box();
    }

    __device__ void set_bounding_box() {
        aabb d1(Q,       Q + u + v);
        aabb d2(Q + u,   Q + v);
        // Use your helper + pad to avoid zero-width slabs (matches your style)
        bbox = aabb::surrounding_box(d1, d2).pad(1e-4f);
    }

    __device__ bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override {
        float denom = dot(normal, r.direction());
        if (fabsf(denom) < 1e-8f) return false;

        float t = (D - dot(normal, r.origin())) / denom;
        if (t < t_min || t > t_max) return false;

        vec3 P  = r.point_at_parameter(t);
        vec3 pl = P - Q;

        float alpha = dot(w, cross(pl, v));
        float beta  = dot(w, cross(u,  pl));
        if (alpha < 0.f || alpha > 1.f || beta < 0.f || beta > 1.f) return false;

        rec.t       = t;
        rec.p       = P;
        rec.u       = alpha;
        rec.v       = beta;
        rec.normal  = normal;  // you’re not using front-face in CUDA path
        rec.mat_ptr = mat_ptr;
        return true;
    }

    // Match your hittable interface exactly:
    // __device__ virtual aabb bounding_box() const = 0;
    __device__ aabb bounding_box() const override { return bbox; }
};
