#pragma once
#include "hittable.cuh"
#include "aabb.cuh"
#include "ray.cuh"
#include "material.cuh"
#include "vec3.cuh"

class quad : public hittable 
{
public:
    vec3 Q, u, v;        // origin + edge vectors
    vec3 w;              // for interior test in plane
    vec3 normal;         // geometric unit normal (from cross(u,v), possibly flipped by 'inward')
    float D;             // plane constant: dot(n, Q)
    aabb  bbox;
    material* mat_ptr;
    bool inward;         // if true, flip the geometric normal
    bool owns_mat;       // delete material in dtor if true

    // NOTE: the 5th arg is *inward*, 6th arg is *owns_mat* (defaults true to preserve old call sites)
    __device__ quad(const vec3& Q_, const vec3& u_, const vec3& v_,
                    material* m, bool inward_ = false, bool owns_ = true)
        : Q(Q_), u(u_), v(v_), mat_ptr(m), inward(inward_), owns_mat(owns_) {

        vec3 n = cross(u, v);             // (u,v) order matters!
        normal = unit_vector(n);
        if (inward) normal = -normal;     // flip if caller asked for inward-facing

        D = dot(normal, Q);
        w = n / dot(n, n);                // for (alpha,beta) interior test

        set_bounding_box();
    }

    __device__ ~quad() override {
        if (owns_mat && mat_ptr) delete mat_ptr;
    }

    __device__ HKind kind() const override { return HK_Quad; }

    __device__ void set_bounding_box() {
        // Bounding box from the four corners, padded to avoid zero-thickness slabs.
        aabb d1(Q,       Q + u + v);
        aabb d2(Q + u,   Q + v);
        bbox = aabb::surrounding_box(d1, d2).pad(1e-3f);
    }

    __device__ aabb bounding_box() const override {
        return bbox;
    }

    __device__ bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override {
        const float denom = dot(normal, r.direction());

        // Parallel to plane?
        if (fabsf(denom) < 1e-8f) return false;

        const float t = (D - dot(normal, r.origin())) / denom;
        if (t < t_min || t > t_max) return false;

        // Plane coordinates
        const vec3 P  = r.point_at_parameter(t);
        const vec3 pl = P - Q;

        const float alpha = dot(w, cross(pl, v));
        const float beta  = dot(w, cross(u,  pl));
        if (alpha < 0.f || alpha > 1.f || beta < 0.f || beta > 1.f) return false;

        // Fill hit record
        rec.t       = t;
        rec.p       = P;
        rec.u       = alpha;
        rec.v       = beta;

        // Orient shading normal to face the incoming ray (serial's set_face_normal analogue)
        vec3 n = normal;
        if (dot(n, r.direction()) > 0.f) n = -n;
        rec.normal  = n;

        rec.mat_ptr = mat_ptr;
        return true;
    }
};
