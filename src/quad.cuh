#pragma once

#include <math.h>

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

// Concrete container for exactly six faces (a box). No interval usage.
class compound6 : public hittable {
public:
    hittable* faces[6];
    aabb      box;

    __device__ compound6(hittable* f0, hittable* f1, hittable* f2,
                         hittable* f3, hittable* f4, hittable* f5) {
        faces[0]=f0; faces[1]=f1; faces[2]=f2;
        faces[3]=f3; faces[4]=f4; faces[5]=f5;

        // Union the child AABBs
        box = faces[0]->bounding_box();
        #pragma unroll
        for (int i=1;i<6;++i) {
            const aabb b = faces[i]->bounding_box();
            const vec3 mn(fminf(box.minimum.x(), b.minimum.x()),
                          fminf(box.minimum.y(), b.minimum.y()),
                          fminf(box.minimum.z(), b.minimum.z()));
            const vec3 mx(fmaxf(box.maximum.x(), b.maximum.x()),
                          fmaxf(box.maximum.y(), b.maximum.y()),
                          fmaxf(box.maximum.z(), b.maximum.z()));
            box = aabb(mn, mx);
        }
    }

    __device__ bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const override {
        // No AABB early-out that mutates intervals; just do a closest-hit scan.
        bool  hit_any = false;
        float closest = tmax;

        #pragma unroll
        for (int i=0;i<6;++i) {
            hit_record tmp;
            if (faces[i]->hit(r, tmin, closest, tmp)) {
                hit_any = true;
                closest = tmp.t;
                rec     = tmp;
            }
        }
        return hit_any;
    }

    __device__ aabb bounding_box() const override { return box; }
    __device__ HKind kind() const override { return HK_Composite; }
};

__device__ inline hittable* make_box(const vec3& a, const vec3& b, material* mat) 
{
    const vec3 minp(fminf(a.x(), b.x()), fminf(a.y(), b.y()), fminf(a.z(), b.z()));
    const vec3 maxp(fmaxf(a.x(), b.x()), fmaxf(a.y(), b.y()), fmaxf(a.z(), b.z()));

    const vec3 dx(maxp.x() - minp.x(), 0.f, 0.f);
    const vec3 dy(0.f, maxp.y() - minp.y(), 0.f);
    const vec3 dz(0.f, 0.f, maxp.z() - minp.z());

    hittable* front  = new quad(vec3(minp.x(), minp.y(), maxp.z()),  dx,  dy, mat); // +Z
    hittable* right  = new quad(vec3(maxp.x(), minp.y(), maxp.z()), -dz,  dy, mat); // +X
    hittable* back   = new quad(vec3(maxp.x(), minp.y(), minp.z()), -dx,  dy, mat); // -Z
    hittable* left   = new quad(vec3(minp.x(), minp.y(), minp.z()),  dz,  dy, mat); // -X
    hittable* top    = new quad(vec3(minp.x(), maxp.y(), maxp.z()),  dx, -dz, mat); // +Y
    hittable* bottom = new quad(vec3(minp.x(), minp.y(), minp.z()),  dx,  dz, mat); // -Y

    return new compound6(front, right, back, left, top, bottom);
}
