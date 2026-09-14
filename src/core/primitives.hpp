#pragma once

#include "aabb.hpp"
#include "cuda_compat.hpp"
#include "ray.hpp"
#include "shading_math.hpp"
#include "vec3.hpp"

#include <cfloat>

// Flat, POD geometry.
//
// This replaces the `__device__ virtual hittable` hierarchy. Three things fall
// out of that, in order of how much they matter:
//
//   1. A POD struct has no vtable, so it can be built on the host and
//      cudaMemcpy'd to the device. That is what lets the BVH be built on the
//      CPU instead of by one CUDA thread. (A vtable pointer is a device
//      address; a polymorphic object memcpy'd from the host would fault on its
//      first virtual call. That single fact is what forced the old design.)
//   2. Dispatch becomes a switch on a small enum rather than an indirect call,
//      so intersection code inlines and a warp straddling two primitive types
//      costs one predicated branch instead of two serialised call targets.
//   3. Primitives live in contiguous arrays, so the loads coalesce instead of
//      chasing pointers around the device heap.
//
// Everything here is dual-compiled: pure float math, no RNG state, no
// allocation. The intersection routines are therefore unit-tested on the host,
// which is most of the point.

enum PrimKind : int {
    PRIM_SPHERE = 0,
    PRIM_QUAD   = 1,
    PRIM_MEDIUM = 2,
};

// What a hit returns. `mat` indexes the scene's material table rather than
// pointing at a material, so the struct stays POD and host-constructible.
struct Hit {
    float t;
    vec3  p;
    vec3  normal;
    float u, v;
    int   mat;
};

// ---------------------------------------------------------------------------
// Sphere. A static sphere is just one whose centre does not move, which keeps a
// single code path for both (the book's approach).
// ---------------------------------------------------------------------------
struct Sphere {
    vec3  center0, center1;   // centre at shutter open / close
    float radius;
    int   mat;

    RT_HD vec3 center_at(float time) const {
        return center0 + time * (center1 - center0);
    }

    RT_HD aabb bounds() const {
        const vec3 rv(fabsf(radius), fabsf(radius), fabsf(radius));
        const aabb b0(center0 - rv, center0 + rv);
        const aabb b1(center1 - rv, center1 + rv);
        return aabb::surrounding_box(b0, b1);
    }
};

// ---------------------------------------------------------------------------
// Quad, defined by a corner and two edge vectors, exactly as the book does.
//
// Boxes and instanced geometry are both expressed as quads: a box is six of
// them, and translating or rotating a quad produces another quad, so the old
// translate/rotate_y ray-transform wrappers are applied to the corner and edge
// vectors once at build time instead of to every ray at trace time.
// ---------------------------------------------------------------------------
struct Quad {
    vec3  Q, u, v;
    vec3  normal;     // unit, already flipped if the scene asked for inward
    vec3  w;          // n / dot(n,n), for the interior test
    float D;          // plane constant, dot(normal, Q)
    int   mat;

    // Derived fields are computed once, here, rather than in a constructor --
    // the struct has to stay an aggregate to remain trivially copyable.
    RT_HD void finalize(bool inward) {
        const vec3 n = cross(u, v);
        normal = unit_vector(n);
        if (inward) normal = -normal;
        D = dot(normal, Q);
        w = n / dot(n, n);
    }

    RT_HD aabb bounds() const {
        const aabb d1(Q, Q + u + v);
        const aabb d2(Q + u, Q + v);
        return aabb::surrounding_box(d1, d2).pad(1e-3f);
    }
};

// ---------------------------------------------------------------------------
// Constant-density medium (smoke / fog).
//
// The boundary is stored inline rather than as a pointer to another hittable.
// That costs a little space and buys the ability to compute the entry and exit
// distances analytically in one step -- the book intersects the boundary twice,
// once from -infinity and once from just past the first hit, which needs an
// epsilon nudge that can skip thin boundaries.
// ---------------------------------------------------------------------------
enum MediumBound : int { MEDIUM_SPHERE = 0, MEDIUM_BOX = 1 };

struct Medium {
    int   bound;              // MediumBound
    vec3  center;             // sphere centre, or box centre
    float radius;             // sphere only
    vec3  half_extent;        // box only, in local space
    float sin_t, cos_t;       // box only: rotation about Y through `center`
    float neg_inv_density;    // -1 / density
    int   mat;                // the isotropic phase function

    RT_HD aabb bounds() const {
        if (bound == MEDIUM_SPHERE) {
            const vec3 rv(radius, radius, radius);
            return aabb(center - rv, center + rv);
        }
        // Rotated box: bound the eight rotated corners.
        vec3 lo( FLT_MAX,  FLT_MAX,  FLT_MAX);
        vec3 hi(-FLT_MAX, -FLT_MAX, -FLT_MAX);
        for (int i = 0; i < 8; ++i) {
            const float sx = (i & 1) ? 1.0f : -1.0f;
            const float sy = (i & 2) ? 1.0f : -1.0f;
            const float sz = (i & 4) ? 1.0f : -1.0f;
            const float x = sx * half_extent.x();
            const float y = sy * half_extent.y();
            const float z = sz * half_extent.z();
            const vec3 p(center.x() + cos_t * x + sin_t * z,
                         center.y() + y,
                         center.z() - sin_t * x + cos_t * z);
            lo = vec3(fminf(lo.x(), p.x()), fminf(lo.y(), p.y()), fminf(lo.z(), p.z()));
            hi = vec3(fmaxf(hi.x(), p.x()), fmaxf(hi.y(), p.y()), fmaxf(hi.z(), p.z()));
        }
        return aabb(lo, hi);
    }
};

// ---------------------------------------------------------------------------
// Intersection
// ---------------------------------------------------------------------------

RT_HD inline bool hit_sphere(const Sphere& s, const ray& r,
                             float t_min, float t_max, Hit& rec)
{
    const vec3 current_center = s.center_at(r.time());
    const vec3 oc = r.origin() - current_center;

    const float a = dot(r.direction(), r.direction());
    const float b = dot(oc, r.direction());               // half-b
    const float c = dot(oc, oc) - s.radius * s.radius;
    const float disc = b*b - a*c;
    if (disc <= 0.0f) return false;

    const float sq = sqrtf(disc);

    // Near root first; fall through to the far one when the near root is behind
    // t_min (which is how a ray starting inside the sphere still hits).
    float t = (-b - sq) / a;
    if (t <= t_min || t >= t_max) {
        t = (-b + sq) / a;
        if (t <= t_min || t >= t_max) return false;
    }

    rec.t      = t;
    rec.p      = r.point_at_parameter(t);
    // Dividing by the signed radius is what makes a negative radius invert the
    // normal -- the hollow-glass-bubble trick from the book.
    rec.normal = (rec.p - current_center) / s.radius;
    get_sphere_uv(rec.normal, rec.u, rec.v);
    rec.mat    = s.mat;
    return true;
}

RT_HD inline bool hit_quad(const Quad& q, const ray& r,
                           float t_min, float t_max, Hit& rec)
{
    const float denom = dot(q.normal, r.direction());
    if (fabsf(denom) < 1e-8f) return false;               // parallel to the plane

    const float t = (q.D - dot(q.normal, r.origin())) / denom;
    if (t < t_min || t > t_max) return false;

    const vec3 P  = r.point_at_parameter(t);
    const vec3 pl = P - q.Q;

    const float alpha = dot(q.w, cross(pl, q.v));
    const float beta  = dot(q.w, cross(q.u, pl));
    if (alpha < 0.0f || alpha > 1.0f || beta < 0.0f || beta > 1.0f) return false;

    rec.t = t;
    rec.p = P;
    rec.u = alpha;
    rec.v = beta;

    // Shading normal always opposes the incoming ray.
    vec3 n = q.normal;
    if (dot(n, r.direction()) > 0.0f) n = -n;
    rec.normal = n;

    rec.mat = q.mat;
    return true;
}

// Entry/exit distances along `r` for the medium's boundary. Returns false when
// the ray misses it entirely.
RT_HD inline bool medium_span(const Medium& m, const ray& r, float& t0, float& t1)
{
    if (m.bound == MEDIUM_SPHERE) {
        const vec3  oc = r.origin() - m.center;
        const float a = dot(r.direction(), r.direction());
        const float b = dot(oc, r.direction());
        const float c = dot(oc, oc) - m.radius * m.radius;
        const float disc = b*b - a*c;
        if (disc <= 0.0f) return false;
        const float sq = sqrtf(disc);
        t0 = (-b - sq) / a;
        t1 = (-b + sq) / a;
        return true;
    }

    // Box: rotate the ray into the box's local frame, then a plain slab test.
    const vec3 po = r.origin() - m.center;
    const vec3 lo(m.cos_t * po.x() - m.sin_t * po.z(), po.y(),
                  m.sin_t * po.x() + m.cos_t * po.z());
    const vec3 d = r.direction();
    const vec3 ld(m.cos_t * d.x() - m.sin_t * d.z(), d.y(),
                  m.sin_t * d.x() + m.cos_t * d.z());

    float tmin = -FLT_MAX, tmax = FLT_MAX;
    for (int axis = 0; axis < 3; ++axis) {
        const float inv = 1.0f / ld[axis];
        float a = (-m.half_extent[axis] - lo[axis]) * inv;
        float b = ( m.half_extent[axis] - lo[axis]) * inv;
        if (inv < 0.0f) { const float tmp = a; a = b; b = tmp; }
        tmin = a > tmin ? a : tmin;
        tmax = b < tmax ? b : tmax;
        if (tmax <= tmin) return false;
    }
    t0 = tmin;
    t1 = tmax;
    return true;
}

// `u01` is a uniform sample in [0,1), drawn by the caller. Passing it in rather
// than reaching for cuRAND is what keeps this function pure, dual-compiled and
// host-testable -- the randomness is the caller's problem.
RT_HD inline bool hit_medium(const Medium& m, const ray& r,
                             float t_min, float t_max, float u01, Hit& rec)
{
    float t0, t1;
    if (!medium_span(m, r, t0, t1)) return false;

    if (t0 < t_min) t0 = t_min;
    if (t1 > t_max) t1 = t_max;
    if (t0 >= t1)   return false;
    if (t0 < 0.0f)  t0 = 0.0f;

    const float ray_len = r.direction().length();
    if (!(ray_len > 0.0f)) return false;                  // also rejects NaN

    const float distance_inside = (t1 - t0) * ray_len;

    // Exponential free-flight sampling: d = -(1/sigma) * ln(U).
    const float U = fmaxf(1e-6f, u01);
    const float hit_distance = m.neg_inv_density * logf(U);
    if (hit_distance > distance_inside) return false;

    rec.t      = t0 + hit_distance / ray_len;
    rec.p      = r.point_at_parameter(rec.t);
    rec.normal = vec3(1, 0, 0);                           // arbitrary in a volume
    rec.u = rec.v = 0.0f;
    rec.mat    = m.mat;
    return true;
}
