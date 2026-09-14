// Device-side unit tests: geometry, traversal, materials and the camera.
//
// Everything under test here is locked behind __device__ virtual dispatch and
// device-heap allocation, so it cannot be exercised from the host. The pattern
// is therefore: a single-thread kernel computes, writes primitive floats into a
// managed buffer, and the host asserts on them with the same harness used by
// test_math.cu. Keeping all assertions host-side means failures print real
// values and line numbers instead of a device-side trap.
//
// After the flat/tagged-dispatch refactor most of these should migrate into
// test_math.cu and stop needing a GPU at all. Until then, the same expected
// values apply, which is what makes them a usable refactor harness.

#include "test_harness.h"

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cfloat>

#include "../src/bvh.cuh"
#include "../src/camera.cuh"
#include "../src/hittable.cuh"
#include "../src/material.cuh"
#include "../src/quad.cuh"
#include "../src/sphere.cuh"
#include "../src/texture.cuh"
#include "../src/util.cuh"
#include "../src/vec3.cuh"

namespace {

constexpr float kEps = 1e-4f;
constexpr float kMiss = -1.0f;   // sentinel written when a hit() returns false

// ---------------------------------------------------------------------------
// Managed scratch buffer. Prefilled with NaN so a kernel that silently fails to
// write a slot produces a failing assertion instead of reading a stale zero.
// ---------------------------------------------------------------------------
class Scratch {
public:
    explicit Scratch(int n) : n_(n) {
        if (cudaMallocManaged(&d_, sizeof(float) * n) != cudaSuccess) d_ = nullptr;
        if (d_) for (int i = 0; i < n; ++i) d_[i] = NAN;
    }
    ~Scratch() { if (d_) cudaFree(d_); }
    Scratch(const Scratch&) = delete;
    Scratch& operator=(const Scratch&) = delete;

    float*  data() const { return d_; }
    bool    ok()   const { return d_ != nullptr; }
    float   operator[](int i) const { return d_[i]; }
    // Read three consecutive slots as a vector, for RT_CHECK_VEC.
    vec3    v(int i) const { return vec3(d_[i], d_[i + 1], d_[i + 2]); }

private:
    float* d_ = nullptr;
    int    n_;
};

// Launch errors and device-side traps both surface here rather than as a
// confusing assertion failure three tests later.
bool device_ok(std::string* err) {
    cudaError_t e = cudaGetLastError();
    if (e == cudaSuccess) e = cudaDeviceSynchronize();
    if (e == cudaSuccess) return true;
    *err = std::string(cudaGetErrorName(e)) + ": " + cudaGetErrorString(e);
    return false;
}

#define RT_SYNC()                                                              \
    do {                                                                       \
        std::string rt_err_;                                                   \
        if (!device_ok(&rt_err_)) { RT_FAIL("CUDA: " + rt_err_); return; }     \
    } while (0)

} // namespace

// =============================================================================
// sphere
// =============================================================================

// out: [0]=hit [1]=t [2..4]=p [5..7]=normal [8]=u [9]=v
__global__ void k_sphere_basic(float* out) {
    lambertian mat(vec3(0.5f, 0.5f, 0.5f));
    sphere s(vec3(0, 0, 0), 1.0f, &mat, /*owns=*/false);

    hit_record rec;
    const ray r(vec3(0, 0, -5), vec3(0, 0, 1), 0.0);
    const bool h = s.hit(r, 0.001f, FLT_MAX, rec);

    out[0] = h ? 1.0f : 0.0f;
    if (!h) return;
    out[1] = rec.t;
    out[2] = rec.p.x();      out[3] = rec.p.y();      out[4] = rec.p.z();
    out[5] = rec.normal.x(); out[6] = rec.normal.y(); out[7] = rec.normal.z();
    out[8] = (float)rec.u;   out[9] = (float)rec.v;
}

TEST(sphere, hit_from_outside_takes_near_root) {
    Scratch s(10);
    RT_REQUIRE(s.ok());
    k_sphere_basic<<<1, 1>>>(s.data());
    RT_SYNC();

    RT_CHECK_NEAR(s[0], 1.0f, 0.0f);                  // hit
    RT_CHECK_NEAR(s[1], 4.0f, kEps);                  // near root, not t=6
    RT_CHECK_VEC(s.v(2), 0.0f, 0.0f, -1.0f, kEps);    // front of the sphere
    RT_CHECK_VEC(s.v(5), 0.0f, 0.0f, -1.0f, kEps);    // outward unit normal
    RT_CHECK_NEAR(s[8], 0.75f, 1e-4f);                // matches host sphere_uv test
    RT_CHECK_NEAR(s[9], 0.50f, 1e-4f);
}

// out: [0]=miss ray  [1]=tangent-ish miss  [2]=tmax too small  [3]=tmin too large
//      [4]=hit from inside  [5]=t from inside
__global__ void k_sphere_rejects(float* out) {
    lambertian mat(vec3(0.5f, 0.5f, 0.5f));
    sphere s(vec3(0, 0, 0), 1.0f, &mat, false);
    hit_record rec;

    out[0] = s.hit(ray(vec3(0, 5, -5), vec3(0, 0, 1), 0.0), 0.001f, FLT_MAX, rec) ? 1.f : 0.f;
    out[1] = s.hit(ray(vec3(0, 0, -5), vec3(0, 0, -1), 0.0), 0.001f, FLT_MAX, rec) ? 1.f : 0.f;
    out[2] = s.hit(ray(vec3(0, 0, -5), vec3(0, 0, 1), 0.0), 0.001f, 3.0f, rec) ? 1.f : 0.f;
    out[3] = s.hit(ray(vec3(0, 0, -5), vec3(0, 0, 1), 0.0), 7.0f, FLT_MAX, rec) ? 1.f : 0.f;

    // Origin inside the sphere: the near root is negative, so hit() must fall
    // through to the far root rather than reporting a miss.
    const bool inside = s.hit(ray(vec3(0, 0, 0), vec3(0, 0, 1), 0.0), 0.001f, FLT_MAX, rec);
    out[4] = inside ? 1.f : 0.f;
    out[5] = inside ? rec.t : kMiss;
}

TEST(sphere, rejection_cases_and_hit_from_inside) {
    Scratch s(6);
    RT_REQUIRE(s.ok());
    k_sphere_rejects<<<1, 1>>>(s.data());
    RT_SYNC();

    RT_CHECK_NEAR(s[0], 0.0f, 0.0f);      // passes 5 units above the sphere
    RT_CHECK_NEAR(s[1], 0.0f, 0.0f);      // pointing away
    RT_CHECK_NEAR(s[2], 0.0f, 0.0f);      // t_max clips before t=4
    RT_CHECK_NEAR(s[3], 0.0f, 0.0f);      // t_min clips past t=6
    RT_CHECK_NEAR(s[4], 1.0f, 0.0f);
    RT_CHECK_NEAR(s[5], 1.0f, kEps);      // far root
}

// out: [0..2]=hit p at time 0   [3..5]=hit p at time 1   [6..8]=bbox min  [9..11]=bbox max
__global__ void k_sphere_moving(float* out) {
    lambertian mat(vec3(0.5f, 0.5f, 0.5f));
    // Centre travels from (0,0,0) to (4,0,0) across the shutter interval.
    sphere s(vec3(0, 0, 0), vec3(4, 0, 0), 1.0f, &mat);

    hit_record rec;
    if (s.hit(ray(vec3(0, 0, -5), vec3(0, 0, 1), 0.0), 0.001f, FLT_MAX, rec)) {
        out[0] = rec.p.x(); out[1] = rec.p.y(); out[2] = rec.p.z();
    }
    if (s.hit(ray(vec3(4, 0, -5), vec3(0, 0, 1), 1.0), 0.001f, FLT_MAX, rec)) {
        out[3] = rec.p.x(); out[4] = rec.p.y(); out[5] = rec.p.z();
    }
    const aabb b = s.bounding_box();
    out[6] = b.min().x();  out[7] = b.min().y();  out[8]  = b.min().z();
    out[9] = b.max().x();  out[10] = b.max().y(); out[11] = b.max().z();
}

TEST(sphere, moving_centre_is_sampled_at_ray_time) {
    Scratch s(12);
    RT_REQUIRE(s.ok());
    k_sphere_moving<<<1, 1>>>(s.data());
    RT_SYNC();

    RT_CHECK_VEC(s.v(0), 0.0f, 0.0f, -1.0f, kEps);   // shutter open: centre at x=0
    RT_CHECK_VEC(s.v(3), 4.0f, 0.0f, -1.0f, kEps);   // shutter close: centre at x=4

    // The motion-blur bbox must span the whole swept path, or the BVH will cull
    // the sphere out of frames where it is genuinely visible.
    RT_CHECK_VEC(s.v(6), -1.0f, -1.0f, -1.0f, kEps);
    RT_CHECK_VEC(s.v(9), 5.0f, 1.0f, 1.0f, kEps);
}

// out: [0..2]=bbox min  [3..5]=bbox max  [6]=hit  [7]=t
__global__ void k_sphere_negative_radius(float* out) {
    lambertian mat(vec3(0.5f, 0.5f, 0.5f));
    // The hollow-glass-bubble trick from the Cornell scene: a negative radius
    // inverts the surface normal without changing the geometry.
    sphere s(vec3(10, 0, 0), -2.0f, &mat, false);

    const aabb b = s.bounding_box();
    out[0] = b.min().x(); out[1] = b.min().y(); out[2] = b.min().z();
    out[3] = b.max().x(); out[4] = b.max().y(); out[5] = b.max().z();

    hit_record rec;
    const bool h = s.hit(ray(vec3(10, 0, -5), vec3(0, 0, 1), 0.0), 0.001f, FLT_MAX, rec);
    out[6] = h ? 1.f : 0.f;
    out[7] = h ? rec.t : kMiss;
}

TEST(sphere, negative_radius_still_produces_a_valid_bbox) {
    Scratch s(8);
    RT_REQUIRE(s.ok());
    k_sphere_negative_radius<<<1, 1>>>(s.data());
    RT_SYNC();

    // sphere's ctor computes aabb(cen - rvec, cen + rvec) with a negative rvec,
    // which would invert the box -- except aabb's ctor re-normalizes via
    // fminf/fmaxf. Correct, but only by way of the aabb ctor, so it is worth
    // pinning: an aabb "optimization" that drops the min/max would break this.
    RT_CHECK_VEC(s.v(0), 8.0f, -2.0f, -2.0f, kEps);
    RT_CHECK_VEC(s.v(3), 12.0f, 2.0f, 2.0f, kEps);
    RT_CHECK_NEAR(s[6], 1.0f, 0.0f);
    RT_CHECK_NEAR(s[7], 3.0f, kEps);
}

// =============================================================================
// quad
// =============================================================================

// out: [0]=hit centre [1]=t [2..4]=p [5..7]=normal [8]=u [9]=v
//      [10]=miss outside  [11]=miss parallel  [12..14]=inward normal
__global__ void k_quad(float* out) {
    lambertian mat(vec3(0.5f, 0.5f, 0.5f));
    // Unit quad in the z=0 plane spanning x,y in [0,1].
    quad q(vec3(0, 0, 0), vec3(1, 0, 0), vec3(0, 1, 0), &mat, /*inward=*/false, /*owns=*/false);

    hit_record rec;
    const bool h = q.hit(ray(vec3(0.5f, 0.5f, -3), vec3(0, 0, 1), 0.0), 0.001f, FLT_MAX, rec);
    out[0] = h ? 1.f : 0.f;
    if (h) {
        out[1] = rec.t;
        out[2] = rec.p.x();      out[3] = rec.p.y();      out[4] = rec.p.z();
        out[5] = rec.normal.x(); out[6] = rec.normal.y(); out[7] = rec.normal.z();
        out[8] = (float)rec.u;   out[9] = (float)rec.v;
    }

    // Outside the (alpha,beta) unit square: the plane is hit but the quad isn't.
    out[10] = q.hit(ray(vec3(1.5f, 0.5f, -3), vec3(0, 0, 1), 0.0), 0.001f, FLT_MAX, rec) ? 1.f : 0.f;
    // Travelling parallel to the plane: denom ~ 0, must not divide by zero.
    out[11] = q.hit(ray(vec3(0.5f, 0.5f, -3), vec3(1, 0, 0), 0.0), 0.001f, FLT_MAX, rec) ? 1.f : 0.f;

    // inward=true flips the stored geometric normal (Cornell walls rely on it).
    quad qi(vec3(0, 0, 0), vec3(1, 0, 0), vec3(0, 1, 0), &mat, /*inward=*/true, false);
    out[12] = qi.normal.x(); out[13] = qi.normal.y(); out[14] = qi.normal.z();
}

TEST(quad, hit_interior_miss_exterior_and_parallel) {
    Scratch s(15);
    RT_REQUIRE(s.ok());
    k_quad<<<1, 1>>>(s.data());
    RT_SYNC();

    RT_CHECK_NEAR(s[0], 1.0f, 0.0f);
    RT_CHECK_NEAR(s[1], 3.0f, kEps);
    RT_CHECK_VEC(s.v(2), 0.5f, 0.5f, 0.0f, kEps);
    // cross(u,v) = +z, but the shading normal is flipped to oppose the ray.
    RT_CHECK_VEC(s.v(5), 0.0f, 0.0f, -1.0f, kEps);
    RT_CHECK_NEAR(s[8], 0.5f, kEps);       // alpha
    RT_CHECK_NEAR(s[9], 0.5f, kEps);       // beta

    RT_CHECK_NEAR(s[10], 0.0f, 0.0f);
    RT_CHECK_NEAR(s[11], 0.0f, 0.0f);
    RT_CHECK_VEC(s.v(12), 0.0f, 0.0f, -1.0f, kEps);
}

// out: [0]=hit [1]=t [2..4]=p [5..7]=normal  [8..10]=bbox min  [11..13]=bbox max
__global__ void k_box(float* out) {
    lambertian mat(vec3(0.5f, 0.5f, 0.5f));
    hittable* b = make_box(vec3(-1, -1, -1), vec3(1, 1, 1), &mat);

    hit_record rec;
    const bool h = b->hit(ray(vec3(0, 0, -5), vec3(0, 0, 1), 0.0), 0.001f, FLT_MAX, rec);
    out[0] = h ? 1.f : 0.f;
    if (h) {
        out[1] = rec.t;
        out[2] = rec.p.x();      out[3] = rec.p.y();      out[4] = rec.p.z();
        out[5] = rec.normal.x(); out[6] = rec.normal.y(); out[7] = rec.normal.z();
    }
    const aabb bb = b->bounding_box();
    out[8]  = bb.min().x(); out[9]  = bb.min().y(); out[10] = bb.min().z();
    out[11] = bb.max().x(); out[12] = bb.max().y(); out[13] = bb.max().z();

    delete b;
}

TEST(box, closest_face_wins) {
    Scratch s(14);
    RT_REQUIRE(s.ok());
    k_box<<<1, 1>>>(s.data());
    RT_SYNC();

    RT_CHECK_NEAR(s[0], 1.0f, 0.0f);
    // compound6 scans all six faces; it must report the -z face at t=4, not the
    // +z face at t=6 that it also intersects.
    RT_CHECK_NEAR(s[1], 4.0f, kEps);
    RT_CHECK_VEC(s.v(2), 0.0f, 0.0f, -1.0f, kEps);
    RT_CHECK_VEC(s.v(5), 0.0f, 0.0f, -1.0f, kEps);

    // quad::set_bounding_box pads by 1e-3 to avoid zero-thickness slabs.
    RT_CHECK_NEAR(s[8],  -1.0f, 2e-3f);
    RT_CHECK_NEAR(s[11],  1.0f, 2e-3f);
}

// =============================================================================
// instancing (translate / rotate_y)
// =============================================================================

// out: [0]=hit [1]=t [2..4]=p   then rotate: [5]=hit [6]=t [7..9]=p
__global__ void k_instances(float* out) {
    lambertian mat(vec3(0.5f, 0.5f, 0.5f));
    hit_record rec;

    // Sphere at the origin, translated to (10,0,0). A ray aimed at x=10 must
    // hit, and the returned point must be back in *world* space.
    sphere* base = new sphere(vec3(0, 0, 0), 1.0f, &mat, false);
    translate* t = new translate(base, vec3(10, 0, 0));
    const bool h1 = t->hit(ray(vec3(10, 0, -5), vec3(0, 0, 1), 0.0), 0.001f, FLT_MAX, rec);
    out[0] = h1 ? 1.f : 0.f;
    if (h1) { out[1] = rec.t; out[2] = rec.p.x(); out[3] = rec.p.y(); out[4] = rec.p.z(); }

    // A unit sphere offset to +x=3, rotated 90 degrees about Y, lands on the
    // -z axis (R_y(90) maps +x to -z under this convention).
    sphere* off = new sphere(vec3(3, 0, 0), 1.0f, &mat, false);
    rotate_y* rot = new rotate_y(off, 90.0f);
    const bool h2 = rot->hit(ray(vec3(0, 0, -8), vec3(0, 0, 1), 0.0), 0.001f, FLT_MAX, rec);
    out[5] = h2 ? 1.f : 0.f;
    if (h2) { out[6] = rec.t; out[7] = rec.p.x(); out[8] = rec.p.y(); out[9] = rec.p.z(); }

    // translate/rotate_y do not own their children, so free explicitly.
    delete t; delete base;
    delete rot; delete off;
}

TEST(instancing, translate_and_rotate_y_return_world_space_hits) {
    Scratch s(10);
    RT_REQUIRE(s.ok());
    k_instances<<<1, 1>>>(s.data());
    RT_SYNC();

    RT_CHECK_NEAR(s[0], 1.0f, 0.0f);
    RT_CHECK_NEAR(s[1], 4.0f, kEps);
    RT_CHECK_VEC(s.v(2), 10.0f, 0.0f, -1.0f, kEps);   // world space, not (0,0,-1)

    RT_CHECK_NEAR(s[5], 1.0f, 0.0f);
    RT_CHECK_NEAR(s[6], 4.0f, kEps);
    RT_CHECK_VEC(s.v(7), 0.0f, 0.0f, -4.0f, kEps);
}

// =============================================================================
// BVH -- the refactor harness
// =============================================================================

namespace {
constexpr int kBvhObjects = 96;
constexpr int kBvhRays    = 256;
} // namespace

// Probe rays are generated from a shared helper so the brute-force pass and the
// BVH pass cannot drift apart. Each starts well outside the scene and is aimed
// at a random point inside it, which keeps the hit rate high enough that the
// comparison is actually exercising traversal.
__device__ inline ray probe_ray(int k) {
    const vec3 dir_seed = random_in_unit_cube(k + 100000) * 2.0f - vec3(1, 1, 1);
    const vec3 origin   = unit_vector(dir_seed) * 15.0f;
    const vec3 target   = random_in_unit_cube(k + 200000) * 8.0f - vec3(4, 4, 4);
    return ray(origin, unit_vector(target - origin), 0.0);
}

// For each ray, writes the brute-force closest t and the BVH closest t.
// out[2k] = linear scan, out[2k+1] = BVH. kMiss when nothing was hit.
__global__ void k_bvh_vs_bruteforce(float* out, int nobj, int nray) {
    lambertian* mat = new lambertian(vec3(0.5f, 0.5f, 0.5f));

    hittable** list = (hittable**)malloc(sizeof(hittable*) * nobj);
    hittable** flat = (hittable**)malloc(sizeof(hittable*) * nobj);
    if (!list || !flat) { out[0] = -999.0f; return; }

    for (int i = 0; i < nobj; ++i) {
        // Deterministic scatter in [-4,4]^3 with varied radii, so the tree has
        // a real branching structure rather than a degenerate chain.
        const vec3 u = random_in_unit_cube(i);
        const vec3 c = u * 8.0f - vec3(4, 4, 4);
        const float rad = 0.15f + 0.45f * random_in_unit_cube(i + 7919).x();
        list[i] = new sphere(c, rad, mat, /*owns=*/false);
        flat[i] = list[i];   // keep an unpermuted copy; the BVH sorts in place
    }

    // Reference answer first, from a plain linear scan over every object.
    for (int k = 0; k < nray; ++k) {
        const ray r = probe_ray(k);

        float closest = FLT_MAX;
        bool  any = false;
        hit_record rec;
        for (int i = 0; i < nobj; ++i) {
            if (flat[i]->hit(r, 0.001f, closest, rec)) { any = true; closest = rec.t; }
        }
        out[2 * k] = any ? closest : kMiss;
    }

    bvh_node* root = new bvh_node(list, 0, nobj);

    for (int k = 0; k < nray; ++k) {
        hit_record rec;
        out[2 * k + 1] = root->hit(probe_ray(k), 0.001f, FLT_MAX, rec) ? rec.t : kMiss;
    }

    delete root;                                  // internal nodes only
    for (int i = 0; i < nobj; ++i) delete flat[i];
    delete mat;
    free(list); free(flat);
}

TEST(bvh, traversal_agrees_with_brute_force) {
    // This is the single most valuable test in the suite. A BVH is an
    // acceleration structure: it is only allowed to make the *same* answer
    // arrive faster. Any rewrite -- iterative traversal, SAH splits, flattened
    // nodes, host-side construction -- has to keep this green.
    Scratch s(2 * kBvhRays);
    RT_REQUIRE(s.ok());
    k_bvh_vs_bruteforce<<<1, 1>>>(s.data(), kBvhObjects, kBvhRays);
    RT_SYNC();

    RT_REQUIRE(s[0] > -900.0f);   // device malloc succeeded

    int mismatches = 0, hits = 0;
    for (int k = 0; k < kBvhRays; ++k) {
        const float bf = s[2 * k], bvh = s[2 * k + 1];
        const bool bf_hit = bf > 0.0f, bvh_hit = bvh > 0.0f;
        if (bf_hit) ++hits;

        if (bf_hit != bvh_hit) {
            if (++mismatches <= 5) {
                RT_FAIL("ray " + std::to_string(k) + ": hit disagreement (linear="
                        + (bf_hit ? "hit" : "miss") + ", bvh="
                        + (bvh_hit ? "hit" : "miss") + ")");
            }
        } else if (bf_hit && !rt_test::near_rel(bf, bvh, 1e-5)) {
            if (++mismatches <= 5) {
                RT_FAIL("ray " + std::to_string(k) + ": t disagreement (linear="
                        + rt_test::fmt(bf) + ", bvh=" + rt_test::fmt(bvh) + ")");
            }
        }
    }

    RT_CHECK_EQ(mismatches, 0);
    // Guard against the test passing trivially because every ray missed.
    RT_CHECK(hits > kBvhRays / 8);
}

// out: [0..2] = bbox min, [3..5] = bbox max, [6] = objects enclosed
__global__ void k_bvh_bbox(float* out, int nobj) {
    lambertian* mat = new lambertian(vec3(0.5f, 0.5f, 0.5f));
    hittable** list = (hittable**)malloc(sizeof(hittable*) * nobj);
    if (!list) { out[6] = -999.0f; return; }

    aabb expect;
    for (int i = 0; i < nobj; ++i) {
        const vec3 c = random_in_unit_cube(i) * 8.0f - vec3(4, 4, 4);
        const float rad = 0.15f + 0.45f * random_in_unit_cube(i + 7919).x();
        list[i] = new sphere(c, rad, mat, false);
        expect = aabb::surrounding_box(expect, list[i]->bounding_box());
    }

    bvh_node* root = new bvh_node(list, 0, nobj);
    const aabb got = root->bounding_box();

    // Report the *difference* from the true union so the host sees zeros on success.
    out[0] = got.min().x() - expect.min().x();
    out[1] = got.min().y() - expect.min().y();
    out[2] = got.min().z() - expect.min().z();
    out[3] = got.max().x() - expect.max().x();
    out[4] = got.max().y() - expect.max().y();
    out[5] = got.max().z() - expect.max().z();
    out[6] = (float)nobj;

    delete root;
    for (int i = 0; i < nobj; ++i) delete list[i];
    delete mat;
    free(list);
}

TEST(bvh, root_box_is_exactly_the_union_of_its_leaves) {
    Scratch s(7);
    RT_REQUIRE(s.ok());
    k_bvh_bbox<<<1, 1>>>(s.data(), kBvhObjects);
    RT_SYNC();

    RT_REQUIRE(s[6] > 0.0f);
    // A root box that is too small silently culls geometry; too large only
    // costs traversal time. Demand exactness -- it is cheap to maintain.
    for (int i = 0; i < 6; ++i) RT_CHECK_NEAR(s[i], 0.0f, 1e-5f);
}

// =============================================================================
// materials
// =============================================================================

// out: [0]=scattered? [1..3]=dir [4..6]=attenuation [7]=dielectric scattered?
//      [8..10]=dielectric attenuation  [11]=light scatters?  [12..14]=emitted
__global__ void k_materials(float* out) {
    curandState rng;
    curand_init(7u, 0, 0, &rng);

    hit_record rec;
    rec.p = vec3(0, 0, 0);
    rec.normal = vec3(0, 1, 0);
    rec.u = rec.v = 0.0;

    // Zero-fuzz metal is a perfect mirror: a 45-degree incoming ray leaves at
    // 45 degrees with the tangential component preserved.
    metal m(vec3(0.8f, 0.6f, 0.2f), 0.0f);
    ray scattered; vec3 atten;
    const ray incoming(vec3(-1, 1, 0), unit_vector(vec3(1, -1, 0)), 0.0);
    out[0] = m.scatter(incoming, rec, atten, scattered, &rng) ? 1.f : 0.f;
    const vec3 d = unit_vector(scattered.direction());
    out[1] = d.x(); out[2] = d.y(); out[3] = d.z();
    out[4] = atten.x(); out[5] = atten.y(); out[6] = atten.z();

    // Glass never absorbs: attenuation is always white whichever branch it takes.
    dielectric g(1.5f);
    vec3 gatten;
    out[7] = g.scatter(incoming, rec, gatten, scattered, &rng) ? 1.f : 0.f;
    out[8] = gatten.x(); out[9] = gatten.y(); out[10] = gatten.z();

    // A light emits but does not scatter -- that is what terminates the path.
    diffuse_light L(vec3(4, 5, 6));
    vec3 latten;
    out[11] = L.scatter(incoming, rec, latten, scattered, &rng) ? 1.f : 0.f;
    const vec3 e = L.emitted(0.f, 0.f, vec3(0, 0, 0));
    out[12] = e.x(); out[13] = e.y(); out[14] = e.z();
}

TEST(material, metal_mirrors_dielectric_is_white_light_terminates) {
    Scratch s(15);
    RT_REQUIRE(s.ok());
    k_materials<<<1, 1>>>(s.data());
    RT_SYNC();

    RT_CHECK_NEAR(s[0], 1.0f, 0.0f);
    RT_CHECK_VEC(s.v(1), 0.70710678f, 0.70710678f, 0.0f, 1e-4f);
    RT_CHECK_VEC(s.v(4), 0.8f, 0.6f, 0.2f, kEps);

    RT_CHECK_NEAR(s[7], 1.0f, 0.0f);
    RT_CHECK_VEC(s.v(8), 1.0f, 1.0f, 1.0f, kEps);

    RT_CHECK_NEAR(s[11], 0.0f, 0.0f);
    RT_CHECK_VEC(s.v(12), 4.0f, 5.0f, 6.0f, kEps);
}

// out: [0..2]=even cell, [3..5]=odd cell, [6..8]=solid, [9..11]=uv_offset wrap
__global__ void k_textures(float* out) {
    // scale 1.0 -> one cell per unit, so parity flips every integer step.
    checker_texture chk(1.0f, new solid_color(vec3(1, 0, 0)),
                              new solid_color(vec3(0, 0, 1)));
    const vec3 a = chk.value(0.f, 0.f, vec3(0.5f, 0.5f, 0.5f));   // (0,0,0) -> even
    const vec3 b = chk.value(0.f, 0.f, vec3(1.5f, 0.5f, 0.5f));   // (1,0,0) -> odd
    out[0] = a.x(); out[1] = a.y(); out[2] = a.z();
    out[3] = b.x(); out[4] = b.y(); out[5] = b.z();

    solid_color sc(vec3(0.1f, 0.2f, 0.3f));
    const vec3 c = sc.value(0.9f, 0.4f, vec3(100, 200, 300));     // ignores u,v,p
    out[6] = c.x(); out[7] = c.y(); out[8] = c.z();

    // u + 0.75 must wrap into [0,1) rather than running off the texture.
    solid_color* probe = new solid_color(vec3(0, 0, 0));
    uv_offset_texture off(probe, 0.75f);
    const vec3 e = off.value(0.5f, 0.5f, vec3(0, 0, 0));
    out[9] = e.x(); out[10] = e.y(); out[11] = e.z();
    delete probe;
}

TEST(texture, checker_parity_and_solid_color) {
    Scratch s(12);
    RT_REQUIRE(s.ok());
    k_textures<<<1, 1>>>(s.data());
    RT_SYNC();

    RT_CHECK_VEC(s.v(0), 1.0f, 0.0f, 0.0f, kEps);   // even -> first texture
    RT_CHECK_VEC(s.v(3), 0.0f, 0.0f, 1.0f, kEps);   // odd  -> second texture
    RT_CHECK_VEC(s.v(6), 0.1f, 0.2f, 0.3f, kEps);
    RT_CHECK_VEC(s.v(9), 0.0f, 0.0f, 0.0f, kEps);   // wrapped lookup still valid
}

// =============================================================================
// camera
// =============================================================================

// out: [0..2]=centre ray dir (unit)  [3..5]=origin  [6]=horiz/vert length ratio
//      [7]=time within shutter?  [8]=lens radius
__global__ void k_camera(float* out) {
    curandState rng;
    curand_init(11u, 0, 0, &rng);

    const vec3 lookfrom(0, 0, 5), lookat(0, 0, 0), vup(0, 1, 0);
    camera cam(lookfrom, lookat, vup, 90.0f, 2.0f, /*aperture=*/0.0f,
               /*focus_dist=*/5.0f, /*t0=*/0.25, /*t1=*/0.75);

    const ray r = cam.get_ray(0.5f, 0.5f, &rng);
    const vec3 d = unit_vector(r.direction());
    out[0] = d.x(); out[1] = d.y(); out[2] = d.z();
    out[3] = r.origin().x(); out[4] = r.origin().y(); out[5] = r.origin().z();

    // aspect = 2.0 must make the viewport exactly twice as wide as it is tall.
    out[6] = cam.horizontal.length() / cam.vertical.length();
    out[7] = (r.time() >= 0.25 && r.time() <= 0.75) ? 1.f : 0.f;
    out[8] = cam.lens_radius;
}

TEST(camera, centre_ray_points_at_the_target_and_respects_aspect) {
    Scratch s(9);
    RT_REQUIRE(s.ok());
    k_camera<<<1, 1>>>(s.data());
    RT_SYNC();

    // Looking down -z from (0,0,5) at the origin.
    RT_CHECK_VEC(s.v(0), 0.0f, 0.0f, -1.0f, 1e-4f);
    RT_CHECK_VEC(s.v(3), 0.0f, 0.0f, 5.0f, 1e-4f);   // aperture 0 -> no lens offset
    RT_CHECK_NEAR(s[6], 2.0f, 1e-4f);
    RT_CHECK_NEAR(s[7], 1.0f, 0.0f);                 // shutter time in range
    RT_CHECK_NEAR(s[8], 0.0f, 0.0f);
}

// =============================================================================

int main(int argc, char** argv) {
    std::printf("=== device tests ===\n");

    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::printf("no CUDA device available -- skipping\n");
        return 77;   // CTest treats 77 as "skipped"
    }

    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    std::printf("device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);

    // The BVH is still built recursively on the device, so the traversal tests
    // need the same stack and heap headroom the real scenes ask for. Removing
    // these two lines is one of the concrete goals of the refactor.
    cudaDeviceSetLimit(cudaLimitStackSize,      16384);
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 64 * 1024 * 1024);

    const int rc = rt_test::run_all(argc, argv);
    cudaDeviceReset();
    return rc;
}
