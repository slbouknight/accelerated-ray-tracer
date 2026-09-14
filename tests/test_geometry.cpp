// Host-side tests for the flat geometry, BVH construction and traversal.
//
// Before the refactor every one of these needed a GPU: the geometry was a
// __device__ virtual hierarchy built with device `new`, so a kernel launch was
// the only way to reach it. Now it is POD plus free functions, and the same
// code the renderer runs is exercised here by a plain C++ binary.

#include "test_harness.h"

#include "../src/core/camera.hpp"
#include "../src/core/primitives.hpp"
#include "../src/core/scene_view.hpp"
#include "../src/host/scene_builder.hpp"
#include "../src/host/scenes.hpp"

#include <cfloat>
#include <vector>

namespace {

constexpr float kEps = 1e-4f;

// Gives a SceneBuilder's arrays to a SceneView without uploading anything.
SceneView view_of(const rt::SceneBuilder& b) {
    SceneView v;
    v.spheres    = b.spheres().empty() ? nullptr : b.spheres().data();
    v.quads      = b.quads().empty()   ? nullptr : b.quads().data();
    v.media      = b.media().empty()   ? nullptr : b.media().data();
    v.refs       = b.refs().empty()    ? nullptr : b.refs().data();
    v.nodes      = b.nodes().empty()   ? nullptr : b.nodes().data();
    v.node_count = (int)b.nodes().size();
    return v;
}

// Always returns the same value, so tests that must not depend on randomness
// stay deterministic.
struct FixedRng {
    float value;
    explicit FixedRng(float v = 0.5f) : value(v) {}
    float next() { return value; }
};

} // namespace

// =============================================================================
// Sphere
// =============================================================================

TEST(sphere, hit_from_outside_takes_near_root) {
    const Sphere s{vec3(0, 0, 0), vec3(0, 0, 0), 1.0f, 7};
    Hit rec;
    RT_REQUIRE(hit_sphere(s, ray(vec3(0, 0, -5), vec3(0, 0, 1), 0.0f), 0.001f, FLT_MAX, rec));
    RT_CHECK_NEAR(rec.t, 4.0f, kEps);              // near root, not t=6
    RT_CHECK_VEC(rec.p, 0.0f, 0.0f, -1.0f, kEps);
    RT_CHECK_VEC(rec.normal, 0.0f, 0.0f, -1.0f, kEps);
    RT_CHECK_NEAR(rec.u, 0.75f, kEps);
    RT_CHECK_NEAR(rec.v, 0.50f, kEps);
    RT_CHECK_EQ(rec.mat, 7);                       // material id is carried through
}

TEST(sphere, rejections_and_hit_from_inside) {
    const Sphere s{vec3(0, 0, 0), vec3(0, 0, 0), 1.0f, 0};
    Hit rec;
    RT_CHECK_FALSE(hit_sphere(s, ray(vec3(0, 5, -5), vec3(0, 0, 1), 0.0f), 0.001f, FLT_MAX, rec));
    RT_CHECK_FALSE(hit_sphere(s, ray(vec3(0, 0, -5), vec3(0, 0, -1), 0.0f), 0.001f, FLT_MAX, rec));
    RT_CHECK_FALSE(hit_sphere(s, ray(vec3(0, 0, -5), vec3(0, 0, 1), 0.0f), 0.001f, 3.0f, rec));
    RT_CHECK_FALSE(hit_sphere(s, ray(vec3(0, 0, -5), vec3(0, 0, 1), 0.0f), 7.0f, FLT_MAX, rec));

    // From inside, the near root is behind t_min so the far root must be used.
    RT_REQUIRE(hit_sphere(s, ray(vec3(0, 0, 0), vec3(0, 0, 1), 0.0f), 0.001f, FLT_MAX, rec));
    RT_CHECK_NEAR(rec.t, 1.0f, kEps);
}

TEST(sphere, moving_centre_is_sampled_at_ray_time) {
    const Sphere s{vec3(0, 0, 0), vec3(4, 0, 0), 1.0f, 0};
    Hit rec;

    RT_REQUIRE(hit_sphere(s, ray(vec3(0, 0, -5), vec3(0, 0, 1), 0.0f), 0.001f, FLT_MAX, rec));
    RT_CHECK_VEC(rec.p, 0.0f, 0.0f, -1.0f, kEps);
    RT_REQUIRE(hit_sphere(s, ray(vec3(4, 0, -5), vec3(0, 0, 1), 1.0f), 0.001f, FLT_MAX, rec));
    RT_CHECK_VEC(rec.p, 4.0f, 0.0f, -1.0f, kEps);

    // The bbox has to span the whole swept path or the BVH culls the sphere out
    // of frames where it is genuinely visible.
    const aabb b = s.bounds();
    RT_CHECK_VEC(b.min(), -1.0f, -1.0f, -1.0f, kEps);
    RT_CHECK_VEC(b.max(),  5.0f,  1.0f,  1.0f, kEps);
}

TEST(sphere, negative_radius_inverts_the_normal) {
    // The hollow-glass-bubble trick: dividing by a signed radius flips the
    // normal without changing the geometry.
    const Sphere outer{vec3(0, 0, 0), vec3(0, 0, 0),  2.0f, 0};
    const Sphere inner{vec3(0, 0, 0), vec3(0, 0, 0), -2.0f, 0};
    Hit a, b;
    const ray r(vec3(0, 0, -5), vec3(0, 0, 1), 0.0f);
    RT_REQUIRE(hit_sphere(outer, r, 0.001f, FLT_MAX, a));
    RT_REQUIRE(hit_sphere(inner, r, 0.001f, FLT_MAX, b));
    RT_CHECK_NEAR(a.t, b.t, kEps);                       // same surface
    RT_CHECK_VEC(b.normal, -a.normal.x(), -a.normal.y(), -a.normal.z(), kEps);

    // bounds() must use |radius|, or the BVH box collapses.
    RT_CHECK_VEC(inner.bounds().min(), -2.0f, -2.0f, -2.0f, kEps);
    RT_CHECK_VEC(inner.bounds().max(),  2.0f,  2.0f,  2.0f, kEps);
}

// =============================================================================
// Quad
// =============================================================================

TEST(quad, hit_interior_miss_exterior_and_parallel) {
    Quad q{};
    q.Q = vec3(0, 0, 0); q.u = vec3(1, 0, 0); q.v = vec3(0, 1, 0); q.mat = 3;
    q.finalize(false);

    Hit rec;
    RT_REQUIRE(hit_quad(q, ray(vec3(0.5f, 0.5f, -3), vec3(0, 0, 1), 0.0f), 0.001f, FLT_MAX, rec));
    RT_CHECK_NEAR(rec.t, 3.0f, kEps);
    RT_CHECK_VEC(rec.p, 0.5f, 0.5f, 0.0f, kEps);
    RT_CHECK_VEC(rec.normal, 0.0f, 0.0f, -1.0f, kEps);   // flipped to face the ray
    RT_CHECK_NEAR(rec.u, 0.5f, kEps);
    RT_CHECK_NEAR(rec.v, 0.5f, kEps);
    RT_CHECK_EQ(rec.mat, 3);

    // Hits the plane but outside the (alpha,beta) unit square.
    RT_CHECK_FALSE(hit_quad(q, ray(vec3(1.5f, 0.5f, -3), vec3(0, 0, 1), 0.0f), 0.001f, FLT_MAX, rec));
    // Parallel to the plane: must not divide by zero.
    RT_CHECK_FALSE(hit_quad(q, ray(vec3(0.5f, 0.5f, -3), vec3(1, 0, 0), 0.0f), 0.001f, FLT_MAX, rec));
}

TEST(quad, inward_flag_flips_the_stored_normal) {
    Quad a{}, b{};
    a.Q = b.Q = vec3(0, 0, 0);
    a.u = b.u = vec3(1, 0, 0);
    a.v = b.v = vec3(0, 1, 0);
    a.finalize(false);
    b.finalize(true);
    RT_CHECK_VEC(b.normal, -a.normal.x(), -a.normal.y(), -a.normal.z(), kEps);
}

// =============================================================================
// Baked instancing
// =============================================================================

TEST(transform, rotate_y_90_maps_plus_x_to_minus_z) {
    const rt::Transform t = rt::Transform::rotate_y_degrees(90.0f);
    RT_CHECK_VEC(t.apply_point(vec3(3, 0, 0)), 0.0f, 0.0f, -3.0f, 1e-4f);
    RT_CHECK_VEC(t.apply_point(vec3(0, 0, 1)), 1.0f, 0.0f,  0.0f, 1e-4f);
    RT_CHECK_VEC(t.apply_point(vec3(0, 5, 0)), 0.0f, 5.0f,  0.0f, 1e-4f);
}

TEST(transform, translation_applies_after_rotation) {
    const rt::Transform t = rt::Transform::rotate_y_degrees(90.0f).then_translate(vec3(10, 0, 0));
    // Rotate (3,0,0) -> (0,0,-3), then translate -> (10,0,-3).
    RT_CHECK_VEC(t.apply_point(vec3(3, 0, 0)), 10.0f, 0.0f, -3.0f, 1e-4f);
    // Directions rotate but must not pick up the offset.
    RT_CHECK_VEC(t.apply_dir(vec3(3, 0, 0)), 0.0f, 0.0f, -3.0f, 1e-4f);
}

TEST(box, six_faces_and_closest_one_wins) {
    rt::SceneBuilder b;
    const int mat = b.materials().lambertian(vec3(1, 1, 1));
    b.add_box(vec3(-1, -1, -1), vec3(1, 1, 1), mat);
    b.build_bvh();
    RT_CHECK_EQ(b.quads().size(), 6);
    RT_CHECK_EQ(b.primitive_count(), 6);

    SceneView v = view_of(b);
    FixedRng rng;
    Hit rec;
    RT_REQUIRE(scene_intersect(v, ray(vec3(0, 0, -5), vec3(0, 0, 1), 0.0f), 0.001f, FLT_MAX, rec, rng));
    // Must report the near face at t=4, not the far one at t=6.
    RT_CHECK_NEAR(rec.t, 4.0f, kEps);
    RT_CHECK_VEC(rec.normal, 0.0f, 0.0f, -1.0f, kEps);
}

TEST(box, transform_is_baked_into_the_faces) {
    rt::SceneBuilder b;
    const int mat = b.materials().lambertian(vec3(1, 1, 1));
    // Unit cube at the origin, moved to x=10. A ray down +z at x=10 must hit it.
    b.add_box(vec3(-1, -1, -1), vec3(1, 1, 1), mat,
              rt::Transform{}.then_translate(vec3(10, 0, 0)));
    b.build_bvh();

    SceneView v = view_of(b);
    FixedRng rng;
    Hit rec;
    RT_REQUIRE(scene_intersect(v, ray(vec3(10, 0, -5), vec3(0, 0, 1), 0.0f), 0.001f, FLT_MAX, rec, rng));
    RT_CHECK_NEAR(rec.t, 4.0f, kEps);
    RT_CHECK_VEC(rec.p, 10.0f, 0.0f, -1.0f, kEps);
    // And nothing should remain at the original location.
    RT_CHECK_FALSE(scene_intersect(v, ray(vec3(0, 0, -5), vec3(0, 0, 1), 0.0f), 0.001f, FLT_MAX, rec, rng));
}

// =============================================================================
// Media
// =============================================================================

TEST(medium, sphere_span_is_entry_and_exit) {
    Medium m{};
    m.bound = MEDIUM_SPHERE;
    m.center = vec3(0, 0, 0);
    m.radius = 2.0f;
    m.neg_inv_density = -1.0f / 0.5f;
    m.mat = 1;

    float t0 = 0, t1 = 0;
    RT_REQUIRE(medium_span(m, ray(vec3(0, 0, -10), vec3(0, 0, 1), 0.0f), t0, t1));
    RT_CHECK_NEAR(t0, 8.0f, kEps);
    RT_CHECK_NEAR(t1, 12.0f, kEps);

    RT_CHECK_FALSE(medium_span(m, ray(vec3(0, 9, -10), vec3(0, 0, 1), 0.0f), t0, t1));
}

TEST(medium, box_span_is_entry_and_exit) {
    Medium m{};
    m.bound = MEDIUM_BOX;
    m.center = vec3(0, 0, 0);
    m.half_extent = vec3(1, 1, 1);
    m.sin_t = 0.0f; m.cos_t = 1.0f;       // no rotation
    m.neg_inv_density = -1.0f;
    m.mat = 1;

    float t0 = 0, t1 = 0;
    RT_REQUIRE(medium_span(m, ray(vec3(0, 0, -5), vec3(0, 0, 1), 0.0f), t0, t1));
    RT_CHECK_NEAR(t0, 4.0f, kEps);
    RT_CHECK_NEAR(t1, 6.0f, kEps);

    RT_CHECK_FALSE(medium_span(m, ray(vec3(0, 9, -5), vec3(0, 0, 1), 0.0f), t0, t1));
}

TEST(medium, rotated_box_span) {
    Medium m{};
    m.bound = MEDIUM_BOX;
    m.center = vec3(0, 0, 0);
    m.half_extent = vec3(1, 1, 1);
    // 45 degrees about Y: the cube's diagonal now faces -z, so a central ray
    // enters sqrt(2) from the centre instead of 1.
    m.sin_t = std::sin(45.0f * rt::pi / 180.0f);
    m.cos_t = std::cos(45.0f * rt::pi / 180.0f);
    m.neg_inv_density = -1.0f;
    m.mat = 1;

    float t0 = 0, t1 = 0;
    RT_REQUIRE(medium_span(m, ray(vec3(0, 0, -5), vec3(0, 0, 1), 0.0f), t0, t1));
    RT_CHECK_NEAR(t0, 5.0f - std::sqrt(2.0f), 1e-3f);
    RT_CHECK_NEAR(t1, 5.0f + std::sqrt(2.0f), 1e-3f);
}

TEST(medium, dense_medium_scatters_and_thin_one_usually_does_not) {
    Medium dense{};
    dense.bound = MEDIUM_BOX;
    dense.center = vec3(0, 0, 0);
    dense.half_extent = vec3(50, 50, 50);
    dense.sin_t = 0.0f; dense.cos_t = 1.0f;
    dense.neg_inv_density = -1.0f / 0.5f;     // mean free path 2, box is 100 deep
    dense.mat = 4;

    Hit rec;
    const ray r(vec3(0, 0, -200), vec3(0, 0, 1), 0.0f);
    RT_REQUIRE(hit_medium(dense, r, 0.001f, FLT_MAX, 0.5f, rec));
    RT_CHECK_EQ(rec.mat, 4);
    RT_CHECK(rec.t >= 150.0f);               // inside the box, which starts at t=150
    RT_CHECK(rec.t <= 250.0f);

    // Very low density: a free flight far longer than the box, so no scatter.
    Medium thin = dense;
    thin.neg_inv_density = -1.0f / 0.0001f;
    RT_CHECK_FALSE(hit_medium(thin, r, 0.001f, FLT_MAX, 0.5f, rec));

    // A ray that misses the boundary entirely never scatters.
    RT_CHECK_FALSE(hit_medium(dense, ray(vec3(0, 500, -200), vec3(0, 0, 1), 0.0f),
                              0.001f, FLT_MAX, 0.5f, rec));
}

TEST(medium, unnormalised_ray_direction_is_handled) {
    // The camera emits rays whose direction length is the focus distance, not 1.
    // distance_inside has to be measured in world units, not in t.
    Medium m{};
    m.bound = MEDIUM_BOX;
    m.center = vec3(0, 0, 0);
    m.half_extent = vec3(50, 50, 50);
    m.sin_t = 0.0f; m.cos_t = 1.0f;
    m.neg_inv_density = -1.0f / 0.5f;
    m.mat = 0;

    Hit a, b;
    const bool hit_unit = hit_medium(m, ray(vec3(0, 0, -200), vec3(0, 0, 1), 0.0f),
                                     0.001f, FLT_MAX, 0.5f, a);
    // Same ray, direction scaled by 800 (so t values shrink by 800x).
    const bool hit_scaled = hit_medium(m, ray(vec3(0, 0, -200), vec3(0, 0, 800), 0.0f),
                                       0.001f, FLT_MAX, 0.5f, b);
    RT_CHECK_EQ(hit_unit ? 1 : 0, hit_scaled ? 1 : 0);
    if (hit_unit && hit_scaled) {
        // Different t, but the same world-space point.
        RT_CHECK_VEC(b.p, a.p.x(), a.p.y(), a.p.z(), 1e-2f);
    }
}

TEST(medium, is_reachable_through_bvh_traversal) {
    // The regression that motivated this file: a medium that intersects
    // correctly in isolation but never gets tested during traversal.
    rt::SceneBuilder b;
    const int phase = b.materials().isotropic(vec3(0.5f, 0.5f, 0.5f));
    b.add_medium_box(vec3(0, 0, 0), vec3(100, 100, 100), 0.5f, phase);
    b.build_bvh();

    RT_CHECK_EQ(b.media().size(), 1);
    RT_CHECK_EQ(b.primitive_count(), 1);

    SceneView v = view_of(b);
    FixedRng rng(0.5f);
    Hit rec;
    RT_CHECK(scene_intersect(v, ray(vec3(50, 50, -200), vec3(0, 0, 1), 0.0f),
                             0.001f, FLT_MAX, rec, rng));
    RT_CHECK_EQ(rec.mat, phase);
}

// =============================================================================
// BVH
// =============================================================================

namespace {

// A deterministic pile of spheres with enough spread to make a real tree.
rt::SceneBuilder make_test_scene(int n) {
    rt::SceneBuilder b;
    const int mat = b.materials().lambertian(vec3(0.5f, 0.5f, 0.5f));
    XorShiftRng rng(12345u);
    for (int i = 0; i < n; ++i) {
        const vec3 c(rng.next() * 8.0f - 4.0f, rng.next() * 8.0f - 4.0f, rng.next() * 8.0f - 4.0f);
        b.add_sphere(c, 0.15f + 0.45f * rng.next(), mat);
    }
    b.build_bvh();
    return b;
}

ray probe_ray(int k) {
    XorShiftRng rng((unsigned int)(k * 2654435761u) + 1u);
    const vec3 dir(rng.next() * 2.0f - 1.0f, rng.next() * 2.0f - 1.0f, rng.next() * 2.0f - 1.0f);
    const vec3 origin = unit_vector(dir) * 15.0f;
    const vec3 target(rng.next() * 8.0f - 4.0f, rng.next() * 8.0f - 4.0f, rng.next() * 8.0f - 4.0f);
    return ray(origin, unit_vector(target - origin), 0.0f);
}

} // namespace

TEST(bvh, traversal_agrees_with_brute_force) {
    // The load-bearing test. A BVH is an acceleration structure: it is only
    // allowed to make the *same* answer arrive faster.
    const rt::SceneBuilder b = make_test_scene(96);
    const SceneView v = view_of(b);

    int mismatches = 0, hits = 0;
    for (int k = 0; k < 512; ++k) {
        const ray r = probe_ray(k);

        FixedRng rng_a, rng_b;
        Hit bf, tree;
        const bool hit_bf   = scene_intersect_bruteforce(v, r, 0.001f, FLT_MAX,
                                                         b.primitive_count(), bf, rng_a);
        const bool hit_tree = scene_intersect(v, r, 0.001f, FLT_MAX, tree, rng_b);

        if (hit_bf) ++hits;
        if (hit_bf != hit_tree) {
            if (++mismatches <= 5)
                RT_FAIL("ray " + std::to_string(k) + ": hit disagreement");
        } else if (hit_bf && !rt_test::near_rel(bf.t, tree.t, 1e-5)) {
            if (++mismatches <= 5)
                RT_FAIL("ray " + std::to_string(k) + ": t disagreement ("
                        + rt_test::fmt(bf.t) + " vs " + rt_test::fmt(tree.t) + ")");
        } else if (hit_bf) {
            if (bf.mat != tree.mat && ++mismatches <= 5)
                RT_FAIL("ray " + std::to_string(k) + ": material disagreement");
        }
    }
    RT_CHECK_EQ(mismatches, 0);
    RT_CHECK(hits > 128);          // guard against passing because everything missed
}

TEST(bvh, node_bounds_enclose_their_subtree) {
    const rt::SceneBuilder b = make_test_scene(64);
    const auto& nodes = b.nodes();
    RT_REQUIRE(!nodes.empty());

    int leaves = 0, interior = 0;
    for (size_t i = 0; i < nodes.size(); ++i) {
        const BvhNode& n = nodes[i];
        if (n.count == 0) {
            ++interior;
            RT_REQUIRE(n.left >= 0 && n.left < (int)nodes.size());
            RT_REQUIRE(n.right >= 0 && n.right < (int)nodes.size());
            // A parent box must contain both children, or geometry gets culled.
            for (int c : {n.left, n.right}) {
                const aabb& cb = nodes[c].bounds;
                const bool inside =
                    cb.min().x() >= n.bounds.min().x() - 1e-3f &&
                    cb.min().y() >= n.bounds.min().y() - 1e-3f &&
                    cb.min().z() >= n.bounds.min().z() - 1e-3f &&
                    cb.max().x() <= n.bounds.max().x() + 1e-3f &&
                    cb.max().y() <= n.bounds.max().y() + 1e-3f &&
                    cb.max().z() <= n.bounds.max().z() + 1e-3f;
                if (!inside) { RT_FAIL("child box escapes its parent at node " + std::to_string(i)); return; }
            }
        } else {
            ++leaves;
        }
    }
    RT_CHECK_EQ(leaves, 64);                       // one primitive per leaf
    RT_CHECK_EQ(interior, 63);                     // a binary tree with 64 leaves
    RT_CHECK_EQ((int)nodes.size(), 127);
}

TEST(bvh, empty_scene_is_not_a_crash) {
    rt::SceneBuilder b;
    b.build_bvh();
    SceneView v = view_of(b);
    FixedRng rng;
    Hit rec;
    RT_CHECK_FALSE(scene_intersect(v, ray(vec3(0, 0, 0), vec3(0, 0, 1), 0.0f), 0.001f, FLT_MAX, rec, rng));
}

TEST(bvh, single_primitive_scene) {
    rt::SceneBuilder b;
    const int mat = b.materials().lambertian(vec3(1, 1, 1));
    b.add_sphere(vec3(0, 0, 0), 1.0f, mat);
    b.build_bvh();
    RT_CHECK_EQ(b.nodes().size(), 1);

    SceneView v = view_of(b);
    FixedRng rng;
    Hit rec;
    RT_REQUIRE(scene_intersect(v, ray(vec3(0, 0, -5), vec3(0, 0, 1), 0.0f), 0.001f, FLT_MAX, rec, rng));
    RT_CHECK_NEAR(rec.t, 4.0f, kEps);
}

// =============================================================================
// Camera
// =============================================================================

TEST(camera, centre_ray_points_at_the_target) {
    CameraSpec spec;
    spec.lookfrom = vec3(0, 0, 5);
    spec.lookat   = vec3(0, 0, 0);
    spec.vfov     = 90.0f;
    spec.focus_dist = 5.0f;
    const Camera cam = spec.build(2.0f);

    XorShiftRng rng(1u);
    const ray r = cam.get_ray(0.5f, 0.5f, rng);
    RT_CHECK_VEC(unit_vector(r.direction()), 0.0f, 0.0f, -1.0f, 1e-4f);
    RT_CHECK_VEC(r.origin(), 0.0f, 0.0f, 5.0f, 1e-4f);   // aperture 0 -> no lens offset
    // aspect 2.0 must make the viewport twice as wide as it is tall.
    RT_CHECK_NEAR(cam.horizontal.length() / cam.vertical.length(), 2.0f, 1e-4f);
}

TEST(camera, shutter_time_stays_in_range) {
    CameraSpec spec;
    spec.lookfrom = vec3(0, 0, 5);
    spec.time0 = 0.25f;
    spec.time1 = 0.75f;
    const Camera cam = spec.build(1.0f);

    XorShiftRng rng(99u);
    for (int i = 0; i < 200; ++i) {
        const ray r = cam.get_ray(0.5f, 0.5f, rng);
        if (!(r.time() >= 0.25f && r.time() <= 0.75f)) {
            RT_FAIL("shutter time out of range: " + rt_test::fmt(r.time()));
            return;
        }
    }
    RT_CHECK(true);
}

TEST(camera, aperture_offsets_the_origin_within_the_lens) {
    CameraSpec spec;
    spec.lookfrom = vec3(0, 0, 5);
    spec.aperture = 2.0f;              // lens radius 1
    spec.focus_dist = 5.0f;
    const Camera cam = spec.build(1.0f);

    XorShiftRng rng(7u);
    for (int i = 0; i < 200; ++i) {
        const ray r = cam.get_ray(0.5f, 0.5f, rng);
        const float offset = (r.origin() - spec.lookfrom).length();
        if (offset > 1.0f + 1e-4f) {
            RT_FAIL("lens sample outside the aperture: " + rt_test::fmt(offset));
            return;
        }
    }
    RT_CHECK(true);
}

// =============================================================================
// Whole scenes
// =============================================================================

TEST(scenes, every_scene_builds_a_consistent_bvh) {
    for (int i = 0; i < rt::SCENE_COUNT; ++i) {
        const rt::SceneInfo& info = rt::scene_table()[i];
        rt::SceneSetup s;
        rt::build_scene(info.id, 1984, s);

        const int prims = s.world.primitive_count();
        if (prims <= 0) { RT_FAIL(std::string(info.name) + ": no primitives"); continue; }

        // A binary tree with one primitive per leaf has exactly 2n-1 nodes.
        if ((int)s.world.nodes().size() != 2 * prims - 1) {
            RT_FAIL(std::string(info.name) + ": expected " + std::to_string(2 * prims - 1)
                    + " nodes, got " + std::to_string(s.world.nodes().size()));
            continue;
        }

        // Every primitive must be referenced by exactly one leaf.
        std::vector<int> seen(prims, 0);
        for (const BvhNode& n : s.world.nodes())
            for (int k = 0; k < n.count; ++k) ++seen[n.first + k];
        for (int k = 0; k < prims; ++k) {
            if (seen[k] != 1) {
                RT_FAIL(std::string(info.name) + ": ref " + std::to_string(k)
                        + " referenced " + std::to_string(seen[k]) + " times");
                break;
            }
        }

        // Material ids must be in range, or the device will index off the table.
        const int n_mat = (int)s.world.materials().materials().size();
        for (const Sphere& sp : s.world.spheres())
            if (sp.mat < 0 || sp.mat >= n_mat) { RT_FAIL(std::string(info.name) + ": bad sphere material id"); break; }
        for (const Quad& q : s.world.quads())
            if (q.mat < 0 || q.mat >= n_mat) { RT_FAIL(std::string(info.name) + ": bad quad material id"); break; }
        for (const Medium& m : s.world.media())
            if (m.mat < 0 || m.mat >= n_mat) { RT_FAIL(std::string(info.name) + ": bad medium material id"); break; }

        RT_CHECK(true);
    }
}

TEST(scenes, cornell_smoke_media_are_hit_from_the_camera) {
    // Renders nothing, but fires the camera's own centre ray into the scene and
    // checks it reaches a volume -- the exact failure that slipped past the
    // "every scene builds" check.
    rt::SceneSetup s;
    rt::build_scene(rt::SCENE_CORNELL_SMOKE, 1984, s);
    const SceneView v = view_of(s.world);
    const Camera cam = s.cam.build(1.0f);

    // Which material ids belong to media?
    std::vector<int> medium_mats;
    for (const Medium& m : s.world.media()) medium_mats.push_back(m.mat);
    RT_REQUIRE(medium_mats.size() == 2);

    XorShiftRng rng(3u);
    int volume_hits = 0;
    for (int i = 0; i < 2000; ++i) {
        const float u = rng.next(), w = rng.next();
        const ray r = cam.get_ray(u, w, rng);
        Hit rec;
        if (scene_intersect(v, r, 0.001f, FLT_MAX, rec, rng)) {
            for (int m : medium_mats) if (rec.mat == m) { ++volume_hits; break; }
        }
    }
    RT_CHECK(volume_hits > 0);
}

// =============================================================================

int main(int argc, char** argv) {
    std::printf("=== geometry / BVH / scene tests (no GPU required) ===\n");
    return rt_test::run_all(argc, argv);
}
