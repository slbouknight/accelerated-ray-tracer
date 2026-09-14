// Host-side unit tests for the dual-compiled (__host__ __device__) math layer.
//
// Nothing in this file touches the GPU: it runs on a machine with no CUDA
// device present. That is the point -- every routine covered here is pure
// float math with no cuRAND state, no device heap allocation and no virtual
// dispatch, which is exactly the set of code that *should* be dual-compiled.
//
// The inverse is also informative: the small size of this file relative to
// test_device.cu is a direct measure of how much of the renderer is currently
// locked behind device-only polymorphism.

#include "test_harness.h"

#include "../src/aabb.cuh"
#include "../src/material.cuh"
#include "../src/perlin.cuh"
#include "../src/ray.cuh"
#include "../src/sphere.cuh"
#include "../src/vec3.cuh"

namespace {
constexpr float kEps = 1e-5f;
constexpr float kPi  = 3.14159265358979323846f;
} // namespace

// =============================================================================
// vec3
// =============================================================================

TEST(vec3, accessors_alias_rgb_and_xyz) {
    const vec3 v(1.0f, 2.0f, 3.0f);
    RT_CHECK_NEAR(v.x(), 1.0f, 0.0f);
    RT_CHECK_NEAR(v.y(), 2.0f, 0.0f);
    RT_CHECK_NEAR(v.z(), 3.0f, 0.0f);
    // r/g/b are aliases, not a separate storage path.
    RT_CHECK_NEAR(v.r(), v.x(), 0.0f);
    RT_CHECK_NEAR(v.g(), v.y(), 0.0f);
    RT_CHECK_NEAR(v.b(), v.z(), 0.0f);
    RT_CHECK_NEAR(v[0], 1.0f, 0.0f);
    RT_CHECK_NEAR(v[2], 3.0f, 0.0f);
}

TEST(vec3, arithmetic) {
    const vec3 a(1.0f, 2.0f, 3.0f);
    const vec3 b(4.0f, 5.0f, 6.0f);

    RT_CHECK_VEC(a + b, 5.0f, 7.0f, 9.0f, kEps);
    RT_CHECK_VEC(a - b, -3.0f, -3.0f, -3.0f, kEps);
    RT_CHECK_VEC(a * b, 4.0f, 10.0f, 18.0f, kEps);   // component-wise, not dot
    RT_CHECK_VEC(-a, -1.0f, -2.0f, -3.0f, kEps);
    RT_CHECK_VEC(2.0f * a, 2.0f, 4.0f, 6.0f, kEps);
    RT_CHECK_VEC(a * 2.0f, 2.0f, 4.0f, 6.0f, kEps);  // both operand orders exist
    RT_CHECK_VEC(a / 2.0f, 0.5f, 1.0f, 1.5f, kEps);
}

TEST(vec3, compound_assignment_mutates_in_place) {
    vec3 v(1.0f, 2.0f, 3.0f);
    v += vec3(1.0f, 1.0f, 1.0f);
    RT_CHECK_VEC(v, 2.0f, 3.0f, 4.0f, kEps);
    v -= vec3(1.0f, 1.0f, 1.0f);
    RT_CHECK_VEC(v, 1.0f, 2.0f, 3.0f, kEps);
    v *= 2.0f;
    RT_CHECK_VEC(v, 2.0f, 4.0f, 6.0f, kEps);
    v /= 2.0f;
    RT_CHECK_VEC(v, 1.0f, 2.0f, 3.0f, kEps);
    v *= vec3(2.0f, 3.0f, 4.0f);
    RT_CHECK_VEC(v, 2.0f, 6.0f, 12.0f, kEps);
    v /= vec3(2.0f, 3.0f, 4.0f);
    RT_CHECK_VEC(v, 1.0f, 2.0f, 3.0f, kEps);

    // The non-const operator[] returns a reference; render() writes gamma
    // through it, so it has to actually alias storage.
    v[1] = 99.0f;
    RT_CHECK_NEAR(v.y(), 99.0f, 0.0f);
}

TEST(vec3, dot_and_length) {
    RT_CHECK_NEAR(dot(vec3(1, 2, 3), vec3(4, 5, 6)), 32.0f, kEps);
    RT_CHECK_NEAR(dot(vec3(1, 0, 0), vec3(0, 1, 0)), 0.0f, kEps);
    RT_CHECK_NEAR(vec3(3, 4, 0).length(), 5.0f, kEps);
    RT_CHECK_NEAR(vec3(3, 4, 0).squared_length(), 25.0f, kEps);
}

TEST(vec3, cross_is_right_handed_and_orthogonal) {
    // x cross y == z fixes the handedness of the whole camera basis.
    RT_CHECK_VEC(cross(vec3(1, 0, 0), vec3(0, 1, 0)), 0.0f, 0.0f, 1.0f, kEps);
    RT_CHECK_VEC(cross(vec3(0, 1, 0), vec3(0, 0, 1)), 1.0f, 0.0f, 0.0f, kEps);
    RT_CHECK_VEC(cross(vec3(0, 0, 1), vec3(1, 0, 0)), 0.0f, 1.0f, 0.0f, kEps);

    const vec3 a(1.0f, 2.0f, 3.0f), b(-4.0f, 5.0f, 0.5f);
    const vec3 c = cross(a, b);
    RT_CHECK_NEAR(dot(c, a), 0.0f, 1e-4f);
    RT_CHECK_NEAR(dot(c, b), 0.0f, 1e-4f);
    RT_CHECK_VEC(cross(b, a), -c.x(), -c.y(), -c.z(), 1e-4f);  // anticommutative
}

TEST(vec3, unit_vector_normalizes) {
    const vec3 u = unit_vector(vec3(3.0f, 4.0f, 0.0f));
    RT_CHECK_NEAR(u.length(), 1.0f, kEps);
    RT_CHECK_VEC(u, 0.6f, 0.8f, 0.0f, kEps);

    vec3 m(0.0f, 0.0f, -7.0f);
    m.make_unit_vector();
    RT_CHECK_VEC(m, 0.0f, 0.0f, -1.0f, kEps);
}

// =============================================================================
// ray
// =============================================================================

TEST(ray, point_at_parameter) {
    const ray r(vec3(1, 2, 3), vec3(1, 0, 0), 0.0);
    RT_CHECK_VEC(r.point_at_parameter(0.0), 1.0f, 2.0f, 3.0f, kEps);
    RT_CHECK_VEC(r.point_at_parameter(5.0), 6.0f, 2.0f, 3.0f, kEps);
    RT_CHECK_VEC(r.point_at_parameter(-2.0), -1.0f, 2.0f, 3.0f, kEps);
}

TEST(ray, two_arg_ctor_defaults_time_to_zero) {
    // Scenes with a non-zero shutter rely on this; a garbage default time
    // would silently sample a moving sphere at the wrong position.
    const ray r(vec3(0, 0, 0), vec3(0, 0, 1));
    RT_CHECK_NEAR(r.time(), 0.0, 0.0);
}

TEST(ray, direction_is_not_normalized) {
    // sphere::hit divides by `a = dot(dir,dir)` precisely because direction is
    // kept unnormalized. Locking this in stops a future "optimization" from
    // normalizing at construction and breaking every t value in the scene.
    const ray r(vec3(0, 0, 0), vec3(0, 0, 4));
    RT_CHECK_NEAR(r.direction().length(), 4.0f, kEps);
}

// =============================================================================
// aabb
// =============================================================================

TEST(aabb, ctor_normalizes_corner_order) {
    const aabb box(vec3(5, 5, 5), vec3(-1, -2, -3));
    RT_CHECK_VEC(box.min(), -1.0f, -2.0f, -3.0f, kEps);
    RT_CHECK_VEC(box.max(), 5.0f, 5.0f, 5.0f, kEps);
}

TEST(aabb, default_ctor_is_empty_not_universe) {
    // A default box must be *inverted* so surrounding_box() can fold into it.
    const aabb box;
    RT_CHECK(box.min().x() > box.max().x());
}

TEST(aabb, surrounding_box_unions) {
    const aabb a(vec3(0, 0, 0), vec3(1, 1, 1));
    const aabb b(vec3(-2, 0.5f, 3), vec3(-1, 2, 4));
    const aabb u = aabb::surrounding_box(a, b);
    RT_CHECK_VEC(u.min(), -2.0f, 0.0f, 0.0f, kEps);
    RT_CHECK_VEC(u.max(), 1.0f, 2.0f, 4.0f, kEps);

    // Folding the empty box in must be the identity.
    const aabb id = aabb::surrounding_box(aabb(), a);
    RT_CHECK_VEC(id.min(), a.min().x(), a.min().y(), a.min().z(), kEps);
    RT_CHECK_VEC(id.max(), a.max().x(), a.max().y(), a.max().z(), kEps);
}

TEST(aabb, pad_and_translate) {
    const aabb p = aabb(vec3(0, 0, 0), vec3(1, 0, 1)).pad(0.5f);
    RT_CHECK_VEC(p.min(), -0.5f, -0.5f, -0.5f, kEps);
    RT_CHECK_VEC(p.max(), 1.5f, 0.5f, 1.5f, kEps);

    const aabb t = aabb(vec3(0, 0, 0), vec3(1, 1, 1)) + vec3(10, -5, 2);
    RT_CHECK_VEC(t.min(), 10.0f, -5.0f, 2.0f, kEps);
    RT_CHECK_VEC(t.max(), 11.0f, -4.0f, 3.0f, kEps);
}

TEST(aabb, slab_hit_and_miss) {
    const aabb box(vec3(-1, -1, -1), vec3(1, 1, 1));

    RT_CHECK(box.hit(ray(vec3(0, 0, -5), vec3(0, 0, 1), 0.0), 0.001f, FLT_MAX));
    RT_CHECK_FALSE(box.hit(ray(vec3(0, 5, -5), vec3(0, 0, 1), 0.0), 0.001f, FLT_MAX));

    // Pointing away: the intersection is entirely behind t_min.
    RT_CHECK_FALSE(box.hit(ray(vec3(0, 0, -5), vec3(0, 0, -1), 0.0), 0.001f, FLT_MAX));

    // A t_max that stops short of the box must reject it.
    RT_CHECK_FALSE(box.hit(ray(vec3(0, 0, -5), vec3(0, 0, 1), 0.0), 0.001f, 1.0f));
}

TEST(aabb, axis_aligned_ray_does_not_produce_a_false_miss) {
    // A direction component of exactly 0 gives invD = inf and then 0*inf = NaN
    // in the slab test. The `t0 > tmin ? t0 : tmin` form is deliberately
    // NaN-propagating-to-false, which makes the test *conservative* (may report
    // a hit that isn't one) rather than *wrong* (missing a real hit).
    // A refactor that swaps in fminf/fmaxf would silently invert this.
    const aabb box(vec3(-1, -1, -1), vec3(1, 1, 1));
    const ray along_x(vec3(-5, 0, 0), vec3(1, 0, 0), 0.0);
    RT_CHECK(box.hit(along_x, 0.001f, FLT_MAX));

    // Parallel to the slab and outside it: must still be rejected, because the
    // x and z slabs alone are enough to exclude it.
    const ray outside(vec3(-5, 9, 9), vec3(1, 0, 0), 0.0);
    RT_CHECK_FALSE(box.hit(outside, 0.001f, FLT_MAX));
}

// =============================================================================
// material math (reflect / refract / schlick)
// =============================================================================

TEST(material_math, reflect_about_normal) {
    // 45 degrees onto a +y plane -> mirrored y component only.
    RT_CHECK_VEC(reflect(vec3(1, -1, 0), vec3(0, 1, 0)), 1.0f, 1.0f, 0.0f, kEps);
    // Head-on reflection reverses the vector.
    RT_CHECK_VEC(reflect(vec3(0, -1, 0), vec3(0, 1, 0)), 0.0f, 1.0f, 0.0f, kEps);
    // Grazing (perpendicular to the normal) is unchanged.
    RT_CHECK_VEC(reflect(vec3(1, 0, 0), vec3(0, 1, 0)), 1.0f, 0.0f, 0.0f, kEps);
}

TEST(material_math, reflect_preserves_length) {
    const vec3 d(0.3f, -0.8f, 0.5f);
    const vec3 n = unit_vector(vec3(0.1f, 1.0f, -0.2f));
    RT_CHECK_NEAR(reflect(d, n).length(), d.length(), 1e-4f);
}

TEST(material_math, refract_bends_toward_normal_entering_denser_medium) {
    vec3 out;
    // ni_over_nt = 1/1.5: air -> glass. Note this refract() takes the *outward*
    // normal and the incoming direction, matching the book's sign convention.
    const bool ok = refract(unit_vector(vec3(1, -1, 0)), vec3(0, 1, 0), 1.0f / 1.5f, out);
    RT_REQUIRE(ok);
    RT_CHECK_NEAR(out.length(), 1.0f, 1e-4f);

    // Snell: sin(theta_t) = sin(theta_i)/1.5. Incident is 45deg, so the
    // transmitted ray must be closer to the -y axis than 45deg.
    const float sin_i = std::sqrt(0.5f);
    const float sin_t = std::fabs(out.x());        // normal is +y, so x is tangential
    RT_CHECK_NEAR(sin_t, sin_i / 1.5f, 1e-4f);
    RT_CHECK(out.y() < 0.0f);                      // still travelling downward
}

TEST(material_math, refract_reports_total_internal_reflection) {
    vec3 out;
    // Glass -> air (ni_over_nt = 1.5) past the critical angle (~41.8deg).
    // 60deg of incidence is well beyond it: refract() must return false so the
    // dielectric falls back to a pure reflection.
    const float s = std::sin(60.0f * kPi / 180.0f);
    const float c = std::cos(60.0f * kPi / 180.0f);
    RT_CHECK_FALSE(refract(vec3(s, -c, 0.0f), vec3(0, 1, 0), 1.5f, out));

    // Just inside the critical angle it must still transmit.
    const float s2 = std::sin(30.0f * kPi / 180.0f);
    const float c2 = std::cos(30.0f * kPi / 180.0f);
    RT_CHECK(refract(vec3(s2, -c2, 0.0f), vec3(0, 1, 0), 1.5f, out));
}

TEST(material_math, schlick_endpoints_and_monotonicity) {
    // Head-on (cosine = 1) reflectance for n=1.5 is r0 = (0.5/2.5)^2 = 0.04.
    RT_CHECK_NEAR(schlick(1.0f, 1.5f), 0.04f, 1e-4f);
    // Grazing incidence approaches total reflection.
    RT_CHECK_NEAR(schlick(0.0f, 1.5f), 1.0f, 1e-4f);

    float prev = schlick(0.0f, 1.5f);
    for (int i = 1; i <= 20; ++i) {
        const float v = schlick(i / 20.0f, 1.5f);
        RT_CHECK(v <= prev + 1e-6f);   // non-increasing in cosine
        RT_CHECK(v >= -1e-6f && v <= 1.0f + 1e-6f);
        prev = v;
    }
}

// =============================================================================
// sphere UV mapping (static, so host-callable without instantiating a sphere)
// =============================================================================

TEST(sphere_uv, known_points_on_the_unit_sphere) {
    double u = 0.0, v = 0.0;

    // -Z faces the default camera. phi = atan2(1,0)+pi = 3pi/2 -> u = 0.75
    sphere::get_sphere_uv(vec3(0, 0, -1), u, v);
    RT_CHECK_NEAR(u, 0.75, 1e-5);
    RT_CHECK_NEAR(v, 0.50, 1e-5);

    // +X: phi = atan2(0,1)+pi = pi -> u = 0.5
    sphere::get_sphere_uv(vec3(1, 0, 0), u, v);
    RT_CHECK_NEAR(u, 0.50, 1e-5);
    RT_CHECK_NEAR(v, 0.50, 1e-5);

    // Poles: v runs 0 at -Y to 1 at +Y.
    sphere::get_sphere_uv(vec3(0, -1, 0), u, v);
    RT_CHECK_NEAR(v, 0.0, 1e-5);
    sphere::get_sphere_uv(vec3(0, 1, 0), u, v);
    RT_CHECK_NEAR(v, 1.0, 1e-5);
}

TEST(sphere_uv, stays_in_unit_square_over_the_whole_sphere) {
    // Texture lookups index straight into the image with only a clamp01, so a
    // u/v escaping [0,1] is a sampling bug rather than a crash -- easy to miss
    // by eye, trivial to catch here.
    for (int i = 0; i < 64; ++i) {
        const float theta = kPi * (i + 0.5f) / 64.0f;
        for (int j = 0; j < 64; ++j) {
            const float phi = 2.0f * kPi * (j + 0.5f) / 64.0f;
            const vec3 p(std::sin(theta) * std::cos(phi),
                         std::cos(theta),
                         std::sin(theta) * std::sin(phi));
            double u = -1.0, v = -1.0;
            sphere::get_sphere_uv(p, u, v);
            if (!(u >= 0.0 && u <= 1.0 && v >= 0.0 && v <= 1.0)) {
                RT_FAIL("uv out of range at theta/phi index " + std::to_string(i)
                        + "/" + std::to_string(j));
                return;
            }
        }
    }
    RT_CHECK(true);
}

// =============================================================================
// perlin noise
// =============================================================================

TEST(perlin, noise_is_deterministic_and_bounded) {
    for (int i = 0; i < 200; ++i) {
        const vec3 p(i * 0.37f, i * -0.11f, i * 0.83f);
        const float a = perlin::noise(p);
        const float b = perlin::noise(p);
        RT_CHECK_NEAR(a, b, 0.0f);                 // bit-identical on repeat
        if (!(a >= -1.5f && a <= 1.5f)) {
            RT_FAIL("perlin::noise out of expected range: " + std::to_string(a));
            return;
        }
    }
}

TEST(perlin, noise_vanishes_at_lattice_points) {
    // Gradient noise is zero on the integer lattice by construction. If this
    // starts failing, the interpolation weights have drifted.
    for (int x = -2; x <= 2; ++x)
        for (int y = -2; y <= 2; ++y)
            RT_CHECK_NEAR(perlin::noise(vec3((float)x, (float)y, 1.0f)), 0.0f, 1e-5f);
}

TEST(perlin, turbulence_is_non_negative_and_converges) {
    const vec3 p(1.7f, -0.3f, 2.9f);
    for (int depth = 1; depth <= 7; ++depth) {
        const float t = perlin::turb(p, depth);
        RT_CHECK(t >= 0.0f);                       // turb() takes fabsf
    }
    // Octave weights halve, so depth 7 and depth 12 must be close.
    RT_CHECK_NEAR(perlin::turb(p, 7), perlin::turb(p, 12), 0.05f);
}

TEST(perlin, gradients_are_unit_length) {
    for (int i = 0; i < 50; ++i) {
        const vec3 g = perlin::grad(i, i * 3 - 7, i * -5 + 2);
        RT_CHECK_NEAR(g.length(), 1.0f, 1e-4f);
    }
}

// =============================================================================

int main(int argc, char** argv) {
    std::printf("=== host math tests (no GPU required) ===\n");
    return rt_test::run_all(argc, argv);
}
