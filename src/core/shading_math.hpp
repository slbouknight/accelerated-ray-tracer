#pragma once

#include "cuda_compat.hpp"
#include "vec3.hpp"

#include <cmath>

// Scattering math, split out of material.cuh.
//
// The material *classes* are a device-only virtual hierarchy, but the geometry
// they rely on is not: reflect, refract and schlick are closed-form float math
// with no RNG and no allocation. Keeping them here means they are unit-tested
// on the host, and it draws a visible line between "the physics" and "the
// dispatch mechanism" -- only the latter has to change in the refactor.

namespace rt {

inline constexpr float pi = 3.14159265358979323846f;

RT_HD inline float degrees_to_radians(float degrees) { return degrees * (pi / 180.0f); }

} // namespace rt

// Mirror v about the unit normal n.
RT_HD inline vec3 reflect(const vec3& v, const vec3& n)
{
    return v - 2.0f * dot(v, n) * n;
}

// Snell refraction in the book's sign convention: `n` is the outward normal and
// `ni_over_nt` is the ratio of refractive indices. Returns false on total
// internal reflection, in which case `refracted` is left untouched and the
// caller must reflect instead.
RT_HD inline bool refract(const vec3& v, const vec3& n, float ni_over_nt, vec3& refracted)
{
    vec3 uv = unit_vector(v);
    float dt = dot(uv, n);
    float disc = 1.0f - ni_over_nt*ni_over_nt*(1.0f - dt*dt);
    if (disc > 0.0f) {
        refracted = ni_over_nt * (uv - n*dt) - n * sqrtf(disc);
        return true;
    }
    return false;
}

// Schlick's polynomial approximation to the Fresnel reflectance. Monotonically
// decreasing in `cosine`, equal to r0 head-on and 1 at grazing incidence.
RT_HD inline float schlick(float cosine, float ref_idx)
{
    float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * powf(1.0f - cosine, 5.0f);
}

// Map a point on the unit sphere to texture coordinates.
//   u in [0,1) goes around the equator from -X, through -Z, +X, +Z
//   v in [0,1] goes from the -Y pole to the +Y pole
RT_HD inline void get_sphere_uv(const vec3& p, float& u, float& v)
{
    // acosf/atan2f rather than std::acos/std::atan2: the std overloads resolve
    // to the double versions, and this runs on every sphere hit.
    const float theta = acosf(-p.y());
    const float phi   = atan2f(-p.z(), p.x()) + rt::pi;

    u = phi / (2.0f * rt::pi);
    v = theta / rt::pi;
}
