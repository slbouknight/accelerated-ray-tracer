#pragma once

#include "cuda_compat.hpp"
#include "ray.hpp"
#include "shading_math.hpp"
#include "vec3.hpp"

// Thin-lens camera, as POD.
//
// Previously this was a device-only class built by a kernel into a
// `camera**` that every ray then had to dereference. It has no virtual
// functions and nothing device-specific about it, so there was never a reason
// for that: it is now configured on the host and passed to the kernel *by
// value*, which removes an allocation, a launch and a pointer chase per sample.
//
// The projection math is unchanged from the book.

struct Camera {
    vec3  origin;
    vec3  lower_left_corner;
    vec3  horizontal;
    vec3  vertical;
    vec3  u, v, w;
    float lens_radius = 0.0f;
    float time0 = 0.0f, time1 = 0.0f;   // shutter open / close

    // `s` and `t` are normalised film coordinates in [0,1].
    // Templated on the RNG so the same code runs under cuRAND on the device and
    // under a plain xorshift in the host tests.
    template <class Rng>
    RT_HD ray get_ray(float s, float t, Rng& rng) const {
        vec3 offset(0, 0, 0);
        if (lens_radius > 0.0f) {
            const vec3 rd = lens_radius * random_in_unit_disk(rng);
            offset = u * rd.x() + v * rd.y();
        }
        const float tm = time0 + rng.next() * (time1 - time0);
        return ray(origin + offset,
                   lower_left_corner + s * horizontal + t * vertical - origin - offset,
                   tm);
    }

    template <class Rng>
    RT_HD static vec3 random_in_unit_disk(Rng& rng) {
        // Rejection sampling, as the book does. Roughly 1.27 iterations
        // expected; on the GPU a warp loops until its last lane succeeds.
        for (;;) {
            const vec3 p = 2.0f * vec3(rng.next(), rng.next(), 0.0f) - vec3(1, 1, 0);
            if (dot(p, p) < 1.0f) return p;
        }
    }
};

// Host-side configuration, kept separate so the runtime struct stays a plain
// aggregate. Mirrors the book's camera parameters.
struct CameraSpec {
    vec3  lookfrom = vec3(0, 0, 0);
    vec3  lookat   = vec3(0, 0, -1);
    vec3  vup      = vec3(0, 1, 0);
    float vfov     = 40.0f;      // vertical field of view, degrees
    float aperture = 0.0f;
    float focus_dist = 10.0f;
    float time0 = 0.0f, time1 = 1.0f;

    RT_HD Camera build(float aspect) const {
        Camera c;
        c.lens_radius = aperture * 0.5f;
        c.time0 = time0;
        c.time1 = time1;

        const float theta = vfov * (rt::pi / 180.0f);
        const float half_height = tanf(theta * 0.5f);
        const float half_width  = aspect * half_height;

        c.origin = lookfrom;
        c.w = unit_vector(lookfrom - lookat);
        c.u = unit_vector(cross(vup, c.w));
        c.v = cross(c.w, c.u);

        c.lower_left_corner = c.origin
            - half_width  * focus_dist * c.u
            - half_height * focus_dist * c.v
            - focus_dist * c.w;

        c.horizontal = 2.0f * half_width  * focus_dist * c.u;
        c.vertical   = 2.0f * half_height * focus_dist * c.v;
        return c;
    }
};
