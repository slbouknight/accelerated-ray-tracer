#ifndef CAMERAH
#define CAMERAH

#include <math_constants.h>
#include <curand_kernel.h>
#include "ray.h"

__device__ inline vec3 random_in_unit_disk(curandState *local_rand_state) {
    vec3 p;
    do {
        p = 2.0f * vec3(curand_uniform(local_rand_state),
                        curand_uniform(local_rand_state), 0.0f)
          - vec3(1.0f, 1.0f, 0.0f);
    } while (dot(p, p) >= 1.0f);
    return p;
}

class camera {
public:
    // Original constructor preserved: no motion blur (time0 = time1 = 0)
    __device__ camera(vec3 lookfrom, vec3 lookat, vec3 vup,
                      float vfov, float aspect, float aperture, float focus_dist)
        : time0(0.0), time1(0.0) {
        init(lookfrom, lookat, vup, vfov, aspect, aperture, focus_dist);
    }

    // Motion‑blur constructor: supply shutter open/close times
    __device__ camera(vec3 lookfrom, vec3 lookat, vec3 vup,
                      float vfov, float aspect, float aperture, float focus_dist,
                      double t0, double t1)
        : time0(t0), time1(t1) {
        init(lookfrom, lookat, vup, vfov, aspect, aperture, focus_dist);
    }

    __device__ ray get_ray(float s, float t, curandState *local_rand_state) const {
        vec3 rd = lens_radius * random_in_unit_disk(local_rand_state);
        vec3 offset = u * rd.x() + v * rd.y();

        // Uniformly sample a shutter time in [time0, time1]
        double tm = time0 + (double)curand_uniform(local_rand_state) * (time1 - time0);

        return ray(
            origin + offset,
            lower_left_corner + s*horizontal + t*vertical - origin - offset,
            tm
        );
    }

    // Public (POD-ish) members to match your existing style
    vec3 origin;
    vec3 lower_left_corner;
    vec3 horizontal;
    vec3 vertical;
    vec3 u, v, w;
    float lens_radius;
    double time0, time1; // shutter open/close

private:
    __device__ void init(vec3 lookfrom, vec3 lookat, vec3 vup,
                         float vfov, float aspect, float aperture, float focus_dist) {
        lens_radius = aperture * 0.5f;
        float theta = vfov * CUDART_PI_F / 180.0f;
        float half_height = tanf(theta * 0.5f);
        float half_width  = aspect * half_height;

        origin = lookfrom;
        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);

        lower_left_corner = origin
            - half_width * focus_dist * u
            - half_height * focus_dist * v
            - focus_dist * w;

        horizontal = 2.0f * half_width  * focus_dist * u;
        vertical   = 2.0f * half_height * focus_dist * v;
    }
};

#endif