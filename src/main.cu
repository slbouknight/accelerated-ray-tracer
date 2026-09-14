#define STB_IMAGE_IMPLEMENTATION
#include <curand_kernel.h>
#include <float.h>
#include <math.h>
#include <math_constants.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "bvh.cuh"
#include "camera.cuh"
#include "constant_medium.cuh"
#include "hittable_list.cuh"
#include "image_io.h"
#include "material.cuh"
#include "perlin.cuh"
#include "ray.cuh"
#include "quad.cuh"
#include "sphere.cuh"
#include "texture.cuh"
#include "util.cuh"
#include "vec3.cuh"

#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line)
{
    if (result)
    {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << "at " <<
        file << ":" << line << " '" << func << "' \n";

        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

__device__ inline float apply_gamma(float c, float gamma)
{
    if (gamma == 1.0f) return c;
    float inv = 1.0f / gamma;
    return powf(fmaxf(c, 0.0f), inv);
}

__device__ vec3 color(const ray& r0,
                      const vec3& background,
                      bool gradient_bg,
                      int max_depth,
                      hittable **world,
                      curandState *local_rand_state)
{
    ray  cur_ray        = r0;
    vec3 throughput     = vec3(1,1,1);
    vec3 radiance       = vec3(0,0,0);

    for (int bounce = 0; bounce < max_depth; ++bounce)
    {
        hit_record rec;
        if (!(*world)->hit(cur_ray, 0.001f, FLT_MAX, rec, local_rand_state)) {
            // miss: add background
            vec3 bg = background;
            if (gradient_bg) 
            {
                vec3 unit_direction = unit_vector(cur_ray.direction());
                float t = 0.5f*(unit_direction.y() + 1.0f);
                bg = (1.0f - t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
            }
            radiance += throughput * bg;
            break;
        }

        // add emission at this hit
        radiance += throughput * rec.mat_ptr->emitted(rec.u, rec.v, rec.p);

        // scatter
        ray  scattered;
        vec3 attenuation;
        if (!rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) 
        {
            // light or absorbing surface: we’re done
            break;
        }

        throughput *= attenuation;
        cur_ray = scattered;
    }

    return radiance;
}

__global__ void rand_init(curandState *rand_state, unsigned long long seed)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curand_init(seed, 0, 0, rand_state);
    }
}

__global__ void render_init(int max_x, int max_y, curandState *rand_state,
                            unsigned long long seed)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j*max_x + i;

    // Distinct seed per pixel rather than distinct *sequence*. curand_init with
    // a sequence number does a 2^67 skip-ahead, which is far more expensive;
    // varying the seed is the pragmatic choice the NVIDIA blog made too. The
    // streams are only statistically independent, not provably so.
    curand_init(seed + pixel_index, 0, 0, &rand_state[pixel_index]);
}

// Accumulates `ns` samples into `accum` *without* normalising or gamma-encoding.
// Splitting accumulation from resolve is what lets the host issue the sample
// budget in batches: several short launches instead of one multi-minute kernel
// that Windows' display watchdog (TDR, 2s by default) would kill.
__global__ void render_accumulate(vec3 *accum, int max_x, int max_y, int ns,
                                  int max_depth, camera **cam, hittable **world,
                                  curandState *rand_state,
                                  vec3 background, int use_gradient_bg)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;

    int pixel_index = j*max_x + i;
    curandState local_rand_state = rand_state[pixel_index];

    vec3 col(0,0,0);
    for (int s = 0; s < ns; s++)
    {
        float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
        ray r = (*cam)->get_ray(u, v, &local_rand_state);
        col += color(r, background, use_gradient_bg != 0, max_depth, world, &local_rand_state);
    }
    rand_state[pixel_index] = local_rand_state;

    accum[pixel_index] += col;
}

// Normalise by the total sample count and gamma-encode, once, at the end.
__global__ void resolve(vec3 *fb, const vec3 *accum, int max_x, int max_y,
                        int total_samples, float gamma)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;

    int pixel_index = j*max_x + i;
    vec3 col = accum[pixel_index] / float(total_samples);
    col[0] = apply_gamma(col[0], gamma);
    col[1] = apply_gamma(col[1], gamma);
    col[2] = apply_gamma(col[2], gamma);
    fb[pixel_index] = col;
}

__global__ void clear_buffer(vec3 *buf, int n)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) buf[i] = vec3(0,0,0);
}

// ----- Configurable scene parameters -----
// Random helper (same as you had)
#define RND (curand_uniform(&local_rand_state))

// Grid config from the book example
#define GRID_MIN   -11
#define GRID_MAX    11
#define GRID_SIZE   (GRID_MAX - GRID_MIN)     // 22
#define TOTAL_SMALL (GRID_SIZE * GRID_SIZE)   // 22*22 = 484

// 1 ground + 3 big spheres + all small spheres
#define NUM_OBJECTS (1 + 3 + TOTAL_SMALL)

// UT palette + picker
__device__ inline vec3 pick_ut_color(float r) {
    const vec3 UT_ORANGE = vec3(1.0f, 0.51f, 0.0f);  // #FF8200
    const vec3 WHITE     = vec3(1.0f, 1.0f, 1.0f);
    const vec3 GRAY      = vec3(0.60f, 0.60f, 0.60f);
    const vec3 BLACK     = vec3(0.0f, 0.0f, 0.0f);
    if      (r < 0.25f) return WHITE;
    else if (r < 0.50f) return UT_ORANGE;
    else if (r < 0.75f) return GRAY;
    else                return BLACK;
}

__global__ void create_world_bouncing(hittable **d_list, hittable **d_world, camera **d_camera,
                             int nx, int ny, curandState *rand_state, int *d_count)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandState local_rand_state = *rand_state;
        int i = 0;

        // UT orange (hex #FF8200 ≈ sRGB 1.0, 0.51, 0.0)
        const vec3 UT_ORANGE = vec3(1.0f, 0.51f, 0.0f);

        // Ground (checkered) — larger tiles + UT colors
        texture* checker = new checker_texture(
            0.64f,                                  // lower = larger squares (0.04 for even bigger)
            new solid_color(vec3(1.0f, 1.0f, 1.0f)),// white
            new solid_color(UT_ORANGE)              // orange
        );

        d_list[i++] = new sphere(vec3(0.0f, -1000.0f, -1.0f), 1000.0f,
                                new lambertian(checker));

        // Tunables
        const float P_EMISSIVE = 0.10f;         // ~10% of diffuse spheres emit
        const float EMIT_POWER = 4.0f;          // brightness multiplier for emitters

        // Random small spheres on a grid
        for (int a = GRID_MIN; a < GRID_MAX; a++) {
            for (int b = GRID_MIN; b < GRID_MAX; b++) {
                float choose_mat = RND;
                vec3 center(a + 0.9f * RND, 0.2f, b + 0.9f * RND);

                if (choose_mat < 0.8f) {
                    // ---- Diffuse (MOVING) ----
                    // random velocity in each axis
                    vec3 vel(0.0f, 0.5f * RND, 0.25f * (RND - 0.5f));
                    vec3 center2 = center + vel;

                    // Some are emissive UT orange only
                    if (RND < P_EMISSIVE) {
                        vec3 ut_orange = vec3(1.0f, 0.51f, 0.0f);
                        d_list[i++] = new sphere(center, center2, 0.2f,
                                                new diffuse_light(EMIT_POWER * ut_orange));
                    } else {
                        vec3 albedo = pick_ut_color(RND);
                        d_list[i++] = new sphere(center, center2, 0.2f,
                                                new lambertian(albedo));
                    }

                } else if (choose_mat < 0.95f) {
                    // ---- Metal (STATIC) ----
                    vec3 albedo = pick_ut_color(RND);
                    // avoid totally black metal (looks like a void); nudge to dark gray if chosen
                    if (albedo.x() + albedo.y() + albedo.z() < 1e-5f)
                        albedo = vec3(0.15f, 0.15f, 0.15f);

                    float fuzz = 0.5f * RND; // or clamp lower for shinier metals
                    d_list[i++] = new sphere(center, 0.2f, new metal(albedo, fuzz));

                } else {
                    // ---- Dielectric (STATIC) ----
                    d_list[i++] = new sphere(center, 0.2f, new dielectric(1.5f));
                }
            }
        }

        // Three big spheres (static)
        d_list[i++] = new sphere(vec3( 0.0f, 1.0f,  0.0f), 1.0f, new dielectric(1.5f));
        d_list[i++] = new sphere(vec3(-4.0f, 1.0f,  0.0f), 1.0f, new lambertian(vec3(0.4f, 0.2f, 0.1f)));
        d_list[i++] = new sphere(vec3( 4.0f, 1.0f,  0.0f), 1.0f, new metal(vec3(0.7f, 0.6f, 0.5f), 0.0f));

        *rand_state = local_rand_state;
        // Report how many leaves were actually written. The host allocates
        // capacity, but only this many slots are initialised -- freeing past
        // here would delete uninitialised device memory.
        *d_count = i;
        *d_world = new bvh_node(d_list, 0, i);

        // Camera — add shutter times [0,1]
        vec3 lookfrom(13.0f, 2.0f, 3.0f);
        vec3 lookat (0.0f,  0.0f, 0.0f);
        vec3 vup(0.0f, 1.0f, 0.0f);
        float dist_to_focus = (lookfrom - lookat).length();
        float aperture = 0.1f;

        *d_camera = new camera(lookfrom, lookat, vup,
                               30.0f, float(nx)/float(ny),
                               aperture, dist_to_focus,
                               /*time0=*/0.0, /*time1=*/1.0);
    }
}

__global__ void create_world_checker(hittable **d_list, hittable **d_world, camera **d_camera,
                                     int nx, int ny, curandState *rand_state, int *d_count)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // (RNG not strictly needed here, but keep the pattern)
        curandState local_rand_state = *rand_state;
        int i = 0;

        // One shared checker texture + lambertian
        texture* checker = new checker_texture(0.32f,
            new solid_color(vec3(0.2f, 0.3f, 0.1f)),
            new solid_color(vec3(0.9f, 0.9f, 0.9f)));
        material* lam = new lambertian(checker);

        // Two big spheres (y = ±10), like the book’s “checkered_spheres”
      // NOTE: owns=false. This material is shared by several primitives and
        // each ~sphere/~quad would otherwise delete it, so the first teardown
        // frees it and the rest double-free. Ownership moves to a flat,
        // host-side material table in the refactor; until then it leaks once
        // at exit, which is strictly better than corrupting the device heap.
        d_list[i++] = new sphere(vec3(0,-10,0), 10.0f, lam, /*owns=*/false);
        d_list[i++] = new sphere(vec3(0, 10,0), 10.0f, lam, /*owns=*/false);

        // Report how many leaves were actually written. The host allocates
        // capacity, but only this many slots are initialised -- freeing past
        // here would delete uninitialised device memory.
        *d_count = i;
        *d_world = new bvh_node(d_list, 0, i);

        // Camera (pinhole)
        vec3 lookfrom(13.0f, 2.0f, 3.0f);
        vec3 lookat (0.0f, 0.0f, 0.0f);
        vec3 vup(0.0f, 1.0f, 0.0f);
        float dist_to_focus = 10.0f;
        float aperture = 0.0f;

        *d_camera = new camera(lookfrom, lookat, vup,
                               20.0f, float(nx)/float(ny),
                               aperture, dist_to_focus,
                               0.0, 1.0);

        *rand_state = local_rand_state;
    }
}

__global__ void create_world_earth(hittable **d_list, hittable **d_world, camera **d_camera,
                                   int nx, int ny, DeviceImage earth_img, int *d_count)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int i = 0;

        // Textured earth sphere at the origin
        texture*  earth_tex  = new image_texture(earth_img);
        material* earth_lam  = new lambertian(earth_tex);
        d_list[i++] = new sphere(vec3(0,0,0), 2.0f, earth_lam);

        // Wrap in BVH (okay even for 1 object, matches your other scenes)
        // Report how many leaves were actually written. The host allocates
        // capacity, but only this many slots are initialised -- freeing past
        // here would delete uninitialised device memory.
        *d_count = i;
        *d_world = new bvh_node(d_list, 0, i);

        // Camera (pinhole)
        vec3 lookfrom(0.0f, 0.0f, 12.0f);
        vec3 lookat  (0.0f, 0.0f,  0.0f);
        vec3 vup     (0.0f, 1.0f,  0.0f);
        float dist_to_focus = 12.0f;
        float aperture      = 0.0f;

        *d_camera = new camera(lookfrom, lookat, vup,
                               20.0f, float(nx)/float(ny),
                               aperture, dist_to_focus,
                               0.0, 1.0);
    }
}

__global__ void create_world_perlin(hittable **d_list, hittable **d_world, camera **d_camera,
                                    int nx, int ny, float scale, int *d_count)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int i = 0;

        texture* pertext = new noise_texture(scale);
        material* lam    = new lambertian(pertext);

      // NOTE: owns=false. This material is shared by several primitives and
        // each ~sphere/~quad would otherwise delete it, so the first teardown
        // frees it and the rest double-free. Ownership moves to a flat,
        // host-side material table in the refactor; until then it leaks once
        // at exit, which is strictly better than corrupting the device heap.
        d_list[i++] = new sphere(vec3(0,-1000,0), 1000.f, lam, /*owns=*/false);
        d_list[i++] = new sphere(vec3(0,     2,0),    2.f, lam, /*owns=*/false);

        // Report how many leaves were actually written. The host allocates
        // capacity, but only this many slots are initialised -- freeing past
        // here would delete uninitialised device memory.
        *d_count = i;
        *d_world = new bvh_node(d_list, 0, i);

        vec3 lookfrom(13,2,3), lookat(0,0,0), vup(0,1,0);
        *d_camera = new camera(lookfrom, lookat, vup,
                               20.0f, float(nx)/float(ny),
                               0.0f, 10.0f, 0.0, 1.0);
    }
}

__global__ void create_world_quads(hittable **d_list, hittable **d_world, camera **d_camera,
                                   int nx, int ny, int *d_count)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int i = 0;

        // Materials
        material* left_red     = new lambertian(vec3(1.0f, 0.2f, 0.2f));
        material* back_green   = new lambertian(vec3(0.2f, 1.0f, 0.2f));
        material* right_blue   = new lambertian(vec3(0.2f, 0.2f, 1.0f));
        material* upper_orange = new lambertian(vec3(1.0f, 0.5f, 0.0f));
        material* lower_teal   = new lambertian(vec3(0.2f, 0.8f, 0.8f));

        // Quads (same geometry as your serial version)
        d_list[i++] = new quad(vec3(-3,-2, 5), vec3(0, 0,-4), vec3(0, 4, 0), left_red);
        d_list[i++] = new quad(vec3(-2,-2, 0), vec3(4, 0, 0), vec3(0, 4, 0), back_green);
        d_list[i++] = new quad(vec3( 3,-2, 1), vec3(0, 0, 4), vec3(0, 4, 0), right_blue);
        d_list[i++] = new quad(vec3(-2, 3, 1), vec3(4, 0, 0), vec3(0, 0, 4), upper_orange);
        d_list[i++] = new quad(vec3(-2,-3, 5), vec3(4, 0, 0), vec3(0, 0,-4), lower_teal);

        // Report how many leaves were actually written. The host allocates
        // capacity, but only this many slots are initialised -- freeing past
        // here would delete uninitialised device memory.
        *d_count = i;
        *d_world = new bvh_node(d_list, 0, i);

        vec3 lookfrom(0,0,9), lookat(0,0,0), vup(0,1,0);
        *d_camera = new camera(lookfrom, lookat, vup,
                               80.0f, float(nx)/float(ny),
                               0.0f, 10.0f, 0.0, 1.0);
    }
}

__global__ void create_world_simple_light(hittable **d_list, hittable **d_world, camera **d_camera,
                                          int nx, int ny, DeviceImage ball_img, int *d_count)
{
    if (threadIdx.x || blockIdx.x) return;
    int i = 0;

    // --- Ground: your felt material from before ---
    texture* felttex = new felt_texture(vec3(0.06f, 0.36f, 0.18f), 16.0f, 0.08f, 4.0f, 0.03f);
    material* feltlam = new lambertian(felttex);
    d_list[i++] = new sphere(vec3(0,-1000,0), 1000.f, feltlam);

    // --- Pool ball core: image texture + UV rotation ---
    texture* base_img = new image_texture(ball_img);              // from your DeviceImage
    float u_rot_turns = 60.0f/360.0f;                             // rotate decal ~30° toward camera
    texture* ball_tex = new uv_offset_texture(base_img, u_rot_turns);
    material* ball_diffuse = new lambertian(ball_tex);

    const vec3 C(0,2,0);
    const float R = 2.0f;
    d_list[i++] = new sphere(C, R, ball_diffuse);                 // colored core

    // --- Clear-coat lacquer: thin dielectric shell ---
    material* clearcoat = new dielectric(1.5f);                   // glass-like lacquer
    d_list[i++] = new sphere(C, R + 0.02f, clearcoat);            // thin outer coat

    // --- Lights (unchanged) ---
    material* light1 = new diffuse_light(vec3(4,4,4));
    material* light2 = new diffuse_light(vec3(4,4,4));
    d_list[i++] = new sphere(vec3(0,7,0), 2.f,  light1);
    d_list[i++] = new quad  (vec3(3,1,-2), vec3(2,0,0), vec3(0,2,0), light2);

    // Report how many leaves were actually written. The host allocates
    // capacity, but only this many slots are initialised -- freeing past
    // here would delete uninitialised device memory.
    *d_count = i;
    *d_world = new bvh_node(d_list, 0, i);

    // Camera
    vec3 lookfrom(26,3,6), lookat(0,2,0), vup(0,1,0);
    float dist_to_focus = (lookfrom - lookat).length();
    *d_camera = new camera(lookfrom, lookat, vup,
                           20.0f, float(nx)/float(ny),
                           0.0f, dist_to_focus,
                           0.0, 1.0);
}

__global__ void create_world_cornell(hittable **d_list, hittable **d_world, camera **d_camera,
                                     int nx, int ny, int *d_count)
{
    if (threadIdx.x || blockIdx.x) return;
    int i = 0;

    // Only 3 lambertian materials
    material* red    = new lambertian(vec3(.65f,.05f,.05f));
    material* blue  = new lambertian(vec3(.15f,.15f,.75f));
    material* white  = new lambertian(vec3(.73f,.73f,.73f));   // reuse everywhere
    material* light  = new diffuse_light(vec3(15.f,15.f,15.f));

    // Cornell walls (inward-facing quads)
    d_list[i++] = new quad(vec3(0,0,0),       vec3(0,555,0),  vec3(0,0,555),  blue,  true); // left
    d_list[i++] = new quad(vec3(555,0,555),   vec3(0,555,0),  vec3(0,0,-555), red,    true); // right
  // NOTE: owns=false. This material is shared by several primitives and
  // each ~sphere/~quad would otherwise delete it, so the first teardown
  // frees it and the rest double-free. Ownership moves to a flat,
  // host-side material table in the refactor; until then it leaks once
  // at exit, which is strictly better than corrupting the device heap.
    d_list[i++] = new quad(vec3(0,0,0),       vec3(555,0,0),  vec3(0,0,555),  white,  true, false); // floor
    d_list[i++] = new quad(vec3(0,555,555),   vec3(555,0,0),  vec3(0,0,-555), white,  true, false); // ceiling
    d_list[i++] = new quad(vec3(555,0,555),   vec3(-555,0,0), vec3(0,555,0),  white,  true, false); // back
    d_list[i++] = new quad(vec3(213,554,227), vec3(130,0,0),  vec3(0,0,105),  light,  true); // light

    // ---- Instanced boxes ----
    // Build two *properly sized* prototypes (same geometry type, different height).
    hittable* proto_short = make_box(vec3(0,0,0), vec3(165,165,165), white);
    hittable* proto_tall  = make_box(vec3(0,0,0), vec3(165,330,165), white);

    // Place them using the canonical Cornell transforms.
    d_list[i++] = new translate(new rotate_y(proto_short, -18.f), vec3(130.f, 0.f,  65.f));
    d_list[i++] = new translate(new rotate_y(proto_tall,   15.f), vec3(265.f, 0.f, 295.f));

    // Add a floating dielectric sphere (glass)
    material* glass = new dielectric(1.5f);  // IOR ~1.5 (glass)

    // Place it toward the front center so it doesn't intersect boxes or the ceiling light
    // Cornell is 555³; this puts the center ~185 units up, radius 60
    d_list[i++] = new sphere(vec3(278.f, 335.f, 150.f), 60.f, glass, /*owns=*/false);

    // make it a *hollow* glass bubble with a thin shell:
    d_list[i++] = new sphere(vec3(278.f, 335.f, 150.f), -59.0f, glass, /*owns=*/false);

    // Report how many leaves were actually written. The host allocates
    // capacity, but only this many slots are initialised -- freeing past
    // here would delete uninitialised device memory.
    *d_count = i;
    *d_world = new bvh_node(d_list, 0, i);

    // Camera
    vec3 lookfrom(278,278,-800), lookat(278,278,0), vup(0,1,0);
    float dist_to_focus = (lookfrom - lookat).length();
    *d_camera = new camera(lookfrom, lookat, vup,
                           40.0f, float(nx)/float(ny),
                           0.0f, dist_to_focus,
                           0.0, 1.0);
}

__global__ void create_world_cornell_smoke(hittable **d_list, hittable **d_world, camera **d_camera,
                                           int nx, int ny, int *d_count) {
    if (threadIdx.x || blockIdx.x) return;
    int i = 0;

    material* red   = new lambertian(vec3(.65f, .05f, .05f));
    material* white = new lambertian(vec3(.73f, .73f, .73f));
    material* green = new lambertian(vec3(.12f, .45f, .15f));
    material* light = new diffuse_light(vec3(7.f, 7.f, 7.f));

    // Walls (inward-facing)
    d_list[i++] = new quad(vec3(555,0,0),   vec3(0,555,0),  vec3(0,0,555),  green, true);
    d_list[i++] = new quad(vec3(0,0,0),     vec3(0,555,0),  vec3(0,0,555),  red,   true);
  // NOTE: owns=false. This material is shared by several primitives and
  // each ~sphere/~quad would otherwise delete it, so the first teardown
  // frees it and the rest double-free. Ownership moves to a flat,
  // host-side material table in the refactor; until then it leaks once
  // at exit, which is strictly better than corrupting the device heap.
    d_list[i++] = new quad(vec3(0,555,0),   vec3(555,0,0),  vec3(0,0,555),  white, true, false);
    d_list[i++] = new quad(vec3(0,0,0),     vec3(555,0,0),  vec3(0,0,555),  white, true, false);
    d_list[i++] = new quad(vec3(0,0,555),   vec3(555,0,0),  vec3(0,555,0),  white, true, false);
    d_list[i++] = new quad(vec3(113,554,127), vec3(330,0,0), vec3(0,0,305), light, true);

    // Two boxes -> rotate/translate -> wrap each in constant_medium
    hittable* b1 = make_box(vec3(0,0,0), vec3(165,330,165), white);
    b1 = new translate(new rotate_y(b1, 15.f), vec3(265.f,0.f,295.f));
    hittable* b2 = make_box(vec3(0,0,0), vec3(165,165,165), white);
    b2 = new translate(new rotate_y(b2,-18.f), vec3(130.f,0.f, 65.f));

    d_list[i++] = new constant_medium(b1, 0.01f, vec3(0.5,0.5,0.5)); // black smoke
    d_list[i++] = new constant_medium(b2, 0.01f, vec3(1,1,1)); // white smoke

    // Report how many leaves were actually written. The host allocates
    // capacity, but only this many slots are initialised -- freeing past
    // here would delete uninitialised device memory.
    *d_count = i;
    *d_world = new bvh_node(d_list, 0, i);

    // Camera
    vec3 lookfrom(278, 278, -800), lookat(278, 278, 0), vup(0,1,0);
    float dist = (lookfrom - lookat).length();
    *d_camera = new camera(lookfrom, lookat, vup, 40.0f, float(nx)/float(ny),
                           0.0f, dist, 0.0f, 1.0f);
}

// degrees -> radians
__device__ inline float deg2rad(float d) { return d * 0.017453292519943295f; }

__device__ inline vec3 rotate_y_deg(const vec3& p, float deg) {
    float r = deg2rad(deg);
    float c = cosf(r), s = sinf(r);
    // R_y * p
    return vec3(c*p.x() + s*p.z(), p.y(), -s*p.x() + c*p.z());
}

__global__ void create_world_final(hittable **d_list, hittable **d_world, camera **d_camera,
                                   int nx, int ny, DeviceImage earth_img, int *d_count) {
    if (threadIdx.x || blockIdx.x) return;

    int i = 0;
    material* white = new lambertian(vec3(.73f,.73f,.73f));
    material* ground= new lambertian(vec3(0.48f,0.83f,0.53f));
    material* light = new diffuse_light(vec3(7,7,7));

    // --- Boxes "ground" 20x20 with random heights
    const int S = 20;
    for (int ix=0; ix<S; ++ix) for (int iz=0; iz<S; ++iz) {
        float w  = 100.0f;
        float x0 = -1000.0f + ix*w;
        float z0 = -1000.0f + iz*w;
        float y1 = 1.0f + 100.0f * ( (ix*13 + iz*37) % 100 ) / 100.0f; // stable pseudo-rand
        d_list[i++] = make_box(vec3(x0,0,z0), vec3(x0+w,y1,z0+w), ground);
    }

    // --- Area light quad
    d_list[i++] = new quad(vec3(123,554,147), vec3(300,0,0), vec3(0,0,265), light, true);

    // --- Moving lambertian sphere
    vec3 c1(400,400,200), c2 = c1 + vec3(30,0,0);
    d_list[i++] = new sphere(c1, c2, 50.f, new lambertian(vec3(0.7f,0.3f,0.1f)));

    // --- Glass & metal spheres
    d_list[i++] = new sphere(vec3(260,150,45), 50.f, new dielectric(1.5f));
    d_list[i++] = new sphere(vec3(0,150,145),  50.f, new metal(vec3(0.8f,0.8f,0.9f), 1.0f));

    // --- Blue fog sphere inside glass boundary
    hittable* boundary = new sphere(vec3(360,150,145), 70.f, new dielectric(1.5f));
    d_list[i++] = boundary; // boundary surface visible too (like the book)
    d_list[i++] = new constant_medium(new sphere(vec3(360,150,145), 70.f, new dielectric(1.5f)),
                                      0.2f, vec3(0.2f,0.4f,0.9f));

    // --- Global thin white fog
    d_list[i++] = new constant_medium(new sphere(vec3(0,0,0), 5000.f, new dielectric(1.5f)),
                                      0.0001f, vec3(1,1,1));

    // --- Earth-textured sphere
    texture* earth_tex = new image_texture(earth_img);
    d_list[i++] = new sphere(vec3(400,200,400), 100.f, new lambertian(earth_tex));

    // --- Perlin sphere
    d_list[i++] = new sphere(vec3(220,280,300), 80.f, new lambertian(new noise_texture(0.2f)));

    // --- Cluster of 1000 white balls (bake transform per-point)
    const int ns = 1000;
    for (int j = 0; j < ns; ++j) 
    {
        vec3 p = random_in_unit_cube(j) * 165.0f;   // see note below
        p = rotate_y_deg(p, 15.0f) + vec3(-100, 270, 395);  // match CPU scene
        // owns=false: all 1000 spheres share one material (see note above).
        d_list[i++] = new sphere(p, 10.0f, white, /*owns=*/false);
    }

    // Report how many leaves were actually written. The host allocates
    // capacity, but only this many slots are initialised -- freeing past
    // here would delete uninitialised device memory.
    *d_count = i;
    *d_world = new bvh_node(d_list, 0, i);

    // Camera
    vec3 lookfrom(478,278,-600), lookat(278,278,0), vup(0,1,0);
    *d_camera = new camera(lookfrom, lookat, vup,
                           40.0f, float(nx)/float(ny),
                           0.0f, (lookfrom-lookat).length(),
                           0.0, 1.0);
}

__global__ void create_world_original(hittable **d_list, hittable **d_world, camera **d_camera,
                                   int nx, int ny, DeviceImage earth_img, DeviceImage ball_img, int *d_count) {
    if (threadIdx.x || blockIdx.x) return;

    int i = 0;
    material* white = new lambertian(vec3(.73f,.73f,.73f));
    material* ground= new lambertian(vec3(0.88f, 0.50f, 0.76f));
    material* light = new diffuse_light(vec3(7,7,7));

    // --- Boxes "ground" 20x20 with random heights
    const int S = 20;
    for (int ix=0; ix<S; ++ix) for (int iz=0; iz<S; ++iz) {
        float w  = 100.0f;
        float x0 = -1000.0f + ix*w;
        float z0 = -1000.0f + iz*w;
        float y1 = 1.0f + 100.0f * ( (ix*13 + iz*37) % 100 ) / 100.0f; // stable pseudo-rand
        d_list[i++] = make_box(vec3(x0,0,z0), vec3(x0+w,y1,z0+w), ground);
    }

    // --- Area light quad
    d_list[i++] = new quad(vec3(123,554,147), vec3(300,0,0), vec3(0,0,265), light, true);

    // --- Moving lambertian sphere
    vec3 c1(400,400,200), c2 = c1 + vec3(30,0,0);
    d_list[i++] = new sphere(c1, c2, 50.f, new lambertian(vec3(0.0488f, 0.0148f, 0.0171f)));

    // --- Glass & metal spheres
    d_list[i++] = new sphere(vec3(260,150,45), 50.f, new dielectric(1.5f));
    d_list[i++] = new sphere(vec3(0,150,145),  50.f, new metal(vec3(0.6387f, 0.3605f, 0.8826f), 1.0f));

    // --- 8-ball textured lambertian (replaces glass boundary + constant_medium)
    // (360,150,145) radius 70 matches the original placement
    {
        texture* tex8 = new image_texture(ball_img);

        // Optional: rotate the decal toward camera (uncomment if you added uv_offset_texture)
        // tex8 = new uv_offset_texture(tex8, 30.f/360.f);  // +30° around Y

        material* eightball = new lambertian(tex8);
        d_list[i++] = new sphere(vec3(360.f, 150.f, 145.f), 70.f, eightball);
        material* coat = new dielectric(1.5f);
        d_list[i++] = new sphere(vec3(360,150,145), 70.f + 0.5f, coat); // small offset for a big sphere
    }

    // --- Global thin white fog
    d_list[i++] = new constant_medium(new sphere(vec3(0,0,0), 5000.f, new dielectric(1.5f)),
                                      0.0001f, vec3(1,1,1));

    // Highly polished metal (set fuzz to 0 for mirror, or 0.02 for a touch of blur)
    d_list[i++] = new sphere(vec3(400,200,400), 100.f, new metal(vec3(0.23, 0.24, 0.85), /*fuzz=*/0.02f));

    // --- Perlin sphere
    d_list[i++] = new sphere(vec3(220,280,300), 80.f, new lambertian(new noodle_texture(0.2f)));

    // --- Cluster of 1000 white balls (bake transform per-point)
    const int ns = 1000;
    for (int j = 0; j < ns; ++j) 
    {
        vec3 p = random_in_unit_cube(j) * 165.0f;   // see note below
        p = rotate_y_deg(p, 15.0f) + vec3(-100, 270, 395);  // match CPU scene
        // owns=false: all 1000 spheres share one material (see note above).
        d_list[i++] = new sphere(p, 10.0f, white, /*owns=*/false);
    }

    // Report how many leaves were actually written. The host allocates
    // capacity, but only this many slots are initialised -- freeing past
    // here would delete uninitialised device memory.
    *d_count = i;
    *d_world = new bvh_node(d_list, 0, i);

    // Camera
    vec3 lookfrom(478,278,-600), lookat(278,278,0), vup(0,1,0);
    *d_camera = new camera(lookfrom, lookat, vup,
                           40.0f, float(nx)/float(ny),
                           0.0f, (lookfrom-lookat).length(),
                           0.0, 1.0);
}
__global__ void free_world(hittable **d_list, int count,
                           hittable **d_world,
                           camera   **d_camera)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Order matters. ~bvh_node decides whether to recurse by calling the
        // virtual kind() on each child:
        //
        //     if (left && left->kind() == HK_BVH) delete left;
        //
        // so every leaf must still be alive when the tree is torn down. Freeing
        // the leaves first (as this kernel used to) reads a vtable pointer out
        // of freed device memory -- a use-after-free that surfaces as
        // cudaErrorInvalidPc, and only when the heap happens to get reused.
        //
        // 1) BVH internal nodes, while the leaves they point at are still valid.
        delete *d_world;

        // 2) Now the leaves, whose dtors free any material they own.
        for (int i = 0; i < count; ++i)
            delete d_list[i];

        // 3) Camera.
        delete *d_camera;
    }
}

// ===========================================================================
// Scene table
// ===========================================================================
//
// Every scene used to carry its own 70-line copy of the same allocate / build /
// render / print / free sequence. The duplication is where the leaks lived: the
// second cuRAND state was freed in three of ten copies, two scenes loaded the
// same texture twice and leaked the first upload, and two passed a `count` to
// free_world that did not match what the builder had written.
//
// The parameters below are the per-scene values from those original functions,
// carried over unchanged so existing renders reproduce.

enum SceneId {
    SCENE_BOUNCING = 0, SCENE_CHECKERED, SCENE_EARTH, SCENE_PERLIN, SCENE_QUADS,
    SCENE_SIMPLE_LIGHT, SCENE_CORNELL, SCENE_CORNELL_SMOKE, SCENE_FINAL,
    SCENE_ORIGINAL, SCENE_COUNT
};

struct SceneSpec {
    SceneId     id;
    const char* name;
    int         width, height, spp;
    vec3        background;
    int         gradient_bg;       // 1 -> sky gradient on miss, 0 -> flat background
    int         capacity;          // d_list slots to allocate
    size_t      stack_bytes;
    size_t      heap_bytes;
    const char* texture_a;         // nullptr when the scene needs no image texture
    const char* texture_b;
};

static const SceneSpec kScenes[SCENE_COUNT] = {
  // id                   name             W     H     spp   background                      grad  cap   stack   heap         tex_a                    tex_b
  { SCENE_BOUNCING,      "bouncing",      1200,  600, 10000, vec3(0,0,0),                     0,   512,  16384,  64u<<20,  nullptr,                  nullptr },
  { SCENE_CHECKERED,     "checkered",     1200,  600,   500, vec3(0,0,0),                     1,     8,  16384,  64u<<20,  nullptr,                  nullptr },
  { SCENE_EARTH,         "earth",         1200,  600,   500, vec3(0,0,0),                     1,     8,  16384,  64u<<20,  "textures/earthmap.jpg",  nullptr },
  { SCENE_PERLIN,        "perlin",        1200,  600,   500, vec3(0,0,0),                     1,     8,  16384,  64u<<20,  nullptr,                  nullptr },
  { SCENE_QUADS,         "quads",         1200,  600,   500, vec3(0,0,0),                     1,    16,  16384,  64u<<20,  nullptr,                  nullptr },
  // capacity 16, not the original 4: the builder writes 5 leaves (ground, ball
  // core, clear coat, light sphere, light quad) into what was a 4-slot array.
  { SCENE_SIMPLE_LIGHT,  "simple_light",  1200,  600, 10000, vec3(0,0,0),                     0,    16,  16384,  64u<<20,  "textures/poolball.jpg",  nullptr },
  // capacity 16, not the original 6: the builder writes 10 (6 walls, 2 boxes,
  // 2 spheres). Both of these were out-of-bounds device writes.
  { SCENE_CORNELL,       "cornell",        600,  600, 10000, vec3(0,0,0),                     0,    16,  16384,  64u<<20,  nullptr,                  nullptr },
  { SCENE_CORNELL_SMOKE, "cornell_smoke",  600,  600,  1000, vec3(0,0,0),                     0,    16,  65536, 256u<<20,  nullptr,                  nullptr },
  { SCENE_FINAL,         "final",          800,  800, 10000, vec3(0,0,0),                     0,  1800,  32768, 256u<<20,  "textures/earthmap.jpg",  nullptr },
  { SCENE_ORIGINAL,      "original",       800,  800, 10000, vec3(0.043f,0.030f,0.094f),      0,  1800,  32768, 256u<<20,  "textures/porcelain.jpg", "textures/8ball.jpg" },
};

static void launch_scene_builder(const SceneSpec& spec,
                                 hittable **d_list, hittable **d_world, camera **d_camera,
                                 int nx, int ny, curandState *d_rand_state2,
                                 DeviceImage tex_a, DeviceImage tex_b, int *d_count)
{
    switch (spec.id) {
        case SCENE_BOUNCING:
            create_world_bouncing<<<1,1>>>(d_list, d_world, d_camera, nx, ny, d_rand_state2, d_count); break;
        case SCENE_CHECKERED:
            create_world_checker<<<1,1>>>(d_list, d_world, d_camera, nx, ny, d_rand_state2, d_count); break;
        case SCENE_EARTH:
            create_world_earth<<<1,1>>>(d_list, d_world, d_camera, nx, ny, tex_a, d_count); break;
        case SCENE_PERLIN:
            create_world_perlin<<<1,1>>>(d_list, d_world, d_camera, nx, ny, /*scale=*/4.0f, d_count); break;
        case SCENE_QUADS:
            create_world_quads<<<1,1>>>(d_list, d_world, d_camera, nx, ny, d_count); break;
        case SCENE_SIMPLE_LIGHT:
            create_world_simple_light<<<1,1>>>(d_list, d_world, d_camera, nx, ny, tex_a, d_count); break;
        case SCENE_CORNELL:
            create_world_cornell<<<1,1>>>(d_list, d_world, d_camera, nx, ny, d_count); break;
        case SCENE_CORNELL_SMOKE:
            create_world_cornell_smoke<<<1,1>>>(d_list, d_world, d_camera, nx, ny, d_count); break;
        case SCENE_FINAL:
            create_world_final<<<1,1>>>(d_list, d_world, d_camera, nx, ny, tex_a, d_count); break;
        case SCENE_ORIGINAL:
            create_world_original<<<1,1>>>(d_list, d_world, d_camera, nx, ny, tex_a, tex_b, d_count); break;
        default: break;
    }
}

// ===========================================================================
// Command line
// ===========================================================================

struct Options {
    int          scene       = SCENE_ORIGINAL;
    int          width       = -1;      // -1 -> take the scene default
    int          height      = -1;
    int          spp         = -1;
    int          max_depth   = 50;      // was hard-coded in color()
    float        gamma       = 2.2f;
    unsigned long long seed  = 1984;
    int          tx          = 8;
    int          ty          = 8;
    int          batch       = 64;      // samples per kernel launch; see below
    std::string  out;                   // empty -> stdout
    std::string  stats;                 // empty -> no machine-readable stats
    bool         binary     = true;     // P6 by default; --ascii for P3
    bool         quiet      = false;
    bool         list       = false;
};

static void usage(const char* argv0) {
    std::fprintf(stderr,
      "usage: %s [options]\n"
      "\n"
      "  --scene NAME       scene to render (default: original, --list to see all)\n"
      "  --width N          override image width\n"
      "  --height N         override image height\n"
      "  --spp N            samples per pixel\n"
      "  --max-depth N      maximum ray bounces (default 50)\n"
      "  --gamma F          gamma exponent (default 2.2)\n"
      "  --seed N           RNG seed; identical seed + params => identical image\n"
      "  --block X Y        thread block dimensions (default 8 8)\n"
      "  --batch N          samples per kernel launch (default 64, 0 = all at once)\n"
      "  --out FILE         write PPM here instead of stdout\n"
      "  --stats FILE       write timing JSON here\n"
      "  --ascii            emit P3 text PPM instead of P6 binary\n"
      "  --quiet            suppress progress output\n"
      "  --list             list scene names and their defaults\n", argv0);
}

static bool parse_args(int argc, char** argv, Options* o) {
    auto need = [&](int i, const char* what) {
        if (i >= argc) { std::fprintf(stderr, "error: %s requires a value\n", what); return false; }
        return true;
    };
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if      (a == "--list")  { o->list = true; }
        else if (a == "--quiet") { o->quiet = true; }
        else if (a == "--ascii") { o->binary = false; }
        else if (a == "-h" || a == "--help") { usage(argv[0]); return false; }
        else if (a == "--scene") {
            if (!need(++i, "--scene")) return false;
            bool found = false;
            for (int s = 0; s < SCENE_COUNT; ++s)
                if (std::strcmp(kScenes[s].name, argv[i]) == 0) { o->scene = s; found = true; break; }
            if (!found) { std::fprintf(stderr, "error: unknown scene '%s' (try --list)\n", argv[i]); return false; }
        }
        else if (a == "--width")     { if (!need(++i, "--width")) return false;     o->width = std::atoi(argv[i]); }
        else if (a == "--height")    { if (!need(++i, "--height")) return false;    o->height = std::atoi(argv[i]); }
        else if (a == "--spp")       { if (!need(++i, "--spp")) return false;       o->spp = std::atoi(argv[i]); }
        else if (a == "--max-depth") { if (!need(++i, "--max-depth")) return false; o->max_depth = std::atoi(argv[i]); }
        else if (a == "--gamma")     { if (!need(++i, "--gamma")) return false;     o->gamma = (float)std::atof(argv[i]); }
        else if (a == "--seed")      { if (!need(++i, "--seed")) return false;      o->seed = std::strtoull(argv[i], nullptr, 10); }
        else if (a == "--batch")     { if (!need(++i, "--batch")) return false;     o->batch = std::atoi(argv[i]); }
        else if (a == "--out")       { if (!need(++i, "--out")) return false;       o->out = argv[i]; }
        else if (a == "--stats")     { if (!need(++i, "--stats")) return false;     o->stats = argv[i]; }
        else if (a == "--block") {
            if (!need(++i, "--block")) return false; o->tx = std::atoi(argv[i]);
            if (!need(++i, "--block")) return false; o->ty = std::atoi(argv[i]);
        }
        else { std::fprintf(stderr, "error: unknown option '%s'\n", a.c_str()); usage(argv[0]); return false; }
    }
    return true;
}

// ===========================================================================
// Output
// ===========================================================================

// The original wrote P3 with one std::cout insertion per channel: about 2.1M
// formatted stream writes for a 1200x600 frame, which took longer than some of
// the renders. P6 is the same image as a single fwrite.
static bool write_ppm(const char* path, bool binary, const vec3* fb, int nx, int ny)
{
    std::FILE* f = path ? std::fopen(path, binary ? "wb" : "w") : stdout;
    if (!f) { std::fprintf(stderr, "error: cannot open '%s' for writing\n", path); return false; }

    // Row 0 of the framebuffer is the *bottom* of the image, so emit rows in
    // reverse to match PPM's top-to-bottom order.
    auto quantise = [](float c) -> unsigned char {
        const int v = (int)(255.99f * c);
        return (unsigned char)(v < 0 ? 0 : (v > 255 ? 255 : v));
    };

    if (binary) {
        std::fprintf(f, "P6\n%d %d\n255\n", nx, ny);
        std::vector<unsigned char> row((size_t)nx * 3);
        for (int j = ny - 1; j >= 0; --j) {
            for (int i = 0; i < nx; ++i) {
                const vec3& c = fb[(size_t)j * nx + i];
                row[i * 3 + 0] = quantise(c.r());
                row[i * 3 + 1] = quantise(c.g());
                row[i * 3 + 2] = quantise(c.b());
            }
            std::fwrite(row.data(), 1, row.size(), f);
        }
    } else {
        std::fprintf(f, "P3\n%d %d\n255\n", nx, ny);
        for (int j = ny - 1; j >= 0; --j)
            for (int i = 0; i < nx; ++i) {
                const vec3& c = fb[(size_t)j * nx + i];
                std::fprintf(f, "%d %d %d\n", quantise(c.r()), quantise(c.g()), quantise(c.b()));
            }
    }

    const bool ok = std::ferror(f) == 0;
    if (path) std::fclose(f);
    else      std::fflush(f);
    return ok;
}

// ===========================================================================
// Driver
// ===========================================================================

static int run_scene(const SceneSpec& spec, const Options& opt)
{
    const int nx = opt.width  > 0 ? opt.width  : spec.width;
    const int ny = opt.height > 0 ? opt.height : spec.height;
    const int ns = opt.spp    > 0 ? opt.spp    : spec.spp;

    // The BVH is still built recursively on the device, so each thread needs a
    // deep stack, and every hittable/material/texture comes from device `new`,
    // so the malloc heap has to be grown too. Both limits disappear once the
    // scene is built host-side into flat arrays.
    checkCudaErrors(cudaDeviceSetLimit(cudaLimitStackSize,      spec.stack_bytes));
    checkCudaErrors(cudaDeviceSetLimit(cudaLimitMallocHeapSize, spec.heap_bytes));

    // --- textures -----------------------------------------------------------
    // Loaded once. The old final/original scenes called load_image_to_device
    // twice on the same path and leaked the first upload every run.
    DeviceImage tex_a{}, tex_b{};
    if (spec.texture_a) {
        tex_a = load_image_to_device(spec.texture_a);
        if (!tex_a.valid()) { std::fprintf(stderr, "error: failed to load %s\n", spec.texture_a); return 1; }
    }
    if (spec.texture_b) {
        tex_b = load_image_to_device(spec.texture_b);
        if (!tex_b.valid()) { std::fprintf(stderr, "error: failed to load %s\n", spec.texture_b); free_device_image(tex_a); return 1; }
    }

    const int    num_pixels = nx * ny;
    const size_t fb_bytes   = (size_t)num_pixels * sizeof(vec3);

    // --- allocations --------------------------------------------------------
    vec3        *fb = nullptr, *accum = nullptr;
    curandState *d_rand_state = nullptr, *d_rand_state2 = nullptr;
    camera     **d_camera = nullptr;
    hittable   **d_list = nullptr, **d_world = nullptr;
    int         *d_count = nullptr;

    checkCudaErrors(cudaMallocManaged((void**)&fb,    fb_bytes));
    checkCudaErrors(cudaMalloc((void**)&accum,        fb_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_rand_state,  (size_t)num_pixels * sizeof(curandState)));
    checkCudaErrors(cudaMalloc((void**)&d_rand_state2, sizeof(curandState)));
    checkCudaErrors(cudaMalloc((void**)&d_camera,      sizeof(camera*)));
    checkCudaErrors(cudaMalloc((void**)&d_world,       sizeof(hittable*)));
    checkCudaErrors(cudaMalloc((void**)&d_list,        (size_t)spec.capacity * sizeof(hittable*)));
    checkCudaErrors(cudaMallocManaged((void**)&d_count, sizeof(int)));

    // Zeroing matters: free_world walks d_list and calls delete on each slot.
    // Any slot the builder did not write would otherwise be a garbage pointer.
    checkCudaErrors(cudaMemset(d_list, 0, (size_t)spec.capacity * sizeof(hittable*)));
    *d_count = 0;

    const dim3 blocks((nx + opt.tx - 1) / opt.tx, (ny + opt.ty - 1) / opt.ty);
    const dim3 threads(opt.tx, opt.ty);

    // --- build --------------------------------------------------------------
    // cudaEvent measures GPU time directly; the old clock() call measured host
    // CPU time and has ~10-15 ms resolution on Windows.
    cudaEvent_t ev_build0, ev_build1, ev_render0, ev_render1;
    checkCudaErrors(cudaEventCreate(&ev_build0));  checkCudaErrors(cudaEventCreate(&ev_build1));
    checkCudaErrors(cudaEventCreate(&ev_render0)); checkCudaErrors(cudaEventCreate(&ev_render1));

    checkCudaErrors(cudaEventRecord(ev_build0));
    rand_init<<<1,1>>>(d_rand_state2, opt.seed);
    checkCudaErrors(cudaGetLastError());
    launch_scene_builder(spec, d_list, d_world, d_camera, nx, ny, d_rand_state2, tex_a, tex_b, d_count);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaEventRecord(ev_build1));
    checkCudaErrors(cudaDeviceSynchronize());

    const int built = *d_count;
    if (built > spec.capacity) {
        // Already too late to be safe, but far better than failing silently:
        // this is exactly the condition that was corrupting the heap before.
        std::fprintf(stderr, "FATAL: scene '%s' wrote %d objects into %d slots\n",
                     spec.name, built, spec.capacity);
        return 2;
    }
    if (!opt.quiet)
        std::fprintf(stderr, "scene '%s': %d objects, %dx%d, %d spp, depth %d\n",
                     spec.name, built, nx, ny, ns, opt.max_depth);

    // --- render -------------------------------------------------------------
    checkCudaErrors(cudaEventRecord(ev_render0));
    clear_buffer<<<(num_pixels + 255) / 256, 256>>>(accum, num_pixels);
    render_init<<<blocks, threads>>>(nx, ny, d_rand_state, opt.seed);
    checkCudaErrors(cudaGetLastError());

    // Issue the sample budget in batches. Mathematically identical to one long
    // launch -- the RNG stream is carried across launches in d_rand_state -- but
    // each launch stays short enough to survive the Windows display watchdog,
    // and it gives us a progress indicator for free.
    const int batch = (opt.batch > 0 && opt.batch < ns) ? opt.batch : ns;
    for (int done = 0; done < ns; done += batch) {
        const int n = (done + batch <= ns) ? batch : (ns - done);
        render_accumulate<<<blocks, threads>>>(accum, nx, ny, n, opt.max_depth,
                                               d_camera, d_world, d_rand_state,
                                               spec.background, spec.gradient_bg);
        checkCudaErrors(cudaGetLastError());
        if (!opt.quiet) {
            checkCudaErrors(cudaDeviceSynchronize());
            std::fprintf(stderr, "\r  %d/%d samples (%.0f%%)   ",
                         done + n, ns, 100.0 * (done + n) / ns);
        }
    }
    resolve<<<blocks, threads>>>(fb, accum, nx, ny, ns, opt.gamma);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaEventRecord(ev_render1));
    checkCudaErrors(cudaDeviceSynchronize());
    if (!opt.quiet) std::fprintf(stderr, "\n");

    float build_ms = 0.0f, render_ms = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&build_ms,  ev_build0,  ev_build1));
    checkCudaErrors(cudaEventElapsedTime(&render_ms, ev_render0, ev_render1));

    if (!opt.quiet)
        std::fprintf(stderr, "build %.1f ms, render %.3f s (%.2f Mpaths/s)\n",
                     build_ms, render_ms / 1000.0,
                     ((double)num_pixels * ns) / (render_ms * 1000.0));

    // --- output -------------------------------------------------------------
    const bool wrote = write_ppm(opt.out.empty() ? nullptr : opt.out.c_str(),
                                 opt.binary, fb, nx, ny);
    if (!wrote) std::fprintf(stderr, "error: failed writing image\n");

    if (!opt.stats.empty()) {
        std::FILE* sf = std::fopen(opt.stats.c_str(), "w");
        if (sf) {
            cudaDeviceProp prop{};
            cudaGetDeviceProperties(&prop, 0);
            std::fprintf(sf,
                "{\n"
                "  \"backend\": \"cuda\",\n"
                "  \"device\": \"%s\",\n"
                "  \"scene\": \"%s\",\n"
                "  \"width\": %d,\n"
                "  \"height\": %d,\n"
                "  \"spp\": %d,\n"
                "  \"max_depth\": %d,\n"
                "  \"seed\": %llu,\n"
                "  \"objects\": %d,\n"
                "  \"block\": [%d, %d],\n"
                "  \"batch\": %d,\n"
                "  \"build_ms\": %.4f,\n"
                "  \"render_ms\": %.4f,\n"
                "  \"primary_rays\": %lld\n"
                "}\n",
                prop.name, spec.name, nx, ny, ns, opt.max_depth, opt.seed, built,
                opt.tx, opt.ty, batch, build_ms, render_ms,
                (long long)num_pixels * ns);
            std::fclose(sf);
        } else {
            std::fprintf(stderr, "warning: cannot write stats to '%s'\n", opt.stats.c_str());
        }
    }

    // --- teardown -----------------------------------------------------------
    // `built`, not `capacity`: deleting past the last initialised slot was
    // walking uninitialised device memory in the final/original scenes.
    free_world<<<1,1>>>(d_list, built, d_world, d_camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaEventDestroy(ev_build0));  checkCudaErrors(cudaEventDestroy(ev_build1));
    checkCudaErrors(cudaEventDestroy(ev_render0)); checkCudaErrors(cudaEventDestroy(ev_render1));

    checkCudaErrors(cudaFree(d_count));
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_rand_state2));   // was leaked by seven of ten scenes
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(accum));
    checkCudaErrors(cudaFree(fb));
    free_device_image(tex_a);
    free_device_image(tex_b);                   // was leaked by the original scene

    return wrote ? 0 : 1;
}

int main(int argc, char** argv)
{
    Options opt;
    if (!parse_args(argc, argv, &opt)) return 1;

    if (opt.list) {
        std::printf("%-16s %9s %7s  %s\n", "scene", "size", "spp", "textures");
        for (int i = 0; i < SCENE_COUNT; ++i) {
            const SceneSpec& s = kScenes[i];
            std::printf("%-16s %4dx%-4d %7d  %s%s%s\n", s.name, s.width, s.height, s.spp,
                        s.texture_a ? s.texture_a : "-",
                        s.texture_b ? ", " : "", s.texture_b ? s.texture_b : "");
        }
        return 0;
    }

    const int rc = run_scene(kScenes[opt.scene], opt);

    // cudaDeviceReset makes leak checking under compute-sanitizer meaningful:
    // it forces the driver to report anything still allocated at exit.
    cudaDeviceReset();
    return rc;
}
