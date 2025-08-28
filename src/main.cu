#define STB_IMAGE_IMPLEMENTATION
#include <curand_kernel.h>
#include <float.h>
#include <iostream>
#include <math.h>
#include <math_constants.h>
#include <time.h>

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
                      hittable **world,
                      curandState *local_rand_state)
{
    ray  cur_ray        = r0;
    vec3 throughput     = vec3(1,1,1);
    vec3 radiance       = vec3(0,0,0);

    for (int bounce = 0; bounce < 50; ++bounce) 
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

__global__ void rand_init(curandState *rand_state) 
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curand_init(1984, 0, 0, rand_state);
    }
}

__global__ void render_init(int max_x, int max_y, curandState *rand_state)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j*max_x + i;

    // Same seed for each thread
    curand_init(1984+pixel_index, 0, 0, &rand_state[pixel_index]);
}

__global__ void render(vec3 *fb, int max_x, int max_y, int ns, float gamma,
                       camera **cam, hittable **world, curandState *rand_state,
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
        col += color(r, background, use_gradient_bg != 0, world, &local_rand_state);
    }
    rand_state[pixel_index] = local_rand_state;

    col /= float(ns);
    col[0] = apply_gamma(col[0], gamma);
    col[1] = apply_gamma(col[1], gamma);
    col[2] = apply_gamma(col[2], gamma);
    fb[pixel_index] = col;
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
                             int nx, int ny, curandState *rand_state)
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
                                     int nx, int ny, curandState *rand_state)
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
        d_list[i++] = new sphere(vec3(0,-10,0), 10.0f, lam);
        d_list[i++] = new sphere(vec3(0, 10,0), 10.0f, lam);

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
                                   int nx, int ny, DeviceImage earth_img)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int i = 0;

        // Textured earth sphere at the origin
        texture*  earth_tex  = new image_texture(earth_img);
        material* earth_lam  = new lambertian(earth_tex);
        d_list[i++] = new sphere(vec3(0,0,0), 2.0f, earth_lam);

        // Wrap in BVH (okay even for 1 object, matches your other scenes)
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
                                    int nx, int ny, float scale)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int i = 0;

        texture* pertext = new noise_texture(scale);
        material* lam    = new lambertian(pertext);

        d_list[i++] = new sphere(vec3(0,-1000,0), 1000.f, lam);
        d_list[i++] = new sphere(vec3(0,     2,0),    2.f, lam);

        *d_world = new bvh_node(d_list, 0, i);

        vec3 lookfrom(13,2,3), lookat(0,0,0), vup(0,1,0);
        *d_camera = new camera(lookfrom, lookat, vup,
                               20.0f, float(nx)/float(ny),
                               0.0f, 10.0f, 0.0, 1.0);
    }
}

__global__ void create_world_quads(hittable **d_list, hittable **d_world, camera **d_camera,
                                   int nx, int ny)
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

        *d_world = new bvh_node(d_list, 0, i);

        vec3 lookfrom(0,0,9), lookat(0,0,0), vup(0,1,0);
        *d_camera = new camera(lookfrom, lookat, vup,
                               80.0f, float(nx)/float(ny),
                               0.0f, 10.0f, 0.0, 1.0);
    }
}

__global__ void create_world_simple_light(hittable **d_list, hittable **d_world, camera **d_camera,
                                          int nx, int ny, DeviceImage ball_img)
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
                                     int nx, int ny)
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
    d_list[i++] = new quad(vec3(0,0,0),       vec3(555,0,0),  vec3(0,0,555),  white,  true); // floor
    d_list[i++] = new quad(vec3(0,555,555),   vec3(555,0,0),  vec3(0,0,-555), white,  true); // ceiling
    d_list[i++] = new quad(vec3(555,0,555),   vec3(-555,0,0), vec3(0,555,0),  white,  true); // back
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
    d_list[i++] = new sphere(vec3(278.f, 335.f, 150.f), 60.f, glass);

    // make it a *hollow* glass bubble with a thin shell:
    d_list[i++] = new sphere(vec3(278.f, 335.f, 150.f), -59.0f, glass);

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
                                           int nx, int ny) {
    if (threadIdx.x || blockIdx.x) return;
    int i = 0;

    material* red   = new lambertian(vec3(.65f, .05f, .05f));
    material* white = new lambertian(vec3(.73f, .73f, .73f));
    material* green = new lambertian(vec3(.12f, .45f, .15f));
    material* light = new diffuse_light(vec3(7.f, 7.f, 7.f));

    // Walls (inward-facing)
    d_list[i++] = new quad(vec3(555,0,0),   vec3(0,555,0),  vec3(0,0,555),  green, true);
    d_list[i++] = new quad(vec3(0,0,0),     vec3(0,555,0),  vec3(0,0,555),  red,   true);
    d_list[i++] = new quad(vec3(0,555,0),   vec3(555,0,0),  vec3(0,0,555),  white, true);
    d_list[i++] = new quad(vec3(0,0,0),     vec3(555,0,0),  vec3(0,0,555),  white, true);
    d_list[i++] = new quad(vec3(0,0,555),   vec3(555,0,0),  vec3(0,555,0),  white, true);
    d_list[i++] = new quad(vec3(113,554,127), vec3(330,0,0), vec3(0,0,305), light, true);

    // Two boxes -> rotate/translate -> wrap each in constant_medium
    hittable* b1 = make_box(vec3(0,0,0), vec3(165,330,165), white);
    b1 = new translate(new rotate_y(b1, 15.f), vec3(265.f,0.f,295.f));
    hittable* b2 = make_box(vec3(0,0,0), vec3(165,165,165), white);
    b2 = new translate(new rotate_y(b2,-18.f), vec3(130.f,0.f, 65.f));

    d_list[i++] = new constant_medium(b1, 0.01f, vec3(0.5,0.5,0.5)); // black smoke
    d_list[i++] = new constant_medium(b2, 0.01f, vec3(1,1,1)); // white smoke

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
                                   int nx, int ny, DeviceImage earth_img) {
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
        d_list[i++] = new sphere(p, 10.0f, white);
    }

    *d_world = new bvh_node(d_list, 0, i);

    // Camera
    vec3 lookfrom(478,278,-600), lookat(278,278,0), vup(0,1,0);
    *d_camera = new camera(lookfrom, lookat, vup,
                           40.0f, float(nx)/float(ny),
                           0.0f, (lookfrom-lookat).length(),
                           0.0, 1.0);
}

__global__ void create_world_original(hittable **d_list, hittable **d_world, camera **d_camera,
                                   int nx, int ny, DeviceImage earth_img, DeviceImage ball_img) {
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
        d_list[i++] = new sphere(p, 10.0f, white);
    }

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
        // 1) Delete all leaves (their virtual dtors free owned materials)
        for (int i = 0; i < count; ++i)
            delete d_list[i];

        // 2) Delete BVH internal nodes (dtor skips leaves by design)
        delete *d_world;

        // 3) Delete camera
        delete *d_camera;
    }
}

void bouncing_spheres()
{
    int nx = 1200;
    int ny = 600;
    int ns = 10000;
    float gamma = 2.2f;
    int tx = 8;
    int ty = 8;

    // Increase per thread call stack size and device heap
    // Temporary workaround since bvh is still recursive 
    cudaDeviceSetLimit(cudaLimitStackSize,      16384);        // 16 KB
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 64*1024*1024); // 64 MB device heap

    std::cerr << "Rendering a " << nx << "x" << ny << " image ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    int num_pixels = nx * ny;
    size_t fb_size = num_pixels * sizeof(vec3);

    // Allocate frame buffer
    vec3 *fb;
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

    // Allocate random state
    curandState *d_rand_state;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels*sizeof(curandState)));
    curandState *d_rand_state2;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state2, 1*sizeof(curandState)));

    // we need that 2nd random state to be initialized for the world creation
    rand_init<<<1,1>>>(d_rand_state2);

    // Make camera and world with hittables
    camera **d_camera;
    checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(camera *)));
    hittable **d_list;
    int num_hitables = 22*22 + 1 + 3;
    checkCudaErrors(cudaMalloc((void **)&d_list, num_hitables * sizeof(hittable *)));
    hittable **d_world;
    checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hittable *)));
    create_world_bouncing<<<1,1>>>(d_list, d_world, d_camera, nx, ny, d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    clock_t start, stop;
    start = clock();
    // Render buffer
    dim3 blocks(nx/tx+1, ny/ty+1);
    dim3 threads(tx, ty);
    render_init<<<blocks, threads>>>(nx, ny, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    render<<<blocks, threads>>>(fb, nx, ny,  ns, gamma, d_camera, d_world, d_rand_state, vec3(0,0,0), 0);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    stop = clock();
    double timer_seconds = ((double(stop - start)) / CLOCKS_PER_SEC);
    std::cerr << "took " << timer_seconds << " seconds.\n";

    // Output frame buffer as image
    std::cout << "P3\n" << nx << " " << ny << "\n255\n";

    for (int j = ny-1; j >= 0; j--)
    {
        for (int i = 0; i < nx; i++)
        {
            size_t pixel_index = j*nx + i;
            int ir = int(255.99*fb[pixel_index].r());
            int ig = int(255.99*fb[pixel_index].g());
            int ib = int(255.99*fb[pixel_index].b());
            std::cout << ir << " " << ig << " " << ib << "\n";
        }
    }
    
    // Clean up
    checkCudaErrors(cudaDeviceSynchronize());                 // make sure render finished

    free_world<<<1,1>>>(d_list, num_hitables, d_world, d_camera);   // NOTE: count is 2nd arg
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());                 // wait for device-side deletes

    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(fb));

    // Useful for cuda-memcheck --leak-check full
    cudaDeviceReset();
}

int checkered_spheres() {
    int nx = 1200, ny = 600, ns = 500;
    float gamma = 2.2f;
    int tx = 8, ty = 8;

    cudaDeviceSetLimit(cudaLimitStackSize,      16384);
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 64*1024*1024);

    int num_pixels = nx * ny;
    size_t fb_size = num_pixels * sizeof(vec3);

    vec3 *fb;                      checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));
    curandState *d_rand_state;     checkCudaErrors(cudaMalloc((void **)&d_rand_state,  num_pixels*sizeof(curandState)));
    curandState *d_rand_state2;    checkCudaErrors(cudaMalloc((void **)&d_rand_state2, 1*sizeof(curandState)));
    rand_init<<<1,1>>>(d_rand_state2);

    camera **d_camera;             checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(camera *)));
    int num_hitables = 2;          // two big spheres
    hittable **d_list;             checkCudaErrors(cudaMalloc((void **)&d_list, num_hitables * sizeof(hittable *)));
    hittable **d_world;            checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hittable *)));

    create_world_checker<<<1,1>>>(d_list, d_world, d_camera, nx, ny, d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    dim3 blocks(nx/tx+1, ny/ty+1), threads(tx, ty);
    render_init<<<blocks, threads>>>(nx, ny, d_rand_state);
    checkCudaErrors(cudaDeviceSynchronize());
    render<<<blocks, threads>>>(fb, nx, ny, ns, gamma, d_camera, d_world, d_rand_state, vec3(0,0,0), 1);
    checkCudaErrors(cudaDeviceSynchronize());

    std::cout << "P3\n" << nx << " " << ny << "\n255\n";
    for (int j = ny-1; j >= 0; --j) for (int i = 0; i < nx; ++i) {
        size_t k = j*nx + i;
        int ir = int(255.99f*fb[k].r());
        int ig = int(255.99f*fb[k].g());
        int ib = int(255.99f*fb[k].b());
        std::cout << ir << " " << ig << " " << ib << "\n";
    }

    // --- Clean up ---
    checkCudaErrors(cudaDeviceSynchronize());                 // make sure render finished

    free_world<<<1,1>>>(d_list, num_hitables, d_world, d_camera);   // NOTE: count is 2nd arg
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());                 // wait for device-side deletes

    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(fb));
    cudaDeviceReset();
    return 0;
}

int earth() {
    // Render params (match your other scenes)
    int nx = 1200;
    int ny = 600;
    int ns = 500;
    float gamma = 2.2f;
    int tx = 8, ty = 8;

    // Device limits (you already do this elsewhere too)
    cudaDeviceSetLimit(cudaLimitStackSize,      16384);
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 64*1024*1024);

    // Load earth texture on HOST, upload to device
    unsigned char* d_pixels = nullptr;
    DeviceImage earth_img   = load_image_to_device("textures/earthmap.jpg", &d_pixels);
    if (!earth_img.valid()) {
        std::cerr << "Failed to load earthmap.jpg\n";
        return 1;
    }

    // Framebuffer
    int num_pixels = nx * ny;
    size_t fb_size = num_pixels * sizeof(vec3);
    vec3 *fb; checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

    // RNG
    curandState *d_rand_state;  checkCudaErrors(cudaMalloc((void **)&d_rand_state,  num_pixels*sizeof(curandState)));
    curandState *d_rand_state2; checkCudaErrors(cudaMalloc((void **)&d_rand_state2, 1*sizeof(curandState)));
    rand_init<<<1,1>>>(d_rand_state2);

    // Scene allocations
    camera **d_camera; checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(camera *)));
    int num_hitables = 1;       // just the globe
    hittable **d_list;  checkCudaErrors(cudaMalloc((void **)&d_list,  num_hitables * sizeof(hittable *)));
    hittable **d_world; checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hittable *)));

    // Build world on device
    create_world_earth<<<1,1>>>(d_list, d_world, d_camera, nx, ny, earth_img);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Render
    dim3 blocks(nx/tx+1, ny/ty+1), threads(tx, ty);
    render_init<<<blocks, threads>>>(nx, ny, d_rand_state);
    checkCudaErrors(cudaDeviceSynchronize());
    render<<<blocks, threads>>>(fb, nx, ny, ns, gamma, d_camera, d_world, d_rand_state, vec3(0,0,0), 1);
    checkCudaErrors(cudaDeviceSynchronize());

    // Output PPM
    std::cout << "P3\n" << nx << " " << ny << "\n255\n";
    for (int j = ny-1; j >= 0; --j) {
        for (int i = 0; i < nx; ++i) {
            size_t k = j*nx + i;
            int ir = int(255.99f*fb[k].r());
            int ig = int(255.99f*fb[k].g());
            int ib = int(255.99f*fb[k].b());
            std::cout << ir << " " << ig << " " << ib << "\n";
        }
    }

    // Cleanup
    checkCudaErrors(cudaDeviceSynchronize());                 // make sure render finished

    free_world<<<1,1>>>(d_list, num_hitables, d_world, d_camera);   // NOTE: count is 2nd arg
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());                 // wait for device-side deletes

    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(fb));

    // Free device pixels for the earth texture
    free_device_image(earth_img);

    cudaDeviceReset();
    return 0;
}

int perlin() {
    int nx = 1200, ny = 600, ns = 500;
    float gamma = 2.2f;
    int tx = 8, ty = 8;

    cudaDeviceSetLimit(cudaLimitStackSize,      16384);
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 64*1024*1024);

    int num_pixels = nx * ny;
    size_t fb_size = num_pixels * sizeof(vec3);

    vec3 *fb;                      checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));
    curandState *d_rand_state;     checkCudaErrors(cudaMalloc((void **)&d_rand_state,  num_pixels*sizeof(curandState)));
    curandState *d_rand_state2;    checkCudaErrors(cudaMalloc((void **)&d_rand_state2, 1*sizeof(curandState)));
    rand_init<<<1,1>>>(d_rand_state2);

    camera **d_camera;             checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(camera *)));
    int num_hitables = 2;          // ground + sphere
    hittable **d_list;             checkCudaErrors(cudaMalloc((void **)&d_list,  num_hitables * sizeof(hittable *)));
    hittable **d_world;            checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hittable *)));

    const float scale = 4.0f;      // tweak to taste (like the book)
    create_world_perlin<<<1,1>>>(d_list, d_world, d_camera, nx, ny, scale);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    dim3 blocks(nx/tx+1, ny/ty+1), threads(tx, ty);
    render_init<<<blocks, threads>>>(nx, ny, d_rand_state);
    checkCudaErrors(cudaDeviceSynchronize());
    render<<<blocks, threads>>>(fb, nx, ny, ns, gamma, d_camera, d_world, d_rand_state, vec3(0,0,0), 1);
    checkCudaErrors(cudaDeviceSynchronize());

    std::cout << "P3\n" << nx << " " << ny << "\n255\n";
    for (int j = ny-1; j >= 0; --j) for (int i = 0; i < nx; ++i) {
        size_t k = j*nx + i;
        int ir = int(255.99f*fb[k].r());
        int ig = int(255.99f*fb[k].g());
        int ib = int(255.99f*fb[k].b());
        std::cout << ir << " " << ig << " " << ib << "\n";
    }

    // --- Clean up ---
    checkCudaErrors(cudaDeviceSynchronize());                 // make sure render finished

    free_world<<<1,1>>>(d_list, num_hitables, d_world, d_camera);   // NOTE: count is 2nd arg
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());                 // wait for device-side deletes

    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(fb));
    cudaDeviceReset();
    return 0;
}

int quads_scene() {
    int nx = 1200, ny = 600, ns = 500;
    float gamma = 2.2f;
    int tx = 8, ty = 8;

    cudaDeviceSetLimit(cudaLimitStackSize,      16384);
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 64*1024*1024);

    int num_pixels = nx * ny;
    size_t fb_size = num_pixels * sizeof(vec3);

    vec3 *fb;                   checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));
    curandState *d_rand_state;  checkCudaErrors(cudaMalloc((void**)&d_rand_state,  num_pixels*sizeof(curandState)));
    curandState *d_rand_state2; checkCudaErrors(cudaMalloc((void**)&d_rand_state2, 1*sizeof(curandState)));
    rand_init<<<1,1>>>(d_rand_state2);

    camera **d_camera; checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(camera*)));
    int num_hitables = 5;
    hittable **d_list; checkCudaErrors(cudaMalloc((void**)&d_list,  num_hitables*sizeof(hittable*)));
    hittable **d_world;checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(hittable*)));

    create_world_quads<<<1,1>>>(d_list, d_world, d_camera, nx, ny);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    dim3 blocks(nx/tx+1, ny/ty+1), threads(tx, ty);
    render_init<<<blocks, threads>>>(nx, ny, d_rand_state);
    checkCudaErrors(cudaDeviceSynchronize());
    render<<<blocks, threads>>>(fb, nx, ny, ns, gamma, d_camera, d_world, d_rand_state, vec3(0,0,0), 1);
    checkCudaErrors(cudaDeviceSynchronize());

    std::cout << "P3\n" << nx << " " << ny << "\n255\n";
    for (int j = ny-1; j >= 0; --j) for (int i = 0; i < nx; ++i) {
        size_t k = j*nx + i;
        int ir = int(255.99f*fb[k].r());
        int ig = int(255.99f*fb[k].g());
        int ib = int(255.99f*fb[k].b());
        std::cout << ir << " " << ig << " " << ib << "\n";
    }

    // --- Clean up ---
    checkCudaErrors(cudaDeviceSynchronize());                 // make sure render finished

    free_world<<<1,1>>>(d_list, num_hitables, d_world, d_camera);   // NOTE: count is 2nd arg
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());                 // wait for device-side deletes

    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(fb));
    cudaDeviceReset();
    return 0;
}

int simple_light() 
{
    // image / render params
    int nx = 1200;
    int ny = 600;
    int ns = 10000;
    float gamma = 2.2f;
    int tx = 8, ty = 8;

    // device limits (same as your other scenes)
    cudaDeviceSetLimit(cudaLimitStackSize,      16384);
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 64*1024*1024);

    // Load ball texture on HOST, upload to device
    unsigned char* d_pixels = nullptr;
    DeviceImage ball_img   = load_image_to_device("textures/poolball.jpg", &d_pixels);
    if (!ball_img.valid()) {
        std::cerr << "Failed to load poolball.jpg\n";
        return 1;
    }

    // frame buffer
    int num_pixels = nx * ny;
    size_t fb_size = num_pixels * sizeof(vec3);
    vec3 *fb; checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));

    // RNG
    curandState *d_rand_state;  checkCudaErrors(cudaMalloc((void**)&d_rand_state,  num_pixels*sizeof(curandState)));
    curandState *d_rand_state2; checkCudaErrors(cudaMalloc((void**)&d_rand_state2, sizeof(curandState)));
    rand_init<<<1,1>>>(d_rand_state2);

    // scene storage
    camera   **d_camera; checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(camera*)));
    const int num_hitables = 4; // ground + gray sphere + light sphere + light quad
    hittable **d_list;   checkCudaErrors(cudaMalloc((void**)&d_list,   num_hitables * sizeof(hittable*)));
    hittable **d_world;  checkCudaErrors(cudaMalloc((void**)&d_world,  sizeof(hittable*)));

    // build the scene on device
    create_world_simple_light<<<1,1>>>(d_list, d_world, d_camera, nx, ny, ball_img);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // render
    dim3 blocks(nx/tx+1, ny/ty+1), threads(tx,ty);
    render_init<<<blocks, threads>>>(nx, ny, d_rand_state);              checkCudaErrors(cudaDeviceSynchronize());
    render     <<<blocks, threads>>>(fb, nx, ny, ns, gamma, d_camera, d_world, d_rand_state,
                                     /*background*/ vec3(0,0,0), /*use_gradient_bg*/ 0);
    checkCudaErrors(cudaDeviceSynchronize());

    // output PPM
    std::cout << "P3\n" << nx << " " << ny << "\n255\n";
    for (int j = ny-1; j >= 0; --j) {
        for (int i = 0; i < nx; ++i) {
            const size_t k = j*nx + i;
            int ir = int(255.99f*fb[k].r());
            int ig = int(255.99f*fb[k].g());
            int ib = int(255.99f*fb[k].b());
            std::cout << ir << ' ' << ig << ' ' << ib << '\n';
        }
    }

    // --- Clean up ---
    checkCudaErrors(cudaDeviceSynchronize());                 // make sure render finished

    free_world<<<1,1>>>(d_list, num_hitables, d_world, d_camera);   // NOTE: count is 2nd arg
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());                 // wait for device-side deletes

    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(fb));
    cudaDeviceReset();
    return 0;
}

int cornell_box() 
{
    int nx = 600, ny = 600, ns = 10000;
    float gamma = 2.2f;
    int tx = 8, ty = 8;

    cudaDeviceSetLimit(cudaLimitStackSize,      16384);
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 64*1024*1024);

    int num_pixels = nx * ny;
    size_t fb_size = num_pixels * sizeof(vec3);
    vec3 *fb; checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));

    curandState *d_rand_state;  checkCudaErrors(cudaMalloc((void**)&d_rand_state,  num_pixels*sizeof(curandState)));
    curandState *d_rand_state2; checkCudaErrors(cudaMalloc((void**)&d_rand_state2, sizeof(curandState)));
    rand_init<<<1,1>>>(d_rand_state2);

    camera **d_camera; checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(camera*)));
    const int num_hitables = 6; // the 6 Cornell quads shown
    hittable **d_list;  checkCudaErrors(cudaMalloc((void**)&d_list,  num_hitables * sizeof(hittable*)));
    hittable **d_world; checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(hittable*)));

    create_world_cornell<<<1,1>>>(d_list, d_world, d_camera, nx, ny);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    dim3 blocks(nx/tx+1, ny/ty+1), threads(tx,ty);
    render_init<<<blocks, threads>>>(nx, ny, d_rand_state);              checkCudaErrors(cudaDeviceSynchronize());
    render     <<<blocks, threads>>>(fb, nx, ny, ns, gamma, d_camera, d_world, d_rand_state,
                                     /*background*/ vec3(0,0,0), /*use_gradient_bg*/ 0);
    checkCudaErrors(cudaDeviceSynchronize());

    std::cout << "P3\n" << nx << " " << ny << "\n255\n";
    for (int j = ny-1; j >= 0; --j) for (int i = 0; i < nx; ++i) {
        size_t k = j*nx + i;
        int ir = int(255.99f*fb[k].r());
        int ig = int(255.99f*fb[k].g());
        int ib = int(255.99f*fb[k].b());
        std::cout << ir << ' ' << ig << ' ' << ib << '\n';
    }

    // --- Clean up ---
    checkCudaErrors(cudaDeviceSynchronize());                 // make sure render finished

    free_world<<<1,1>>>(d_list, num_hitables, d_world, d_camera);   // NOTE: count is 2nd arg
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());                 // wait for device-side deletes

    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(fb));
    cudaDeviceReset();
    return 0;
}

void cornell_smoke() {
    int nx=600, ny=600, ns=1000; float gamma=2.2f; int tx=8, ty=8;

    cudaDeviceSetLimit(cudaLimitStackSize,      65536);           // was 16384
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 256*1024*1024);   // was 64 MB

    // Framebuffer + RNGs
    int num_pixels = nx*ny;
    size_t fb_size = num_pixels * sizeof(vec3);
    vec3 *fb; checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));
    curandState *d_rand_state;  checkCudaErrors(cudaMalloc((void **)&d_rand_state,  num_pixels*sizeof(curandState)));
    curandState *d_rand_state2; checkCudaErrors(cudaMalloc((void **)&d_rand_state2, 1*sizeof(curandState)));
    rand_init<<<1,1>>>(d_rand_state2);

    // Scene allocations
    camera **d_camera; checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(camera *)));
    // 5 walls + light + 2 media = 8 objects
    int num_hitables = 8;
    hittable **d_list;  checkCudaErrors(cudaMalloc((void **)&d_list,  num_hitables * sizeof(hittable *)));
    checkCudaErrors(cudaMemset(d_list, 0, num_hitables * sizeof(hittable*)));
    hittable **d_world; checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hittable *)));

    create_world_cornell_smoke<<<1,1>>>(d_list, d_world, d_camera, nx, ny);
    checkCudaErrors(cudaDeviceSynchronize());

    dim3 blocks(nx/tx+1, ny/ty+1), threads(tx,ty);
    render_init<<<blocks, threads>>>(nx, ny, d_rand_state);
    render<<<blocks, threads>>>(fb, nx, ny, ns, gamma, d_camera, d_world, d_rand_state, vec3(0,0,0), 0);
    checkCudaErrors(cudaDeviceSynchronize());

    // Output PPM to stdout (matches your other scenes)
    std::cout << "P3\n" << nx << " " << ny << "\n255\n";
    for (int j = ny - 1; j >= 0; --j) {
        for (int i = 0; i < nx; ++i) {
            const size_t k = j * nx + i;
            int ir = int(255.99f * fb[k].r());
            int ig = int(255.99f * fb[k].g());
            int ib = int(255.99f * fb[k].b());
            std::cout << ir << ' ' << ig << ' ' << ib << '\n';
        }
    }

    free_world<<<1,1>>>(d_list, num_hitables, d_world, d_camera);
    checkCudaErrors(cudaDeviceSynchronize());
    cudaFree(d_rand_state); cudaFree(d_rand_state2); cudaFree(fb);
    cudaFree(d_list); cudaFree(d_world); cudaFree(d_camera);
    cudaDeviceReset();
}

void final_scene() {
    int nx=800, ny=800, ns=10000; float gamma=2.2f; int tx=8, ty=8;

    cudaDeviceSetLimit(cudaLimitStackSize,      32768);          // 32 KB
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 256*1024*1024);  // 256 MB

    // Load earth texture to device (same as your earth() scene)
    unsigned char* d_pixels = nullptr;
    DeviceImage earth_img = load_image_to_device("textures/earthmap.jpg", &d_pixels);
    earth_img = load_image_to_device("textures/earthmap.jpg", &d_pixels);
    if (!earth_img.valid()) { std::cerr << "Failed to load earthmap.jpg\n"; return; }

    // Framebuffer + RNGs
    int num_pixels = nx*ny; size_t fb_size = num_pixels*sizeof(vec3);
    vec3 *fb; checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));
    curandState *d_rand_state;  checkCudaErrors(cudaMalloc((void **)&d_rand_state,  num_pixels*sizeof(curandState)));
    curandState *d_rand_state2; checkCudaErrors(cudaMalloc((void **)&d_rand_state2, 1*sizeof(curandState)));
    rand_init<<<1,1>>>(d_rand_state2);

    // Scene allocations (upper bound ~ 1600 leaves)
    int num_hitables = 1800;
    hittable **d_list;  checkCudaErrors(cudaMalloc((void **)&d_list,  num_hitables*sizeof(hittable *)));
    hittable **d_world; checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hittable *)));
    camera   **d_camera;checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(camera *)));

    create_world_final<<<1,1>>>(d_list, d_world, d_camera, nx, ny, earth_img);
    checkCudaErrors(cudaDeviceSynchronize());

    dim3 blocks(nx/tx+1, ny/ty+1), threads(tx,ty);
    render_init<<<blocks, threads>>>(nx, ny, d_rand_state);
    render<<<blocks, threads>>>(fb, nx, ny, ns, gamma, d_camera, d_world, d_rand_state, vec3(0,0,0), 0);
    checkCudaErrors(cudaDeviceSynchronize());

    // Output PPM to stdout (matches your other scenes)
    std::cout << "P3\n" << nx << " " << ny << "\n255\n";
    for (int j = ny - 1; j >= 0; --j) {
        for (int i = 0; i < nx; ++i) {
            const size_t k = j * nx + i;
            int ir = int(255.99f * fb[k].r());
            int ig = int(255.99f * fb[k].g());
            int ib = int(255.99f * fb[k].b());
            std::cout << ir << ' ' << ig << ' ' << ib << '\n';
        }
    }

    free_world<<<1,1>>>(d_list, num_hitables, d_world, d_camera);
    checkCudaErrors(cudaDeviceSynchronize());

    cudaFree(d_rand_state); 
    cudaFree(d_rand_state2); 
    cudaFree(fb);
    cudaFree(d_list); 
    cudaFree(d_world); 
    cudaFree(d_camera);

    // Free device pixels for the earth texture (same as earth())
    free_device_image(earth_img);

    cudaDeviceReset();
}

void original_scene() 
{
    int nx=800, ny=800, ns=10000; float gamma=2.2f; int tx=8, ty=8;

    cudaDeviceSetLimit(cudaLimitStackSize,      32768);          // 32 KB
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 256*1024*1024);  // 256 MB

    // Load textures to device (same as your earth() scene)
    unsigned char* d_pixels = nullptr;
    DeviceImage earth_img = load_image_to_device("textures/porcelain.jpg", &d_pixels);
    earth_img = load_image_to_device("textures/porcelain.jpg", &d_pixels);
    if (!earth_img.valid()) { std::cerr << "Failed to load porcelain.jpg\n"; return; }

    // Load textures to device (same as your earth() scene)
    d_pixels = nullptr;
    DeviceImage ball_img = load_image_to_device("textures/8ball.jpg", &d_pixels);
    ball_img = load_image_to_device("textures/8ball.jpg", &d_pixels);
    if (!ball_img.valid()) { std::cerr << "Failed to load 8ball.jpg\n"; return; }

    // Framebuffer + RNGs
    int num_pixels = nx*ny; size_t fb_size = num_pixels*sizeof(vec3);
    vec3 *fb; checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));
    curandState *d_rand_state;  checkCudaErrors(cudaMalloc((void **)&d_rand_state,  num_pixels*sizeof(curandState)));
    curandState *d_rand_state2; checkCudaErrors(cudaMalloc((void **)&d_rand_state2, 1*sizeof(curandState)));
    rand_init<<<1,1>>>(d_rand_state2);

    // Scene allocations (upper bound ~ 1600 leaves)
    int num_hitables = 1800;
    hittable **d_list;  checkCudaErrors(cudaMalloc((void **)&d_list,  num_hitables*sizeof(hittable *)));
    hittable **d_world; checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hittable *)));
    camera   **d_camera;checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(camera *)));

    create_world_original<<<1,1>>>(d_list, d_world, d_camera, nx, ny, earth_img, ball_img);
    checkCudaErrors(cudaDeviceSynchronize());

    dim3 blocks(nx/tx+1, ny/ty+1), threads(tx,ty);
    render_init<<<blocks, threads>>>(nx, ny, d_rand_state);
    render<<<blocks, threads>>>(fb, nx, ny, ns, gamma, d_camera, d_world, d_rand_state, vec3(0.043f, 0.030f, 0.094f), 0);
    checkCudaErrors(cudaDeviceSynchronize());

    // Output PPM to stdout (matches your other scenes)
    std::cout << "P3\n" << nx << " " << ny << "\n255\n";
    for (int j = ny - 1; j >= 0; --j) {
        for (int i = 0; i < nx; ++i) {
            const size_t k = j * nx + i;
            int ir = int(255.99f * fb[k].r());
            int ig = int(255.99f * fb[k].g());
            int ib = int(255.99f * fb[k].b());
            std::cout << ir << ' ' << ig << ' ' << ib << '\n';
        }
    }

    free_world<<<1,1>>>(d_list, num_hitables, d_world, d_camera);
    checkCudaErrors(cudaDeviceSynchronize());

    cudaFree(d_rand_state); 
    cudaFree(d_rand_state2); 
    cudaFree(fb);
    cudaFree(d_list); 
    cudaFree(d_world); 
    cudaFree(d_camera);

    // Free device pixels for the earth texture (same as earth())
    free_device_image(earth_img);

    cudaDeviceReset();
}

int main() 
{
    switch (10) 
    {
        case 1: bouncing_spheres();
        case 2: checkered_spheres();
        case 3: earth();
        case 4: perlin();
        case 5: quads_scene();
        case 6: simple_light();
        case 7: cornell_box();
        case 8: cornell_smoke();  break;
        case 9: final_scene();    break;
        case 10: original_scene();
    }
}
