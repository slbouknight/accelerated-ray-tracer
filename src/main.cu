#define STB_IMAGE_IMPLEMENTATION
#include <curand_kernel.h>
#include <float.h>
#include <iostream>
#include <math.h>
#include <math_constants.h>
#include <time.h>

#include "bvh.cuh"
#include "camera.cuh"
#include "hittable_list.cuh"
#include "image_io.h"
#include "material.cuh"
#include "ray.cuh"
#include "sphere.cuh"
#include "texture.cuh"
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

__device__ vec3 color(const ray& r, hittable **world, curandState *local_rand_state)
{
    ray cur_ray = r;
    vec3 cur_attenuation = vec3(1.0, 1.0, 1.0);
    for(int i = 0; i < 50; i++)
    {
        hit_record rec;
        if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec))
        {
            ray scattered;
            vec3 attenuation;
            if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state))
            {
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            }
            else
            {
                return vec3(0.0, 0.0, 0.0);
            }
        }
        else
        {
            vec3 unit_direction = unit_vector(cur_ray.direction());
            float t = 0.5f*(unit_direction.y() + 1.0f);
            vec3 c = (1.0f-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
            return cur_attenuation * c;
        }
    }
    return vec3(0.0, 0.0, 0.0);
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

__global__ void render(vec3 *fb, int max_x, int max_y, int ns, float gamma, camera **cam, hittable **world, curandState *rand_state)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j*max_x + i;
    curandState local_rand_state = rand_state[pixel_index];
    vec3 col(0, 0, 0);
    for(int s=0; s < ns; s++)
    {
        float u = float( i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
        ray r = (*cam)->get_ray(u, v, &local_rand_state);
        col += color(r, world, &local_rand_state);
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

__global__ void create_world_bouncing(hittable **d_list, hittable **d_world, camera **d_camera,
                             int nx, int ny, curandState *rand_state)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandState local_rand_state = *rand_state;
        int i = 0;

        // Ground (checkered)
        texture* checker = new checker_texture(0.32f,
            new solid_color(vec3(0.2f, 0.3f, 0.1f)),
            new solid_color(vec3(0.9f, 0.9f, 0.9f)));

        d_list[i++] = new sphere(vec3(0.0f, -1000.0f, -1.0f), 1000.0f,
                                new lambertian(checker));


        // Random small spheres on a grid
        for (int a = GRID_MIN; a < GRID_MAX; a++) {
            for (int b = GRID_MIN; b < GRID_MAX; b++) {
                float choose_mat = RND;
                vec3 center(a + 0.9f * RND, 0.2f, b + 0.9f * RND); // 0.9 like the book

                if (choose_mat < 0.8f) {
                    // Diffuse — MOVING
                    vec3 albedo(RND*RND, RND*RND, RND*RND);

                    // random velocity in each axis
                    vec3 vel(0.0f, 0.5f*RND, 0.25f*(RND - 0.5f));
                    vec3 center2 = center + vel;
                    d_list[i++] = new sphere(center, center2, 0.2f, new lambertian(albedo));
                } else if (choose_mat < 0.95f) {
                    // Metal with fuzz (static)
                    vec3 albedo(0.5f*(1.0f+RND), 0.5f*(1.0f+RND), 0.5f*(1.0f+RND));
                    float fuzz = 0.5f * RND;
                    d_list[i++] = new sphere(center, 0.2f, new metal(albedo, fuzz));
                } else {
                    // Dielectric (static)
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

__global__ void free_world(hittable **d_list, hittable **d_world, camera **d_camera, int num_objects)
{
    for (int i = 0; i < num_objects; i++) {
        delete ((sphere*)d_list[i])->mat_ptr;
        delete d_list[i];
    }
    delete *d_world;
    delete *d_camera;
}

void bouncing_spheres()
{
    int nx = 1200;
    int ny = 600;
    int ns = 500;
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
    render<<<blocks, threads>>>(fb, nx, ny,  ns, gamma, d_camera, d_world, d_rand_state);
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
    checkCudaErrors(cudaDeviceSynchronize());
    free_world<<<1,1>>>(d_list,d_world,d_camera, num_hitables);
    checkCudaErrors(cudaGetLastError());
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
    render<<<blocks, threads>>>(fb, nx, ny, ns, gamma, d_camera, d_world, d_rand_state);
    checkCudaErrors(cudaDeviceSynchronize());

    std::cout << "P3\n" << nx << " " << ny << "\n255\n";
    for (int j = ny-1; j >= 0; --j) for (int i = 0; i < nx; ++i) {
        size_t k = j*nx + i;
        int ir = int(255.99f*fb[k].r());
        int ig = int(255.99f*fb[k].g());
        int ib = int(255.99f*fb[k].b());
        std::cout << ir << " " << ig << " " << ib << "\n";
    }

    free_world<<<1,1>>>(d_list, d_world, d_camera, /*num_objects=*/num_hitables);
    checkCudaErrors(cudaDeviceSynchronize());
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
    render<<<blocks, threads>>>(fb, nx, ny, ns, gamma, d_camera, d_world, d_rand_state);
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
    free_world<<<1,1>>>(d_list, d_world, d_camera, /*num_objects=*/num_hitables);
    checkCudaErrors(cudaDeviceSynchronize());
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

int main() {
    switch (3) 
    {                  // 1 = bouncing, 2 = checkered, 3 = earth
        case 1: bouncing_spheres();
        case 2: checkered_spheres();
        case 3: earth();
    }
}
