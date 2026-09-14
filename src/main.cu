#define STB_IMAGE_IMPLEMENTATION

#include <curand_kernel.h>
#include <cfloat>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// core/ is dual-compiled pure math and POD; scene/ is the device-only material
// hierarchy plus the GPU upload; host/ builds scenes on the CPU.
#include "core/camera.hpp"
#include "core/primitives.hpp"
#include "core/scene_view.hpp"
#include "core/vec3.hpp"

#include "io/image_io.hpp"

#include "host/scenes.hpp"

#include "scene/device_scene.cuh"
#include "scene/material.cuh"

#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)
static void check_cuda(cudaError_t result, char const* const func,
                       const char* const file, int const line)
{
    if (result) {
        std::fprintf(stderr, "CUDA error %d (%s) at %s:%d '%s'\n",
                     (int)result, cudaGetErrorString(result), file, line, func);
        cudaDeviceReset();
        std::exit(99);
    }
}

// ===========================================================================
// Kernels
// ===========================================================================

// Adapts cuRAND to the `float next()` interface the dual-compiled traversal and
// camera expect. Inlines away completely.
struct CurandRng {
    curandState* s;
    __device__ explicit CurandRng(curandState* st) : s(st) {}
    __device__ float next() { return curand_uniform(s); }
};

__device__ inline float apply_gamma(float c, float gamma)
{
    if (gamma == 1.0f) return c;
    return powf(fmaxf(c, 0.0f), 1.0f / gamma);
}

// The book's recursive ray_color, flattened into a loop with an explicit
// throughput term so the device never needs a deep stack.
__device__ vec3 trace(const ray& r0, const vec3& background, bool gradient_bg,
                      int max_depth, const SceneView& scene,
                      material* const* mats, CurandRng& rng)
{
    ray  cur_ray    = r0;
    vec3 throughput = vec3(1, 1, 1);
    vec3 radiance   = vec3(0, 0, 0);

    for (int bounce = 0; bounce < max_depth; ++bounce) {
        Hit rec;
        if (!scene_intersect(scene, cur_ray, 0.001f, FLT_MAX, rec, rng)) {
            vec3 bg = background;
            if (gradient_bg) {
                const vec3 unit_direction = unit_vector(cur_ray.direction());
                const float t = 0.5f * (unit_direction.y() + 1.0f);
                bg = (1.0f - t) * vec3(1.0f, 1.0f, 1.0f) + t * vec3(0.5f, 0.7f, 1.0f);
            }
            radiance += throughput * bg;
            break;
        }

        const material* m = mats[rec.mat];
        radiance += throughput * m->emitted(rec.u, rec.v, rec.p);

        ray  scattered;
        vec3 attenuation;
        if (!m->scatter(cur_ray, rec, attenuation, scattered, rng.s)) break;

        throughput *= attenuation;
        cur_ray = scattered;
    }

    return radiance;
}

__global__ void render_init(int max_x, int max_y, curandState* rand_state,
                            unsigned long long seed)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= max_x || j >= max_y) return;
    const int pixel_index = j * max_x + i;

    // Distinct seed per pixel rather than distinct sequence: curand_init with a
    // sequence number does a 2^67 skip-ahead, which is far more expensive. The
    // streams are statistically independent, not provably so.
    curand_init(seed + pixel_index, 0, 0, &rand_state[pixel_index]);
}

// Accumulates `ns` samples without normalising, so the host can issue the
// sample budget in batches -- several short launches instead of one multi-minute
// kernel that Windows' display watchdog (TDR, 2 s) would kill.
__global__ void render_accumulate(vec3* __restrict__ accum, int max_x, int max_y,
                                  int ns, int max_depth, Camera cam, SceneView scene,
                                  material* const* __restrict__ mats,
                                  curandState* __restrict__ rand_state,
                                  vec3 background, int use_gradient_bg)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= max_x || j >= max_y) return;

    const int pixel_index = j * max_x + i;
    curandState local_state = rand_state[pixel_index];
    CurandRng rng(&local_state);

    vec3 col(0, 0, 0);
    for (int s = 0; s < ns; ++s) {
        const float u = float(i + rng.next()) / float(max_x);
        const float v = float(j + rng.next()) / float(max_y);
        const ray r = cam.get_ray(u, v, rng);
        col += trace(r, background, use_gradient_bg != 0, max_depth, scene, mats, rng);
    }
    rand_state[pixel_index] = local_state;

    accum[pixel_index] += col;
}

__global__ void resolve(vec3* __restrict__ fb, const vec3* __restrict__ accum,
                        int max_x, int max_y, int total_samples, float gamma)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= max_x || j >= max_y) return;

    const int pixel_index = j * max_x + i;
    vec3 col = accum[pixel_index] / float(total_samples);
    col[0] = apply_gamma(col[0], gamma);
    col[1] = apply_gamma(col[1], gamma);
    col[2] = apply_gamma(col[2], gamma);
    fb[pixel_index] = col;
}

__global__ void clear_buffer(vec3* buf, int n)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) buf[i] = vec3(0, 0, 0);
}

// ===========================================================================
// Command line
// ===========================================================================

struct Options {
    int          scene       = rt::SCENE_ORIGINAL;
    int          width       = -1;      // -1 -> the scene's default
    int          height      = -1;
    int          spp         = -1;
    int          max_depth   = 50;
    float        gamma       = 2.2f;
    unsigned long long seed  = 1984;
    int          tx          = 8;
    int          ty          = 8;
    int          batch       = 64;
    std::string  out;                   // empty -> stdout
    std::string  stats;
    bool         binary      = true;    // P6; --ascii for P3
    bool         quiet       = false;
    bool         list        = false;
};

static void usage(const char* argv0) {
    std::fprintf(stderr,
      "usage: %s [options]\n\n"
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
        if      (a == "--list")  o->list = true;
        else if (a == "--quiet") o->quiet = true;
        else if (a == "--ascii") o->binary = false;
        else if (a == "-h" || a == "--help") { usage(argv[0]); return false; }
        else if (a == "--scene") {
            if (!need(++i, "--scene")) return false;
            bool found = false;
            for (int s = 0; s < rt::SCENE_COUNT; ++s)
                if (std::strcmp(rt::scene_table()[s].name, argv[i]) == 0) { o->scene = s; found = true; break; }
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

// P6 by default. The original wrote P3 with one iostream insertion per channel:
// about 2.1M formatted writes for a 1200x600 frame, which took longer than some
// of the renders.
static bool write_ppm(const char* path, bool binary, const vec3* fb, int nx, int ny)
{
    std::FILE* f = path ? std::fopen(path, binary ? "wb" : "w") : stdout;
    if (!f) { std::fprintf(stderr, "error: cannot open '%s' for writing\n", path); return false; }

    auto quantise = [](float c) -> unsigned char {
        const int v = (int)(255.99f * c);
        return (unsigned char)(v < 0 ? 0 : (v > 255 ? 255 : v));
    };

    // Row 0 of the framebuffer is the bottom of the image; PPM runs top-down.
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
    if (path) std::fclose(f); else std::fflush(f);
    return ok;
}

// ===========================================================================
// Driver
// ===========================================================================

static int run_scene(const rt::SceneInfo& info, const Options& opt)
{
    const int nx = opt.width  > 0 ? opt.width  : info.width;
    const int ny = opt.height > 0 ? opt.height : info.height;
    const int ns = opt.spp    > 0 ? opt.spp    : info.spp;

    // --- build on the host --------------------------------------------------
    const auto t_build0 = std::chrono::steady_clock::now();
    rt::SceneSetup setup;
    rt::build_scene(info.id, (unsigned int)opt.seed, setup);
    const auto t_build1 = std::chrono::steady_clock::now();
    const double build_ms =
        std::chrono::duration<double, std::milli>(t_build1 - t_build0).count();

    // --- textures -----------------------------------------------------------
    std::vector<DeviceImage> images;
    for (const std::string& path : setup.images) {
        DeviceImage img = load_image_to_device(path.c_str());
        if (!img.valid()) {
            std::fprintf(stderr, "error: failed to load %s\n", path.c_str());
            for (DeviceImage& prev : images) free_device_image(prev);
            return 1;
        }
        images.push_back(img);
    }

    // --- upload -------------------------------------------------------------
    const auto t_up0 = std::chrono::steady_clock::now();
    DeviceScene scene = upload_scene(setup.world, images);
    if (!scene.ok) { free_scene(scene); return 2; }
    const auto t_up1 = std::chrono::steady_clock::now();
    const double upload_ms =
        std::chrono::duration<double, std::milli>(t_up1 - t_up0).count();

    if (!opt.quiet) {
        std::fprintf(stderr,
            "scene '%s': %d primitives (%zu spheres, %zu quads, %zu media), "
            "%d materials, %zu BVH nodes\n",
            info.name, setup.world.primitive_count(),
            setup.world.spheres().size(), setup.world.quads().size(),
            setup.world.media().size(), scene.n_mat, setup.world.nodes().size());
        std::fprintf(stderr, "  %dx%d, %d spp, depth %d\n", nx, ny, ns, opt.max_depth);
    }

    // --- allocations --------------------------------------------------------
    const int    num_pixels = nx * ny;
    const size_t fb_bytes   = (size_t)num_pixels * sizeof(vec3);

    vec3*        fb = nullptr;
    vec3*        accum = nullptr;
    curandState* d_rand_state = nullptr;

    checkCudaErrors(cudaMallocManaged((void**)&fb, fb_bytes));
    checkCudaErrors(cudaMalloc((void**)&accum, fb_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_rand_state,
                               (size_t)num_pixels * sizeof(curandState)));

    const dim3 blocks((nx + opt.tx - 1) / opt.tx, (ny + opt.ty - 1) / opt.ty);
    const dim3 threads(opt.tx, opt.ty);

    // The camera is a POD built on the host and passed to the kernel by value,
    // so there is no allocation and no per-ray pointer chase for it.
    const Camera cam = setup.cam.build(float(nx) / float(ny));

    // --- render -------------------------------------------------------------
    cudaEvent_t ev0, ev1;
    checkCudaErrors(cudaEventCreate(&ev0));
    checkCudaErrors(cudaEventCreate(&ev1));
    checkCudaErrors(cudaEventRecord(ev0));

    clear_buffer<<<(num_pixels + 255) / 256, 256>>>(accum, num_pixels);
    render_init<<<blocks, threads>>>(nx, ny, d_rand_state, opt.seed);
    checkCudaErrors(cudaGetLastError());

    const int batch = (opt.batch > 0 && opt.batch < ns) ? opt.batch : ns;
    for (int done = 0; done < ns; done += batch) {
        const int n = (done + batch <= ns) ? batch : (ns - done);
        render_accumulate<<<blocks, threads>>>(accum, nx, ny, n, opt.max_depth,
                                               cam, scene.view, scene.materials,
                                               d_rand_state, setup.background,
                                               setup.gradient_bg ? 1 : 0);
        checkCudaErrors(cudaGetLastError());
        if (!opt.quiet) {
            checkCudaErrors(cudaDeviceSynchronize());
            std::fprintf(stderr, "\r  %d/%d samples (%.0f%%)   ",
                         done + n, ns, 100.0 * (done + n) / ns);
        }
    }
    resolve<<<blocks, threads>>>(fb, accum, nx, ny, ns, opt.gamma);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaEventRecord(ev1));
    checkCudaErrors(cudaDeviceSynchronize());
    if (!opt.quiet) std::fprintf(stderr, "\n");

    float render_ms = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&render_ms, ev0, ev1));

    if (!opt.quiet) {
        std::fprintf(stderr,
            "build %.1f ms (host) + upload %.1f ms, render %.3f s (%.2f Mpaths/s)\n",
            build_ms, upload_ms, render_ms / 1000.0,
            ((double)num_pixels * ns) / (render_ms * 1000.0));
    }

    // --- output -------------------------------------------------------------
    const bool wrote = write_ppm(opt.out.empty() ? nullptr : opt.out.c_str(),
                                 opt.binary, fb, nx, ny);
    if (!wrote) std::fprintf(stderr, "error: failed writing image\n");

    if (!opt.stats.empty()) {
        if (std::FILE* sf = std::fopen(opt.stats.c_str(), "w")) {
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
                "  \"primitives\": %d,\n"
                "  \"bvh_nodes\": %zu,\n"
                "  \"materials\": %d,\n"
                "  \"block\": [%d, %d],\n"
                "  \"batch\": %d,\n"
                "  \"build_ms\": %.4f,\n"
                "  \"upload_ms\": %.4f,\n"
                "  \"render_ms\": %.4f,\n"
                "  \"primary_rays\": %lld\n"
                "}\n",
                prop.name, info.name, nx, ny, ns, opt.max_depth, opt.seed,
                setup.world.primitive_count(), setup.world.nodes().size(),
                scene.n_mat, opt.tx, opt.ty, batch,
                build_ms, upload_ms, (double)render_ms,
                (long long)num_pixels * ns);
            std::fclose(sf);
        } else {
            std::fprintf(stderr, "warning: cannot write stats to '%s'\n", opt.stats.c_str());
        }
    }

    // --- teardown -----------------------------------------------------------
    checkCudaErrors(cudaEventDestroy(ev0));
    checkCudaErrors(cudaEventDestroy(ev1));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(accum));
    checkCudaErrors(cudaFree(fb));
    free_scene(scene);

    return wrote ? 0 : 1;
}

int main(int argc, char** argv)
{
    // Lets scenes find textures/ whether the binary is launched from the build
    // root, from bin/Release, or from a script somewhere else.
    set_asset_search_root(argc > 0 ? argv[0] : nullptr);

    Options opt;
    if (!parse_args(argc, argv, &opt)) return 1;

    if (opt.list) {
        std::printf("%-16s %9s %7s\n", "scene", "size", "spp");
        for (int i = 0; i < rt::SCENE_COUNT; ++i) {
            const rt::SceneInfo& s = rt::scene_table()[i];
            std::printf("%-16s %4dx%-4d %7d\n", s.name, s.width, s.height, s.spp);
        }
        return 0;
    }

    const int rc = run_scene(rt::scene_table()[opt.scene], opt);

    // Makes leak checking under compute-sanitizer meaningful: the driver
    // reports anything still allocated at reset.
    cudaDeviceReset();
    return rc;
}
