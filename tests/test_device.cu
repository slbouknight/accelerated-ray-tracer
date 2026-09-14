// Device-side tests: materials, textures, and the material table.
//
// This file used to cover geometry, instancing, BVH traversal and the camera as
// well -- all of which needed a GPU because they were `__device__` virtuals
// built with device `new`. Those are now POD and live in test_geometry.cpp,
// which runs on the host.
//
// What is left is genuinely device-only: materials and textures are still a
// virtual hierarchy, and a vtable pointer is a device address, so they can only
// be constructed and called on the device. Shading is also where the book's
// class structure earns its keep and there are tens of them per scene rather
// than thousands, so the indirect call is not on the critical path.
//
// Pattern: a single-thread kernel computes and writes primitive floats into a
// managed buffer; the host asserts on them with the same harness as the host
// tests, so a failure prints real values and a line number instead of a trap.

#include "test_harness.h"

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cfloat>

#include "../src/core/primitives.hpp"
#include "../src/core/vec3.hpp"
#include "../src/host/material_desc.hpp"
#include "../src/scene/material.cuh"
#include "../src/scene/material_table.cuh"
#include "../src/scene/texture.cuh"

namespace {

constexpr float kEps = 1e-4f;

// Managed scratch, prefilled with NaN so a slot the kernel forgets to write
// produces a failing assertion rather than a stale zero that looks plausible.
class Scratch {
public:
    explicit Scratch(int n) {
        if (cudaMallocManaged(&d_, sizeof(float) * n) != cudaSuccess) d_ = nullptr;
        if (d_) for (int i = 0; i < n; ++i) d_[i] = NAN;
    }
    ~Scratch() { if (d_) cudaFree(d_); }
    Scratch(const Scratch&) = delete;
    Scratch& operator=(const Scratch&) = delete;

    float* data() const { return d_; }
    bool   ok()   const { return d_ != nullptr; }
    float  operator[](int i) const { return d_[i]; }
    vec3   v(int i) const { return vec3(d_[i], d_[i + 1], d_[i + 2]); }

private:
    float* d_ = nullptr;
};

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
// Materials
// =============================================================================

// out: [0]=scattered? [1..3]=unit dir [4..6]=attenuation
//      [7]=dielectric scattered? [8..10]=dielectric attenuation
//      [11]=light scatters? [12..14]=emitted
__global__ void k_materials(float* out) {
    curandState rng;
    curand_init(7u, 0, 0, &rng);

    Hit rec;
    rec.p = vec3(0, 0, 0);
    rec.normal = vec3(0, 1, 0);
    rec.u = rec.v = 0.0f;
    rec.mat = 0;

    // Zero-fuzz metal is a perfect mirror: 45 degrees in, 45 out, with the
    // tangential component preserved.
    metal m(vec3(0.8f, 0.6f, 0.2f), 0.0f);
    ray scattered; vec3 atten;
    const ray incoming(vec3(-1, 1, 0), unit_vector(vec3(1, -1, 0)), 0.0f);
    out[0] = m.scatter(incoming, rec, atten, scattered, &rng) ? 1.f : 0.f;
    const vec3 d = unit_vector(scattered.direction());
    out[1] = d.x(); out[2] = d.y(); out[3] = d.z();
    out[4] = atten.x(); out[5] = atten.y(); out[6] = atten.z();

    // Glass never absorbs: white attenuation whichever branch it takes.
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
    RT_CHECK_VEC(s.v(1), 0.70710678f, 0.70710678f, 0.0f, kEps);
    RT_CHECK_VEC(s.v(4), 0.8f, 0.6f, 0.2f, kEps);

    RT_CHECK_NEAR(s[7], 1.0f, 0.0f);
    RT_CHECK_VEC(s.v(8), 1.0f, 1.0f, 1.0f, kEps);

    RT_CHECK_NEAR(s[11], 0.0f, 0.0f);
    RT_CHECK_VEC(s.v(12), 4.0f, 5.0f, 6.0f, kEps);
}

// out: [0]=scattered? [1..3]=attenuation [4]=direction length <= 1?
//      [5]=scatter origin matches hit point?
__global__ void k_lambertian_and_isotropic(float* out) {
    curandState rng;
    curand_init(3u, 0, 0, &rng);

    Hit rec;
    rec.p = vec3(2, 3, 4);
    rec.normal = vec3(0, 1, 0);
    rec.u = rec.v = 0.25f;
    rec.mat = 0;

    solid_color tex(vec3(0.3f, 0.6f, 0.9f));
    lambertian lam(&tex, /*owns=*/false);
    ray scattered; vec3 atten;
    out[0] = lam.scatter(ray(vec3(0, 5, 0), vec3(0, -1, 0), 0.0f), rec, atten, scattered, &rng) ? 1.f : 0.f;
    out[1] = atten.x(); out[2] = atten.y(); out[3] = atten.z();

    // Isotropic scatters into the unit sphere, so |dir| <= 1.
    isotropic iso(&tex, false);
    ray iso_scattered; vec3 iso_atten;
    iso.scatter(ray(vec3(0, 5, 0), vec3(0, -1, 0), 0.0f), rec, iso_atten, iso_scattered, &rng);
    out[4] = iso_scattered.direction().length() <= 1.0f ? 1.f : 0.f;
    out[5] = (iso_scattered.origin() - rec.p).length() < 1e-5f ? 1.f : 0.f;
}

TEST(material, lambertian_attenuation_comes_from_its_texture) {
    Scratch s(6);
    RT_REQUIRE(s.ok());
    k_lambertian_and_isotropic<<<1, 1>>>(s.data());
    RT_SYNC();

    RT_CHECK_NEAR(s[0], 1.0f, 0.0f);
    RT_CHECK_VEC(s.v(1), 0.3f, 0.6f, 0.9f, kEps);
    RT_CHECK_NEAR(s[4], 1.0f, 0.0f);
    RT_CHECK_NEAR(s[5], 1.0f, 0.0f);
}

// =============================================================================
// Textures
// =============================================================================

// out: [0..2]=even cell [3..5]=odd cell [6..8]=solid [9..11]=uv wrap
//      [12]=noise in range? [13]=turb non-negative?
__global__ void k_textures(float* out) {
    // scale 1.0 -> one cell per unit, so parity flips every integer step.
    solid_color a(vec3(1, 0, 0)), b(vec3(0, 0, 1));
    checker_texture chk(1.0f, &a, &b, /*owns=*/false);
    const vec3 e = chk.value(0.f, 0.f, vec3(0.5f, 0.5f, 0.5f));   // (0,0,0) -> even
    const vec3 o = chk.value(0.f, 0.f, vec3(1.5f, 0.5f, 0.5f));   // (1,0,0) -> odd
    out[0] = e.x(); out[1] = e.y(); out[2] = e.z();
    out[3] = o.x(); out[4] = o.y(); out[5] = o.z();

    solid_color sc(vec3(0.1f, 0.2f, 0.3f));
    const vec3 c = sc.value(0.9f, 0.4f, vec3(100, 200, 300));     // ignores u,v,p
    out[6] = c.x(); out[7] = c.y(); out[8] = c.z();

    // u + 0.75 must wrap into [0,1) rather than running off the texture.
    solid_color probe(vec3(0.5f, 0.5f, 0.5f));
    uv_offset_texture off(&probe, 0.75f);
    const vec3 w = off.value(0.5f, 0.5f, vec3(0, 0, 0));
    out[9] = w.x(); out[10] = w.y(); out[11] = w.z();

    // noise_texture uses the __sinf intrinsic, which is why it stays device-only.
    noise_texture nt(4.0f);
    bool in_range = true, non_neg = true;
    for (int i = 0; i < 32; ++i) {
        const vec3 p(i * 0.31f, i * -0.17f, i * 0.53f);
        const vec3 n = nt.value(0.f, 0.f, p);
        if (n.x() < -0.001f || n.x() > 1.001f) in_range = false;
        if (perlin::turb(p, 5) < 0.0f) non_neg = false;
    }
    out[12] = in_range ? 1.f : 0.f;
    out[13] = non_neg ? 1.f : 0.f;
}

TEST(texture, checker_parity_solid_and_uv_wrap) {
    Scratch s(14);
    RT_REQUIRE(s.ok());
    k_textures<<<1, 1>>>(s.data());
    RT_SYNC();

    RT_CHECK_VEC(s.v(0), 1.0f, 0.0f, 0.0f, kEps);   // even -> first texture
    RT_CHECK_VEC(s.v(3), 0.0f, 0.0f, 1.0f, kEps);   // odd  -> second texture
    RT_CHECK_VEC(s.v(6), 0.1f, 0.2f, 0.3f, kEps);
    RT_CHECK_VEC(s.v(9), 0.5f, 0.5f, 0.5f, kEps);   // wrapped lookup still valid
    RT_CHECK_NEAR(s[12], 1.0f, 0.0f);
    RT_CHECK_NEAR(s[13], 1.0f, 0.0f);
}

// =============================================================================
// Material table
// =============================================================================
//
// The bridge between host descriptors and device objects. If an index is
// mishandled here every primitive in the scene gets the wrong look, so it is
// worth checking end to end.

// out[i*3 .. i*3+2] = albedo/emission observed for material i
__global__ void k_probe_table(material** mats, int n, float* out) {
    if (threadIdx.x || blockIdx.x) return;
    curandState rng;
    curand_init(5u, 0, 0, &rng);

    Hit rec;
    rec.p = vec3(0, 0, 0);
    rec.normal = vec3(0, 1, 0);
    rec.u = rec.v = 0.5f;
    rec.mat = 0;

    for (int i = 0; i < n; ++i) {
        ray scattered; vec3 atten(0, 0, 0);
        const ray incoming(vec3(0, 1, 0), vec3(0, -1, 0), 0.0f);
        if (mats[i]->scatter(incoming, rec, atten, scattered, &rng)) {
            out[i * 3 + 0] = atten.x(); out[i * 3 + 1] = atten.y(); out[i * 3 + 2] = atten.z();
        } else {
            // Non-scattering material: report what it emits instead.
            const vec3 e = mats[i]->emitted(rec.u, rec.v, rec.p);
            out[i * 3 + 0] = e.x(); out[i * 3 + 1] = e.y(); out[i * 3 + 2] = e.z();
        }
    }
}

TEST(material_table, host_descriptors_become_the_right_device_objects) {
    rt::MaterialLibrary lib;
    const int lam   = lib.lambertian(vec3(0.1f, 0.2f, 0.3f));
    const int met   = lib.metal(vec3(0.4f, 0.5f, 0.6f), 0.0f);
    const int glass = lib.dielectric(1.5f);
    const int light = lib.diffuse_light(vec3(7.0f, 8.0f, 9.0f));
    const int iso   = lib.isotropic(vec3(0.7f, 0.7f, 0.7f));
    RT_CHECK_EQ(lam, 0); RT_CHECK_EQ(met, 1); RT_CHECK_EQ(glass, 2);
    RT_CHECK_EQ(light, 3); RT_CHECK_EQ(iso, 4);

    const auto& texd = lib.textures();
    const auto& matd = lib.materials();
    const int n_tex = (int)texd.size(), n_mat = (int)matd.size();

    rt::TexDesc* d_tex = nullptr; rt::MatDesc* d_mat = nullptr;
    texture** tex_tab = nullptr;  material** mat_tab = nullptr;
    RT_REQUIRE(cudaMalloc(&d_tex, n_tex * sizeof(rt::TexDesc)) == cudaSuccess);
    RT_REQUIRE(cudaMalloc(&d_mat, n_mat * sizeof(rt::MatDesc)) == cudaSuccess);
    RT_REQUIRE(cudaMalloc(&tex_tab, n_tex * sizeof(texture*)) == cudaSuccess);
    RT_REQUIRE(cudaMalloc(&mat_tab, n_mat * sizeof(material*)) == cudaSuccess);
    cudaMemcpy(d_tex, texd.data(), n_tex * sizeof(rt::TexDesc), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat, matd.data(), n_mat * sizeof(rt::MatDesc), cudaMemcpyHostToDevice);

    build_material_table<<<1, 1>>>(d_tex, n_tex, d_mat, n_mat, nullptr, tex_tab, mat_tab);
    RT_SYNC();

    Scratch s(n_mat * 3);
    RT_REQUIRE(s.ok());
    k_probe_table<<<1, 1>>>(mat_tab, n_mat, s.data());
    RT_SYNC();

    RT_CHECK_VEC(s.v(lam * 3),   0.1f, 0.2f, 0.3f, kEps);
    RT_CHECK_VEC(s.v(met * 3),   0.4f, 0.5f, 0.6f, kEps);
    RT_CHECK_VEC(s.v(glass * 3), 1.0f, 1.0f, 1.0f, kEps);   // glass never absorbs
    RT_CHECK_VEC(s.v(light * 3), 7.0f, 8.0f, 9.0f, kEps);   // emitted, not scattered
    RT_CHECK_VEC(s.v(iso * 3),   0.7f, 0.7f, 0.7f, kEps);

    free_material_table<<<1, 1>>>(tex_tab, n_tex, mat_tab, n_mat);
    RT_SYNC();
    cudaFree(mat_tab); cudaFree(tex_tab); cudaFree(d_mat); cudaFree(d_tex);
}

TEST(material_table, nested_textures_resolve_by_index) {
    // checker(solid, solid) and uv_offset(solid) both reference children by id.
    // The library only hands out an id after the child has one, so a single
    // forward pass in the build kernel is enough -- this pins that invariant.
    rt::MaterialLibrary lib;
    const int chk = lib.lambertian(lib.checker(1.0f, vec3(1, 0, 0), vec3(0, 0, 1)));
    const int off = lib.lambertian(lib.uv_offset(lib.solid(vec3(0.25f, 0.5f, 0.75f)), 0.5f));

    const auto& texd = lib.textures();
    const auto& matd = lib.materials();
    for (size_t i = 0; i < texd.size(); ++i) {
        if (texd[i].child0 >= (int)i || texd[i].child1 >= (int)i) {
            RT_FAIL("texture " + std::to_string(i) + " references a child at or after itself");
            return;
        }
    }

    const int n_tex = (int)texd.size(), n_mat = (int)matd.size();
    rt::TexDesc* d_tex = nullptr; rt::MatDesc* d_mat = nullptr;
    texture** tex_tab = nullptr;  material** mat_tab = nullptr;
    RT_REQUIRE(cudaMalloc(&d_tex, n_tex * sizeof(rt::TexDesc)) == cudaSuccess);
    RT_REQUIRE(cudaMalloc(&d_mat, n_mat * sizeof(rt::MatDesc)) == cudaSuccess);
    RT_REQUIRE(cudaMalloc(&tex_tab, n_tex * sizeof(texture*)) == cudaSuccess);
    RT_REQUIRE(cudaMalloc(&mat_tab, n_mat * sizeof(material*)) == cudaSuccess);
    cudaMemcpy(d_tex, texd.data(), n_tex * sizeof(rt::TexDesc), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat, matd.data(), n_mat * sizeof(rt::MatDesc), cudaMemcpyHostToDevice);

    build_material_table<<<1, 1>>>(d_tex, n_tex, d_mat, n_mat, nullptr, tex_tab, mat_tab);
    RT_SYNC();

    Scratch s(n_mat * 3);
    RT_REQUIRE(s.ok());
    k_probe_table<<<1, 1>>>(mat_tab, n_mat, s.data());
    RT_SYNC();

    // The probe hits p=(0,0,0), which is the even cell -> the first colour.
    RT_CHECK_VEC(s.v(chk * 3), 1.0f, 0.0f, 0.0f, kEps);
    RT_CHECK_VEC(s.v(off * 3), 0.25f, 0.5f, 0.75f, kEps);

    free_material_table<<<1, 1>>>(tex_tab, n_tex, mat_tab, n_mat);
    RT_SYNC();
    cudaFree(mat_tab); cudaFree(tex_tab); cudaFree(d_mat); cudaFree(d_tex);
}

// =============================================================================

int main(int argc, char** argv) {
    std::printf("=== device tests (materials, textures, material table) ===\n");

    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::printf("no CUDA device available -- skipping\n");
        return 77;   // CTest treats 77 as "skipped"
    }

    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    std::printf("device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);

    // Note what is *not* here any more: the old suite had to raise
    // cudaLimitStackSize and the malloc heap before it could build a BVH on the
    // device. Geometry no longer touches either.

    const int rc = rt_test::run_all(argc, argv);
    cudaDeviceReset();
    return rc;
}
