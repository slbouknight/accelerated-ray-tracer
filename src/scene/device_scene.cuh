#pragma once

#include "../core/scene_view.hpp"
#include "../host/scene_builder.hpp"
#include "../io/image_io.hpp"
#include "material_table.cuh"

#include <cstdio>
#include <vector>

// Uploads a host-built scene to the device and owns the result.
//
// The geometry side is now just five cudaMemcpys of contiguous arrays -- the
// whole point of making primitives POD. Only the material table still needs a
// kernel, because materials remain polymorphic and a vtable pointer has to be
// created on the device that will use it.

struct DeviceScene {
    // Geometry, uploaded wholesale.
    Sphere*  spheres = nullptr;
    Quad*    quads   = nullptr;
    Medium*  media   = nullptr;
    PrimRef* refs    = nullptr;
    BvhNode* nodes   = nullptr;

    // Material table, instantiated on the device from host descriptors.
    rt::TexDesc* tex_desc = nullptr;
    rt::MatDesc* mat_desc = nullptr;
    DeviceImage* images   = nullptr;
    texture**    textures = nullptr;
    material**   materials = nullptr;

    int n_tex = 0, n_mat = 0, n_images = 0;
    int prim_count = 0;

    SceneView view;
    std::vector<DeviceImage> host_images;   // kept so they can be freed

    bool ok = false;
};

namespace detail {

template <class T>
inline bool upload(T** dst, const std::vector<T>& src) {
    if (src.empty()) { *dst = nullptr; return true; }
    const size_t bytes = src.size() * sizeof(T);
    if (cudaMalloc(dst, bytes) != cudaSuccess) return false;
    return cudaMemcpy(*dst, src.data(), bytes, cudaMemcpyHostToDevice) == cudaSuccess;
}

} // namespace detail

// `images` are the already-loaded textures, in the order the scene's TEX_IMAGE
// descriptors index them.
inline DeviceScene upload_scene(const rt::SceneBuilder& world,
                                const std::vector<DeviceImage>& images)
{
    DeviceScene d;
    d.host_images = images;
    d.prim_count  = world.primitive_count();

    if (!detail::upload(&d.spheres, world.spheres()) ||
        !detail::upload(&d.quads,   world.quads())   ||
        !detail::upload(&d.media,   world.media())   ||
        !detail::upload(&d.refs,    world.refs())    ||
        !detail::upload(&d.nodes,   world.nodes())) {
        std::fprintf(stderr, "error: failed to upload scene geometry\n");
        return d;
    }

    const auto& texd = world.materials().textures();
    const auto& matd = world.materials().materials();
    d.n_tex    = (int)texd.size();
    d.n_mat    = (int)matd.size();
    d.n_images = (int)images.size();

    if (!detail::upload(&d.tex_desc, texd) ||
        !detail::upload(&d.mat_desc, matd) ||
        !detail::upload(&d.images,   images)) {
        std::fprintf(stderr, "error: failed to upload material descriptors\n");
        return d;
    }

    if (d.n_tex > 0 && cudaMalloc(&d.textures, d.n_tex * sizeof(texture*)) != cudaSuccess) return d;
    if (d.n_mat > 0 && cudaMalloc(&d.materials, d.n_mat * sizeof(material*)) != cudaSuccess) return d;

    build_material_table<<<1, 1>>>(d.tex_desc, d.n_tex, d.mat_desc, d.n_mat,
                                   d.images, d.textures, d.materials);
    if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
        std::fprintf(stderr, "error: material table construction failed\n");
        return d;
    }

    d.view.spheres    = d.spheres;
    d.view.quads      = d.quads;
    d.view.media      = d.media;
    d.view.refs       = d.refs;
    d.view.nodes      = d.nodes;
    d.view.node_count = (int)world.nodes().size();

    d.ok = true;
    return d;
}

inline void free_scene(DeviceScene& d) {
    if (d.textures || d.materials) {
        free_material_table<<<1, 1>>>(d.textures, d.n_tex, d.materials, d.n_mat);
        cudaDeviceSynchronize();
    }
    cudaFree(d.materials); cudaFree(d.textures);
    cudaFree(d.images);    cudaFree(d.mat_desc); cudaFree(d.tex_desc);
    cudaFree(d.nodes);     cudaFree(d.refs);
    cudaFree(d.media);     cudaFree(d.quads);    cudaFree(d.spheres);
    for (DeviceImage& img : d.host_images) free_device_image(img);
    d = DeviceScene{};
}
