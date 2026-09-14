#pragma once

#include "../core/primitives.hpp"
#include "../host/material_desc.hpp"
#include "../io/image_io.hpp"
#include "material.cuh"
#include "texture.cuh"

// Instantiates the device-side material and texture objects from the host's
// descriptor records.
//
// This is the only remaining device-side construction, and it is deliberately
// small: tens of objects per scene instead of thousands. It runs on one thread
// because building 500 materials takes under a millisecond -- it was the
// O(n^2) BVH sort over thousands of primitives that cost 1.6 seconds, not the
// allocations.

__global__ void build_material_table(const rt::TexDesc* tex_desc, int n_tex,
                                     const rt::MatDesc* mat_desc, int n_mat,
                                     const DeviceImage* images,
                                     texture** out_tex, material** out_mat)
{
    if (threadIdx.x || blockIdx.x) return;

    // Textures first. Descriptors reference their children by index, and the
    // library only ever hands out an id after the child already has one, so a
    // single forward pass is enough -- child0/child1 are always < i.
    for (int i = 0; i < n_tex; ++i) {
        const rt::TexDesc& d = tex_desc[i];
        switch (d.kind) {
            case rt::TEX_SOLID:
                out_tex[i] = new solid_color(d.color);
                break;
            case rt::TEX_CHECKER:
                // Non-owning: the table frees every texture itself, so a
                // checker must not also delete its children.
                out_tex[i] = new checker_texture(d.f0, out_tex[d.child0], out_tex[d.child1], false);
                break;
            case rt::TEX_IMAGE:
                out_tex[i] = new image_texture(images[d.image]);
                break;
            case rt::TEX_NOISE:
                out_tex[i] = new noise_texture(d.f0);
                break;
            case rt::TEX_NOODLE:
                out_tex[i] = new noodle_texture(d.f0);
                break;
            case rt::TEX_FELT:
                out_tex[i] = new felt_texture(d.color, d.f0, d.f1, d.f2, d.f3);
                break;
            case rt::TEX_UV_OFFSET:
                out_tex[i] = new uv_offset_texture(out_tex[d.child0], d.f0);
                break;
            default:
                out_tex[i] = new solid_color(vec3(1, 0, 1));   // visible placeholder
                break;
        }
    }

    for (int i = 0; i < n_mat; ++i) {
        const rt::MatDesc& d = mat_desc[i];
        switch (d.kind) {
            case rt::MAT_LAMBERTIAN:
                out_mat[i] = new lambertian(out_tex[d.tex], false);
                break;
            case rt::MAT_METAL:
                out_mat[i] = new metal(d.albedo, d.param);
                break;
            case rt::MAT_DIELECTRIC:
                out_mat[i] = new dielectric(d.param);
                break;
            case rt::MAT_LIGHT:
                out_mat[i] = new diffuse_light(out_tex[d.tex], false);
                break;
            case rt::MAT_ISOTROPIC:
                out_mat[i] = new isotropic(out_tex[d.tex], false);
                break;
            default:
                out_mat[i] = new lambertian(out_tex[0], false);
                break;
        }
    }
}

// Every object is owned by exactly one array, so teardown is a flat loop with
// no ownership rules to get wrong. This is what replaced the tangle of
// owns_mat flags, shared-material double frees and leaked instance wrappers.
__global__ void free_material_table(texture** tex, int n_tex,
                                    material** mat, int n_mat)
{
    if (threadIdx.x || blockIdx.x) return;
    for (int i = 0; i < n_mat; ++i) delete mat[i];
    for (int i = 0; i < n_tex; ++i) delete tex[i];
}
