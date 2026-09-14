#pragma once

#include "../core/vec3.hpp"

#include <vector>

// Host-side *descriptions* of materials and textures.
//
// Geometry is now flat POD, but materials stayed a small virtual hierarchy:
// shading is where the book's class structure earns its keep, and there are
// tens of materials per scene rather than thousands of primitives, so the
// indirect call is not on the critical path the way traversal was.
//
// The bridge is these records. A scene names materials on the host and gets
// back integer ids; a single small kernel later instantiates the real objects
// from the records into a device array, and primitives reference them by index.
// No material pointer ever crosses the host/device boundary, which is what lets
// the geometry be built on the CPU.

namespace rt {

enum TexKind : int {
    TEX_SOLID = 0, TEX_CHECKER, TEX_IMAGE, TEX_NOISE,
    TEX_NOODLE, TEX_FELT, TEX_UV_OFFSET,
};

struct TexDesc {
    int   kind   = TEX_SOLID;
    vec3  color  = vec3(0, 0, 0);
    float f0 = 0, f1 = 0, f2 = 0, f3 = 0;
    int   child0 = -1, child1 = -1;   // texture ids, for checker / uv_offset
    int   image  = -1;                // index into the scene's image list
};

enum MatKind : int {
    MAT_LAMBERTIAN = 0, MAT_METAL, MAT_DIELECTRIC, MAT_LIGHT, MAT_ISOTROPIC,
};

struct MatDesc {
    int   kind   = MAT_LAMBERTIAN;
    int   tex    = -1;                // lambertian / light / isotropic
    vec3  albedo = vec3(0, 0, 0);     // metal
    float param  = 0.0f;              // metal fuzz, or dielectric IOR
};

// Accumulates the descriptions and hands out ids. Textures and materials are
// numbered separately.
class MaterialLibrary {
public:
    // ---- textures -------------------------------------------------------
    int solid(const vec3& c) {
        TexDesc t; t.kind = TEX_SOLID; t.color = c;
        return push(t);
    }
    int checker(float scale, int even_tex, int odd_tex) {
        TexDesc t; t.kind = TEX_CHECKER; t.f0 = scale;
        t.child0 = even_tex; t.child1 = odd_tex;
        return push(t);
    }
    int checker(float scale, const vec3& even, const vec3& odd) {
        return checker(scale, solid(even), solid(odd));
    }
    int image(int image_index) {
        TexDesc t; t.kind = TEX_IMAGE; t.image = image_index;
        return push(t);
    }
    int noise(float scale) {
        TexDesc t; t.kind = TEX_NOISE; t.f0 = scale;
        return push(t);
    }
    int noodle(float stripes_k) {
        TexDesc t; t.kind = TEX_NOODLE; t.f0 = stripes_k;
        return push(t);
    }
    int felt(const vec3& base, float mottling_scale, float mottling_amt,
             float fiber_scale, float fiber_amt) {
        TexDesc t; t.kind = TEX_FELT; t.color = base;
        t.f0 = mottling_scale; t.f1 = mottling_amt;
        t.f2 = fiber_scale;    t.f3 = fiber_amt;
        return push(t);
    }
    // Rotates the u coordinate by `turns` (1.0 == a full revolution).
    int uv_offset(int base_tex, float turns) {
        TexDesc t; t.kind = TEX_UV_OFFSET; t.child0 = base_tex; t.f0 = turns;
        return push(t);
    }

    // ---- materials ------------------------------------------------------
    int lambertian(int tex) {
        MatDesc m; m.kind = MAT_LAMBERTIAN; m.tex = tex;
        return push(m);
    }
    int lambertian(const vec3& albedo) { return lambertian(solid(albedo)); }

    int metal(const vec3& albedo, float fuzz) {
        MatDesc m; m.kind = MAT_METAL; m.albedo = albedo; m.param = fuzz;
        return push(m);
    }
    int dielectric(float ior) {
        MatDesc m; m.kind = MAT_DIELECTRIC; m.param = ior;
        return push(m);
    }
    int diffuse_light(int tex) {
        MatDesc m; m.kind = MAT_LIGHT; m.tex = tex;
        return push(m);
    }
    int diffuse_light(const vec3& emit) { return diffuse_light(solid(emit)); }

    int isotropic(int tex) {
        MatDesc m; m.kind = MAT_ISOTROPIC; m.tex = tex;
        return push(m);
    }
    int isotropic(const vec3& albedo) { return isotropic(solid(albedo)); }

    const std::vector<TexDesc>& textures()  const { return textures_; }
    const std::vector<MatDesc>& materials() const { return materials_; }

private:
    int push(const TexDesc& t) { textures_.push_back(t);  return (int)textures_.size()  - 1; }
    int push(const MatDesc& m) { materials_.push_back(m); return (int)materials_.size() - 1; }

    std::vector<TexDesc> textures_;
    std::vector<MatDesc> materials_;
};

} // namespace rt
