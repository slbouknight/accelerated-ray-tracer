#pragma once

#include "../core/primitives.hpp"
#include "../core/scene_view.hpp"
#include "material_desc.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

// Host-side scene authoring, in the shape the book uses.
//
// You add spheres, quads and boxes; you get back flat arrays and a BVH ready to
// upload. The book's readability lives in the scene *description*, which is
// still ordinary C++ with named materials -- what changed is that the traversal
// representation underneath is flat instead of a pointer graph.
//
// Instancing is applied here rather than at trace time. The old translate and
// rotate_y wrappers transformed every ray on every hit test; a rigid transform
// of a quad is just another quad, so the transform is applied once to the
// corner and edge vectors at build time and costs nothing thereafter.

namespace rt {

// A rigid transform, in the order the book composes them: rotate about Y, then
// translate. Default-constructed, it is the identity.
struct Transform {
    float sin_t = 0.0f, cos_t = 1.0f;
    vec3  offset = vec3(0, 0, 0);

    static Transform rotate_y_degrees(float deg) {
        Transform t;
        const float rad = deg * (pi / 180.0f);
        t.sin_t = std::sin(rad);
        t.cos_t = std::cos(rad);
        return t;
    }
    Transform& then_translate(const vec3& d) { offset = offset + d; return *this; }

    // Rotate a point about the origin, then translate.
    vec3 apply_point(const vec3& p) const {
        return vec3(cos_t * p.x() + sin_t * p.z(), p.y(), -sin_t * p.x() + cos_t * p.z()) + offset;
    }
    // Directions rotate but do not translate.
    vec3 apply_dir(const vec3& d) const {
        return vec3(cos_t * d.x() + sin_t * d.z(), d.y(), -sin_t * d.x() + cos_t * d.z());
    }
};

class SceneBuilder {
public:
    // ---- geometry -------------------------------------------------------

    void add_sphere(const vec3& center, float radius, int mat) {
        spheres_.push_back(Sphere{center, center, radius, mat});
        refs_.push_back(PrimRef{PRIM_SPHERE, (int)spheres_.size() - 1});
    }

    // Moving sphere: centre travels from c0 to c1 across the shutter interval.
    void add_sphere(const vec3& c0, const vec3& c1, float radius, int mat) {
        spheres_.push_back(Sphere{c0, c1, radius, mat});
        refs_.push_back(PrimRef{PRIM_SPHERE, (int)spheres_.size() - 1});
    }

    void add_quad(const vec3& Q, const vec3& u, const vec3& v, int mat,
                  bool inward = false, const Transform& xf = Transform{}) {
        Quad q{};
        q.Q = xf.apply_point(Q);
        q.u = xf.apply_dir(u);
        q.v = xf.apply_dir(v);
        q.mat = mat;
        q.finalize(inward);
        quads_.push_back(q);
        refs_.push_back(PrimRef{PRIM_QUAD, (int)quads_.size() - 1});
    }

    // A box as six quads, exactly as the book's box() helper builds it. Any
    // transform is baked into each face.
    void add_box(const vec3& a, const vec3& b, int mat, const Transform& xf = Transform{}) {
        const vec3 lo(std::fmin(a.x(), b.x()), std::fmin(a.y(), b.y()), std::fmin(a.z(), b.z()));
        const vec3 hi(std::fmax(a.x(), b.x()), std::fmax(a.y(), b.y()), std::fmax(a.z(), b.z()));

        const vec3 dx(hi.x() - lo.x(), 0, 0);
        const vec3 dy(0, hi.y() - lo.y(), 0);
        const vec3 dz(0, 0, hi.z() - lo.z());

        add_quad(vec3(lo.x(), lo.y(), hi.z()),  dx,  dy, mat, false, xf);  // +Z
        add_quad(vec3(hi.x(), lo.y(), hi.z()), -dz,  dy, mat, false, xf);  // +X
        add_quad(vec3(hi.x(), lo.y(), lo.z()), -dx,  dy, mat, false, xf);  // -Z
        add_quad(vec3(lo.x(), lo.y(), lo.z()),  dz,  dy, mat, false, xf);  // -X
        add_quad(vec3(lo.x(), hi.y(), hi.z()),  dx, -dz, mat, false, xf);  // +Y
        add_quad(vec3(lo.x(), lo.y(), lo.z()),  dx,  dz, mat, false, xf);  // -Y
    }

    void add_medium_sphere(const vec3& center, float radius, float density, int phase_mat) {
        Medium m{};
        m.bound  = MEDIUM_SPHERE;
        m.center = center;
        m.radius = radius;
        m.neg_inv_density = -1.0f / density;
        m.mat = phase_mat;
        media_.push_back(m);
        refs_.push_back(PrimRef{PRIM_MEDIUM, (int)media_.size() - 1});
    }

    // Box-bounded medium. `a`/`b` are the untransformed corners; the rotation
    // is taken from `xf` and applied about the box centre.
    void add_medium_box(const vec3& a, const vec3& b, float density, int phase_mat,
                        const Transform& xf = Transform{}) {
        const vec3 lo(std::fmin(a.x(), b.x()), std::fmin(a.y(), b.y()), std::fmin(a.z(), b.z()));
        const vec3 hi(std::fmax(a.x(), b.x()), std::fmax(a.y(), b.y()), std::fmax(a.z(), b.z()));
        const vec3 c = (lo + hi) * 0.5f;

        Medium m{};
        m.bound       = MEDIUM_BOX;
        m.center      = xf.apply_point(c);
        m.half_extent = (hi - lo) * 0.5f;
        m.sin_t       = xf.sin_t;
        m.cos_t       = xf.cos_t;
        m.neg_inv_density = -1.0f / density;
        m.mat = phase_mat;
        media_.push_back(m);
        refs_.push_back(PrimRef{PRIM_MEDIUM, (int)media_.size() - 1});
    }

    // ---- materials and textures -----------------------------------------
    // These record descriptions; the actual (still polymorphic) material and
    // texture objects are instantiated on the device from these records. See
    // material_desc.hpp.

    MaterialLibrary&       materials()       { return mats_; }
    const MaterialLibrary& materials() const { return mats_; }

    // ---- build ----------------------------------------------------------

    int  primitive_count() const { return (int)refs_.size(); }
    const std::vector<Sphere>&  spheres() const { return spheres_; }
    const std::vector<Quad>&    quads()   const { return quads_; }
    const std::vector<Medium>&  media()   const { return media_; }
    const std::vector<PrimRef>& refs()    const { return refs_; }
    const std::vector<BvhNode>& nodes()   const { return nodes_; }

    // Build the BVH over everything added so far.
    //
    // Same tree the book builds -- median split on the axis of greatest spread,
    // one primitive per leaf -- but with an O(n log n) nth_element partition
    // instead of the O(n^2) selection sort the device version used, and running
    // on a CPU core instead of one CUDA thread.
    void build_bvh() {
        nodes_.clear();
        if (refs_.empty()) return;
        nodes_.reserve(2 * refs_.size());
        build_range(0, (int)refs_.size());
    }

private:
    aabb ref_bounds(const PrimRef& r) const {
        switch (r.kind) {
            case PRIM_SPHERE: return spheres_[r.index].bounds();
            case PRIM_QUAD:   return quads_  [r.index].bounds();
            case PRIM_MEDIUM: return media_  [r.index].bounds();
            default:          return aabb();
        }
    }

    static float axis_of(const vec3& v, int axis) {
        return axis == 0 ? v.x() : (axis == 1 ? v.y() : v.z());
    }

    // Returns the index of the node covering refs_[start, end).
    int build_range(int start, int end) {
        const int self = (int)nodes_.size();
        nodes_.push_back(BvhNode{});

        aabb bounds;
        for (int i = start; i < end; ++i)
            bounds = aabb::surrounding_box(bounds, ref_bounds(refs_[i]));

        const int n = end - start;
        if (n <= 1) {
            nodes_[self] = BvhNode{bounds, -1, -1, start, n};
            return self;
        }

        // Split on whichever axis the primitive origins are most spread along.
        float lo[3] = { 1e30f,  1e30f,  1e30f};
        float hi[3] = {-1e30f, -1e30f, -1e30f};
        for (int i = start; i < end; ++i) {
            const vec3 mn = ref_bounds(refs_[i]).min();
            for (int a = 0; a < 3; ++a) {
                lo[a] = std::fmin(lo[a], axis_of(mn, a));
                hi[a] = std::fmax(hi[a], axis_of(mn, a));
            }
        }
        int axis = 0;
        float best = hi[0] - lo[0];
        for (int a = 1; a < 3; ++a) {
            if (hi[a] - lo[a] > best) { best = hi[a] - lo[a]; axis = a; }
        }

        const int mid = start + n / 2;
        std::nth_element(refs_.begin() + start, refs_.begin() + mid, refs_.begin() + end,
                         [&](const PrimRef& x, const PrimRef& y) {
                             return axis_of(ref_bounds(x).min(), axis)
                                  < axis_of(ref_bounds(y).min(), axis);
                         });

        const int l = build_range(start, mid);
        const int r = build_range(mid, end);
        nodes_[self] = BvhNode{bounds, l, r, 0, 0};
        return self;
    }

    std::vector<Sphere>  spheres_;
    std::vector<Quad>    quads_;
    std::vector<Medium>  media_;
    std::vector<PrimRef> refs_;
    std::vector<BvhNode> nodes_;
    MaterialLibrary      mats_;
};

} // namespace rt
