#pragma once

#include "aabb.hpp"
#include "cuda_compat.hpp"
#include "primitives.hpp"
#include "ray.hpp"

// A flattened BVH over flat primitive arrays, plus the traversal that walks it.
//
// The tree is the same one the book builds -- binary, median split on the axis
// of greatest spread, one primitive per leaf -- so it produces the same
// intersections. What changed is that it is an array of PODs built on the host
// rather than a pointer graph built by a single CUDA thread, and that traversal
// is an index-driven loop rather than recursion through virtual calls.

// Which array a leaf's primitive lives in, and where.
struct PrimRef {
    int kind;    // PrimKind
    int index;
};

// Depth-first node array. `count == 0` marks an interior node; otherwise the
// node owns `count` consecutive entries of the scene's PrimRef array starting
// at `first`.
struct BvhNode {
    aabb bounds;
    int  left;     // interior: index of the left child
    int  right;    // interior: index of the right child
    int  first;    // leaf: first PrimRef
    int  count;    // leaf: number of PrimRefs; 0 => interior
};

// Everything a kernel needs to trace the scene, passed by value. Just pointers
// and counts -- no ownership, no virtuals.
struct SceneView {
    const Sphere*  spheres = nullptr;
    const Quad*    quads   = nullptr;
    const Medium*  media   = nullptr;
    const PrimRef* refs    = nullptr;
    const BvhNode* nodes   = nullptr;
    int            node_count = 0;

    RT_HD bool empty() const { return node_count <= 0; }
};

// Intersect one primitive, dispatching on its tag.
//
// `u01` is a uniform sample the caller has already drawn; only PRIM_MEDIUM
// consumes it. Drawing it unconditionally keeps this function pure and free of
// any RNG dependency, at the cost of one wasted random number on the rare rays
// that test a medium and miss.
RT_HD inline bool hit_prim(const SceneView& sc, const PrimRef& ref, const ray& r,
                           float t_min, float t_max, float u01, Hit& rec)
{
    switch (ref.kind) {
        case PRIM_SPHERE: return hit_sphere(sc.spheres[ref.index], r, t_min, t_max, rec);
        case PRIM_QUAD:   return hit_quad  (sc.quads  [ref.index], r, t_min, t_max, rec);
        case PRIM_MEDIUM: return hit_medium(sc.media  [ref.index], r, t_min, t_max, u01, rec);
        default:          return false;
    }
}

// Maximum traversal stack depth. The builder splits at the median, so depth is
// ~log2(n): 64 is far beyond anything these scenes reach, and the array is
// per-thread local memory, so it is worth keeping tight.
inline constexpr int kBvhMaxStack = 64;

// `Rng` only has to provide `float next()` returning a uniform in [0,1).
// Templating rather than taking a curandState* is what lets the host tests
// traverse the same code with a plain xorshift -- the traversal logic is
// verified without a GPU, and the device path pays nothing for the abstraction
// because it inlines.
template <class Rng>
RT_HD inline bool scene_intersect(const SceneView& sc, const ray& r,
                                  float t_min, float t_max, Hit& rec, Rng& rng)
{
    if (sc.empty()) return false;
    if (!sc.nodes[0].bounds.hit(r, t_min, t_max)) return false;

    int stack[kBvhMaxStack];
    int sp = 0;
    int node = 0;

    float closest = t_max;
    bool  hit_any = false;
    Hit   tmp;

    for (;;) {
        const BvhNode& n = sc.nodes[node];

        if (n.count == 0) {
            const int li = n.left, ri = n.right;
            // Test both child boxes against the *current* closest hit, so a
            // near hit prunes the far subtree before it is ever descended.
            const bool hl = li >= 0 && sc.nodes[li].bounds.hit(r, t_min, closest);
            const bool hr = ri >= 0 && sc.nodes[ri].bounds.hit(r, t_min, closest);

            if (hl && hr) {
                if (sp < kBvhMaxStack) stack[sp++] = ri;
                node = li;
                continue;
            }
            if (hl) { node = li; continue; }
            if (hr) { node = ri; continue; }
        } else {
            for (int i = 0; i < n.count; ++i) {
                if (hit_prim(sc, sc.refs[n.first + i], r, t_min, closest, rng.next(), tmp)) {
                    hit_any = true;
                    closest = tmp.t;
                    rec     = tmp;
                }
            }
        }

        if (sp == 0) break;
        node = stack[--sp];
    }

    return hit_any;
}

// Reference implementation: test every primitive, ignore the tree. Far too slow
// to render with, but it is the ground truth the BVH is checked against -- an
// acceleration structure is only allowed to make the same answer arrive faster.
template <class Rng>
RT_HD inline bool scene_intersect_bruteforce(const SceneView& sc, const ray& r,
                                             float t_min, float t_max,
                                             int prim_count, Hit& rec, Rng& rng)
{
    float closest = t_max;
    bool  hit_any = false;
    Hit   tmp;
    for (int i = 0; i < prim_count; ++i) {
        if (hit_prim(sc, sc.refs[i], r, t_min, closest, rng.next(), tmp)) {
            hit_any = true;
            closest = tmp.t;
            rec     = tmp;
        }
    }
    return hit_any;
}

// A deterministic uniform source, for host tests and for any call site that has
// no cuRAND state. Not suitable for rendering -- xorshift32's low bits are weak
// and the streams are not decorrelated.
struct XorShiftRng {
    unsigned int s;
    RT_HD explicit XorShiftRng(unsigned int seed = 1u) : s(seed ? seed : 1u) {}
    RT_HD float next() {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        return (s & 0x00FFFFFFu) * (1.0f / 16777216.0f);
    }
};
