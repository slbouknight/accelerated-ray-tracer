#ifndef BVH_CUH
#define BVH_CUH

#include <curand_kernel.h>   // for curandState in the overload signature
#include "hittable.cuh"
#include "aabb.cuh"

// Helper to read a component from a vec3 when we only know axis at runtime
__device__ inline float get_axis(const vec3& v, int axis) {
    return (axis == 0) ? v.x() : (axis == 1) ? v.y() : v.z();
}

// Compare two hittables by the min corner of their bounding boxes along 'axis'
__device__ inline bool box_min_less(hittable* a, hittable* b, int axis) {
    const aabb ba = a->bounding_box();
    const aabb bb = b->bounding_box();
    return get_axis(ba.minimum, axis) < get_axis(bb.minimum, axis);
}

class bvh_node : public hittable {
public:
    hittable* left  = nullptr;
    hittable* right = nullptr;
    aabb      box;

    __device__ bvh_node() : left(nullptr), right(nullptr), box() {}

    // Build a BVH from objects[start, end)
    __device__ bvh_node(hittable** objects, int start, int end) {
        const int n = end - start;

        if (n <= 0) {
            left = right = nullptr;
            box  = aabb(); // empty
            return;
        }

        if (n == 1) {
            // Leaf: point both children at the same object (standard RTIOW trick)
            left  = right = objects[start];
            box   = left->bounding_box();
            return;
        }

        // Choose split axis by largest spread of box minima (deterministic)
        float minx =  1e30f, maxx = -1e30f;
        float miny =  1e30f, maxy = -1e30f;
        float minz =  1e30f, maxz = -1e30f;

        for (int i = start; i < end; ++i) {
            const aabb bi = objects[i]->bounding_box();
            const vec3 mn = bi.min();
            if (mn.x() < minx) minx = mn.x(); if (mn.x() > maxx) maxx = mn.x();
            if (mn.y() < miny) miny = mn.y(); if (mn.y() > maxy) maxy = mn.y();
            if (mn.z() < minz) minz = mn.z(); if (mn.z() > maxz) maxz = mn.z();
        }
        const float sx = maxx - minx;
        const float sy = maxy - miny;
        const float sz = maxz - minz;

        int axis = 0;
        if (sy > sx && sy >= sz) axis = 1;
        else if (sz > sx && sz >= sy) axis = 2;

        // In-place selection sort by chosen axis on [start, end)
        for (int i = start; i < end - 1; ++i) {
            int best = i;
            for (int j = i + 1; j < end; ++j) {
                if (box_min_less(objects[j], objects[best], axis))
                    best = j;
            }
            if (best != i) {
                hittable* tmp = objects[i];
                objects[i] = objects[best];
                objects[best] = tmp;
            }
        }

        const int mid = start + (n >> 1);
        left  = new bvh_node(objects, start, mid);
        right = new bvh_node(objects, mid,   end);

        box = aabb::surrounding_box(left->bounding_box(), right->bounding_box());
    }

    // Only delete internal BVH nodes. Leaves are freed from d_list in free_world()
    __device__ ~bvh_node() override {
        if (left  && left->kind()  == HK_BVH) delete left;
        if (right && right->kind() == HK_BVH) delete right;
    }

    // ---- Virtuals required by hittable ----

    // 4-arg legacy hit: real implementation lives here
    __device__ bool hit(const ray& r, float tmin, float tmax,
                        hit_record& rec) const override {
        if (!box.hit(r, tmin, tmax)) return false;

        hit_record lrec, rrec;
        const bool hl = left  ? left ->hit(r, tmin, tmax, lrec) : false;
        const bool hr = right ? right->hit(r, tmin, hl ? lrec.t : tmax, rrec) : false;

        if (hr) rec = rrec;
        if (hl && (!hr || lrec.t < rrec.t)) rec = lrec;
        return hl || hr;
    }

    // 5-arg RNG overload: forward to the 4-arg version (silences partial-override warnings)
    __device__ bool hit(const ray& r, float tmin, float tmax,
                        hit_record& rec, curandState* /*rng*/) const override {
        return hit(r, tmin, tmax, rec);
    }

    __device__ aabb  bounding_box() const override { return box; }
    __device__ HKind kind()         const override { return HK_BVH; }
};

#endif // BVH_CUH
