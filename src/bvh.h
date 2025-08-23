#ifndef BVHH
#define BVHH

#include <curand_kernel.h>   // only if you want RNG for axis choice (optional)
#include "hittable.h"
#include "aabb.h"

// ----- helpers: box comparisons & tiny in-place sort over [start,end) -----

__device__ inline float box_axis_min(const hittable* obj, int axis) 
{
    aabb b = obj->bounding_box();
    return axis == 0 ? b.min().x() : (axis == 1 ? b.min().y() : b.min().z());
}

__device__ inline void sort_range_by_axis(hittable** list, int start, int end, int axis) 
{
    // Simple selection sort (OK for your ~500 objects, and it's run once in thread 0)
    for (int i = start; i < end - 1; ++i) {
        int best = i;
        float bestv = box_axis_min(list[i], axis);
        for (int j = i + 1; j < end; ++j) 
        {
            float v = box_axis_min(list[j], axis);
            if (v < bestv) { best = j; bestv = v; }
        }
        if (best != i) 
        {
            hittable* tmp = list[i];
            list[i] = list[best];
            list[best] = tmp;
        }
    }
}

// Optional deterministic axis choice based on range
__device__ inline int pick_axis(int start, int end) 
{
    return (end - start) % 3;
}

// ----- BVH node -----

class bvh_node : public hittable {
public:
    __device__ bvh_node() : left(nullptr), right(nullptr), box() {}

    // Build over list [start, end)
    __device__ bvh_node(hittable** list, int start, int end) : left(nullptr), right(nullptr), box() {
        int n = end - start;

        if (n == 1) {
            // Leaf with a single object
            left  = right = list[start];
            box = left->bounding_box();
            return;
        }

        if (n == 2) {
            // Two children; order them along an axis
            int axis = pick_axis(start, end);
            hittable* a = list[start];
            hittable* b = list[start + 1];
            if (box_axis_min(a, axis) > box_axis_min(b, axis)) {
                left = b; right = a;
            } else {
                left = a; right = b;
            }
            aabb bl = left->bounding_box();
            aabb br = right->bounding_box();
            box = aabb::surrounding_box(bl, br);
            return;
        }

        // n >= 3
        int axis = pick_axis(start, end);
        sort_range_by_axis(list, start, end, axis);
        int mid = start + n / 2;

        left  = new bvh_node(list, start, mid);
        right = new bvh_node(list, mid,   end);

        box = aabb::surrounding_box(left->bounding_box(), right->bounding_box());
    }

    __device__ bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override 
    {
        if (!box.hit(r, t_min, t_max)) return false;

        // Traverse children; visit nearer first can help a bit
        hit_record left_rec;
        bool hit_left  = left && left->hit(r, t_min, t_max, left_rec);

        float new_tmax = hit_left ? left_rec.t : t_max;

        hit_record right_rec;
        bool hit_right = right && right->hit(r, t_min, new_tmax, right_rec);

        if (hit_left && hit_right) {
            rec = (right_rec.t < left_rec.t) ? right_rec : left_rec;
            return true;
        } else if (hit_left) {
            rec = left_rec;
            return true;
        } else if (hit_right) {
            rec = right_rec;
            return true;
        }
        return false;
    }

    __device__ aabb bounding_box() const override { return box; }

    // POD members
    hittable* left;
    hittable* right;
    aabb      box;
};

#endif