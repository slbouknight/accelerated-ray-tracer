#ifndef HITTABLELIST_CUH
#define HITTABLELIST_CUH

#include "hittable.cuh"
#include "aabb.cuh"

class hittable_list : public hittable 
{
public:
    __device__ hittable_list()
        : list(nullptr), list_size(0), bbox() {}

    __device__ hittable_list(hittable** l, int n)
        : list(l), list_size(n), bbox()
    {
        if (n > 0) {
            // start with the first object's box
            bbox = l[0]->bounding_box();
            // union with the rest
            for (int k = 1; k < n; ++k) 
            {
                bbox = aabb::surrounding_box(bbox, l[k]->bounding_box());
            }
        }
        // if n == 0, bbox stays as the default-constructed (empty) box
    }

    // 4-arg legacy override: actual implementation
    __device__ bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const override {
        hit_record temp;
        bool hit_any = false;
        float closest = tmax;

        for (int i = 0; i < list_size; ++i) {
            if (list[i]->hit(r, tmin, closest, temp)) {
                hit_any = true;
                closest = temp.t;
                rec = temp;
            }
        }
        return hit_any;
    }

    // 5-arg RNG overload: forward to 4-arg
    __device__ bool hit(const ray& r, float tmin, float tmax,
                        hit_record& rec, curandState* /*rng*/) const override {
        return hit(r, tmin, tmax, rec);
    }

    __device__ aabb bounding_box() const override { return bbox; }

    hittable** list;
    int list_size;

private:
    aabb bbox;
};

#endif