#ifndef HITTABLELISTH
#define HITTABLELISTH

#include "hittable.h"
#include "aabb.h"

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

    __device__ bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override 
    {
        hit_record temp_rec;
        bool hit_anything = false;
        float closest_so_far = t_max;

        for (int i = 0; i < list_size; ++i) {
            if (list[i]->hit(r, t_min, closest_so_far, temp_rec)) 
            {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }
        return hit_anything;
    }

    __device__ aabb bounding_box() const override { return bbox; }

    hittable** list;
    int list_size;

private:
    aabb bbox;
};

#endif