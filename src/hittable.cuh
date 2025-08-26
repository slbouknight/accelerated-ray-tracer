#ifndef HITTABLE_CUH
#define HITTABLE_CUH

#include <float.h>

#include "aabb.cuh"
#include "ray.cuh"

enum HKind : int { HK_Sphere=0, HK_Quad=1, HK_BVH=2, HK_Composite=3};

class material;

struct hit_record
{
    float t;
    vec3 p;
    vec3 normal;
    material *mat_ptr;
    double u;
    double v;
};

class hittable
{
    public:
        __device__ virtual ~hittable() = default;
        __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const = 0;
        __device__ virtual aabb bounding_box() const = 0;
        __device__ virtual HKind kind() const = 0;
};


// ------------------------------------------------------------
// Translate wrapper
// ------------------------------------------------------------
class translate : public hittable 
{
public:
    hittable* obj;
    vec3      offset;
    aabb      box;

    __device__ translate(hittable* p, const vec3& d) : obj(p), offset(d) {
        box = obj->bounding_box() + d;  // uses aabb+vec3 helper
    }

    __device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const override {
        // Move ray into object space (undo the instance translation)
        ray moved(r.origin() - offset, r.direction(), r.time());
        if (!obj->hit(moved, tmin, tmax, rec)) return false;

        // Bring hit point back to world space
        rec.p += offset;
        // rec.normal is unchanged by pure translation; the child already oriented it against moved.direction() which equals r.direction().
        return true;
    }

    __device__ virtual aabb bounding_box() const override { return box; }
    __device__ virtual HKind kind() const override { return obj->kind(); }
};

// ------------------------------------------------------------
// Rotate around +Y (degrees) wrapper
//   - Rotates ray into object space by -theta
//   - Rotates hit point/normal back by +theta
//   - Precomputes world-space bbox by rotating all 8 corners
// ------------------------------------------------------------
class rotate_y : public hittable 
{
public:
    hittable* obj;
    float     sin_t, cos_t;
    aabb      box;

    __device__ rotate_y(hittable* p, float angle_degrees) : obj(p) {
        const float rad = angle_degrees * 0.017453292519943295769f; // pi/180
        sin_t = sinf(rad);
        cos_t = cosf(rad);

        aabb b = obj->bounding_box();

        vec3 minp( FLT_MAX,  FLT_MAX,  FLT_MAX);
        vec3 maxp(-FLT_MAX, -FLT_MAX, -FLT_MAX);

        // Rotate all 8 corners of the child bbox, compute new bounds
        for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
        for (int k = 0; k < 2; ++k) {
            float x = i ? b.maximum.x() : b.minimum.x();
            float y = j ? b.maximum.y() : b.minimum.y();
            float z = k ? b.maximum.z() : b.minimum.z();
            // forward rotation: world = R * local
            float nx =  cos_t * x + sin_t * z;
            float nz = -sin_t * x + cos_t * z;
            vec3 p(nx, y, nz);

            minp = vec3(fminf(minp.x(), p.x()), fminf(minp.y(), p.y()), fminf(minp.z(), p.z()));
            maxp = vec3(fmaxf(maxp.x(), p.x()), fmaxf(maxp.y(), p.y()), fmaxf(maxp.z(), p.z()));
        }

        box = aabb(minp, maxp);
    }

    __device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const override {
        // Inverse-rotate ray origin/direction into object space by -theta
        const vec3  o = r.origin();
        const vec3  d = r.direction();
        const float ox =  cos_t * o.x() - sin_t * o.z();
        const float oz =  sin_t * o.x() + cos_t * o.z();
        const float dx =  cos_t * d.x() - sin_t * d.z();
        const float dz =  sin_t * d.x() + cos_t * d.z();

        ray rot_r(vec3(ox, o.y(), oz), vec3(dx, d.y(), dz), r.time());
        if (!obj->hit(rot_r, tmin, tmax, rec)) return false;

        // Rotate hit point back to world space (+theta)
        const float px =  cos_t * rec.p.x() + sin_t * rec.p.z();
        const float pz = -sin_t * rec.p.x() + cos_t * rec.p.z();

        // Rotate normal back to world space (+theta)
        const float nx =  cos_t * rec.normal.x() + sin_t * rec.normal.z();
        const float nz = -sin_t * rec.normal.x() + cos_t * rec.normal.z();

        rec.p      = vec3(px, rec.p.y(), pz);
        rec.normal = unit_vector(vec3(nx, rec.normal.y(), nz));

        // Ensure the normal faces against the *original* ray direction
        if (dot(rec.normal, r.direction()) > 0.f) rec.normal = -rec.normal;

        return true;
    }

    __device__ virtual aabb bounding_box() const override { return box; }
    __device__ virtual HKind kind() const override { return obj->kind(); }
};

// ------------------------------------------------------------
// Material override (optional): share geometry, change look
// ------------------------------------------------------------
class with_material : public hittable 
{
public:
    hittable*  obj;
    material*  mat;
    aabb       box;

    __device__ with_material(hittable* p, material* m) : obj(p), mat(m) {
        box = obj->bounding_box();
    }

    __device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const override {
        if (!obj->hit(r, tmin, tmax, rec)) return false;
        rec.mat_ptr = mat;  // per-instance material
        return true;
    }

    __device__ virtual aabb bounding_box() const override { return box; }
    __device__ virtual HKind kind() const override { return obj->kind(); }
};

#endif