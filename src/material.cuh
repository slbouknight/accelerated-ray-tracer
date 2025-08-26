// material.cuh  (no front_face needed)
#pragma once
#include <curand_kernel.h>
#include "ray.cuh"
#include "hittable.cuh"    // hit_record with p, normal, u, v (no front_face required)
#include "texture.cuh"     // solid_color, checker_texture, image_texture
#include "vec3.cuh"

// -------- helpers (same spirit as your current code) ----------
__device__ inline float randf(curandState* st) { return curand_uniform(st); }

__device__ inline vec3 random_in_unit_sphere(curandState* st) 
{
    while (true) {
        vec3 p(2.0f*randf(st)-1.0f, 2.0f*randf(st)-1.0f, 2.0f*randf(st)-1.0f);
        if (p.squared_length() < 1.0f) return p;
    }
}

__device__ inline vec3 reflect(const vec3& v, const vec3& n) 
{
    return v - 2.0f * dot(v, n) * n;
}

// your original style refract (no front_face required)
__device__ inline bool refract(const vec3& v, const vec3& n, float ni_over_nt, vec3& refracted) 
{
    vec3 uv = unit_vector(v);
    float dt = dot(uv, n);
    float disc = 1.0f - ni_over_nt*ni_over_nt*(1.0f - dt*dt);
    if (disc > 0.0f) {
        refracted = ni_over_nt * (uv - n*dt) - n * sqrtf(disc);
        return true;
    }
    return false;
}

__device__ inline float schlick(float cosine, float ref_idx) 
{
    float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * powf(1.0f - cosine, 5.0f);
}

// ---------------- base material ----------------
class material 
{
public:
    __device__ virtual ~material() {}
    __device__ virtual vec3 emitted(float /*u*/, float /*v*/, const vec3& /*p*/) const 
    {
        return vec3(0.f, 0.f, 0.f);
    }
    __device__ virtual bool scatter(
        const ray& r_in, const hit_record& rec,
        vec3& attenuation, ray& scattered,
        curandState* rng
    ) const = 0;
};

// ---------------- lambertian (now texture-backed) ----------------
class lambertian : public material {
public:
    texture* tex; // not owning
    __device__ ~lambertian() override { if (tex) delete tex; }

    // convenience ctor: solid color
    __device__ lambertian(const vec3& albedo)
        : tex(new solid_color(albedo)) {}

    // texture-backed ctor
    __device__ lambertian(texture* t)
        : tex(t) {}

    __device__ bool scatter(
        const ray& r_in, const hit_record& rec,
        vec3& attenuation, ray& scattered,
        curandState* rng
    ) const override 
    {
        // classic diffuse from the gpu tutorial: p + n + random_in_unit_sphere
        vec3 target = rec.p + rec.normal + random_in_unit_sphere(rng);
        scattered   = ray(rec.p, target - rec.p, r_in.time());
        attenuation = tex ? tex->value(rec.u, rec.v, rec.p) : vec3(1,1,1);
        return true;
    }
};

// ---------------- metal ----------------
class metal : public material 
{
public:
    vec3  albedo;
    float fuzz;

    __device__ metal(const vec3& a, float f)
        : albedo(a), fuzz(f < 1.0f ? f : 1.0f) {}

    __device__ bool scatter(
        const ray& r_in, const hit_record& rec,
        vec3& attenuation, ray& scattered,
        curandState* rng
    ) const override 
    {
        vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
        scattered = ray(rec.p, reflected + fuzz * random_in_unit_sphere(rng), r_in.time());
        attenuation = albedo;
        return (dot(scattered.direction(), rec.normal) > 0.0f);
    }
};

// ---------------- dielectric (no front_face path) ----------------
class dielectric : public material 
{
public:
    float ref_idx;
    __device__ dielectric(float ri) : ref_idx(ri) {}

    __device__ bool scatter(
        const ray& r_in, const hit_record& rec,
        vec3& attenuation, ray& scattered,
        curandState* rng
    ) const override 
    {
        vec3 outward_normal;
        vec3 reflected = reflect(r_in.direction(), rec.normal);
        float ni_over_nt;
        attenuation = vec3(1.0f, 1.0f, 1.0f);
        vec3 refracted;
        float reflect_prob;
        float cosine;

        if (dot(r_in.direction(), rec.normal) > 0.0f) 
        {
            // inside the surface
            outward_normal = -rec.normal;
            ni_over_nt = ref_idx;
            cosine = dot(r_in.direction(), rec.normal) / r_in.direction().length();
            cosine = sqrtf(fmaxf(0.0f, 1.0f - ref_idx*ref_idx*(1.0f - cosine*cosine)));
        } else 
        {
            // outside the surface
            outward_normal = rec.normal;
            ni_over_nt = 1.0f / ref_idx;
            cosine = -dot(r_in.direction(), rec.normal) / r_in.direction().length();
        }

        if (refract(r_in.direction(), outward_normal, ni_over_nt, refracted))
            reflect_prob = schlick(cosine, ref_idx);
        else
            reflect_prob = 1.0f;

        if (curand_uniform(rng) < reflect_prob)
            scattered = ray(rec.p, reflected, r_in.time());
        else
            scattered = ray(rec.p, refracted, r_in.time());

        return true;
    }
};

class diffuse_light : public material 
{
public:
    __device__ diffuse_light(texture* t) : tex(t) {}
    __device__ diffuse_light(const vec3& c) : tex(nullptr), solid(c) {}
    __device__ ~diffuse_light() override { if (tex) delete tex; }

    __device__ vec3 emitted(float u, float v, const vec3& p) const override 
    {
        return tex ? tex->value(u, v, p) : solid;
    }

    // lights don’t scatter
    __device__ bool scatter(const ray&, const hit_record&, vec3&, ray&, curandState*) const override 
    {
        return false;
    }

private:
    texture* tex;     // optional
    vec3     solid;   // used if tex == nullptr
};

class isotropic : public material 
{
public:
    texture* tex; // owns
    __device__ isotropic(texture* t) : tex(t) {}
    __device__ isotropic(const vec3& c) : tex(new solid_color(c)) {}
    __device__ ~isotropic() override { if (tex) delete tex; }

    __device__ bool scatter(const ray& r_in, const hit_record& rec,
                            vec3& attenuation, ray& scattered,
                            curandState* rng) const override {
        // Sample random direction uniformly over the sphere
        scattered   = ray(rec.p, random_in_unit_sphere(rng), r_in.time());
        attenuation = tex->value(rec.u, rec.v, rec.p);
        return true;
    }
};