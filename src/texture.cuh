// texture.cuh
#pragma once
#include "vec3.cuh"
#include "image_io.h" // for DeviceImage only

__host__ __device__ inline float clamp01(float x){ return x<0?0:x>1?1:x; }

class texture {
public:
    __device__ virtual ~texture() {}
    __device__ virtual vec3 value(float u, float v, const vec3& p) const = 0;
};

class solid_color : public texture 
{
public:
    vec3 albedo;
    __device__ solid_color() : albedo(0,0,0) {}
    __device__ solid_color(const vec3& a) : albedo(a) {}
    __device__ vec3 value(float, float, const vec3&) const override { return albedo; }
};

class checker_texture : public texture 
{
public:
    float inv_scale = 1.f;
    texture* even = nullptr; 
    texture* odd = nullptr;
    __device__ checker_texture() {}
    __device__ checker_texture(float scale, texture* e, texture* o)
        : inv_scale(1.f/scale), even(e), odd(o) {}
     __device__ ~checker_texture() override { delete even; delete odd; }
    __device__ vec3 value(float u, float v, const vec3& p) const override 
    {
        int xi = int(floorf(inv_scale*p.x()));
        int yi = int(floorf(inv_scale*p.y()));
        int zi = int(floorf(inv_scale*p.z()));
        bool isEven = ((xi+yi+zi) & 1) == 0;
        return isEven ? even->value(u,v,p) : odd->value(u,v,p);
    }
};

class image_texture : public texture 
{
public:
    DeviceImage img;
    __device__ image_texture() {}
    __device__ image_texture(const DeviceImage& dimg) : img(dimg) {}
    __device__ vec3 value(float u, float v, const vec3&) const override {
        if (!img.valid()) return vec3(0,1,1);
        u = clamp01(u); v = clamp01(v);
        int i = min(int(u * img.width),  img.width  - 1);
        int j = min(int((1.f - v) * img.height), img.height - 1);
        int idx = (j * img.width + i) * img.bpp;
        const float inv255 = 1.f/255.f;
        return vec3(inv255*img.data[idx+0], inv255*img.data[idx+1], inv255*img.data[idx+2]);
    }
};
