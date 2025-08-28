#pragma once

#include "image_io.h" // for DeviceImage only
#include "perlin.cuh"
#include "vec3.cuh"

__host__ __device__ inline float clamp01(float x){ return x<0?0:x>1?1:x; }

class texture 
{
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

class noise_texture : public texture 
{
    public:
        __device__ explicit noise_texture(float scale) : scale(scale) {}

        __device__ vec3 value(float /*u*/, float /*v*/, const vec3& p) const override {
            // Same as your serial: 0.5 * (1 + sin(scale*z + 10*turb(p,7)))
            float s = __sinf(scale * p.z() + 10.0f * perlin::turb(p, 7));
            float t = 0.5f * (1.0f + s);
            return vec3(t, t, t);
        }

    private:
        float scale;
};

__host__ __device__ inline float smoothstep(float edge0, float edge1, float x) {
    // Scale/bias x to 0..1 then use cubic Hermite (3t^2 - 2t^3)
    float t = clamp01((x - edge0) / (edge1 - edge0));
    return t * t * (3.0f - 2.0f * t);
}

class noodle_texture : public texture {
public:
    __device__ noodle_texture(float stripes_k=3.0f, float wiggle_amp=3.0f,
                              float wiggle_freq=0.6f, int oct=3,
                              vec3 dir=vec3(0,0,1),
                              vec3 noodle=vec3(0.92f,0.85f,0.65f),   // noodle color
                              vec3 gap=vec3(0.35f,0.20f,0.10f))      // gaps/broth
    : k(stripes_k), A(wiggle_amp), f(wiggle_freq), octaves(oct),
      d(unit_vector(dir)), cN(noodle), cG(gap) {}

    __device__ vec3 value(float, float, const vec3& p) const override {
        float u = dot(p, d);                          // stripe axis (change d to rotate)
        float wig = perlin::turb(p * f, octaves);     // smooth warping field
        float stripes = fabsf(__sinf(k * u + A * wig));
        float t = smoothstep(0.75f, 0.98f, stripes);  // thin bright strands
        return (1.f - t) * cG + t * cN;
    }
private:
    float k, A, f; int octaves; vec3 d, cN, cG;
};


// --- Green felt texture (subtle mottling, no rings) ---
// Uses perlin::noise(vec3) and perlin::turb(vec3, int depth)
// If your perlin::noise returns [-1,1], define FELT_PERLIN_IS_SIGNED.
class felt_texture : public texture {
public:
    __device__ felt_texture(
        const vec3& base      = vec3(0.06f, 0.36f, 0.18f), // billiards felt green
        float mottling_scale  = 16.0f,                     // higher = finer grain
        float mottling_amt    = 0.08f,                     // 0.0–0.2 subtle brightness variation
        float fiber_scale     = 4.0f,                      // directional variation scale
        float fiber_amt       = 0.03f                      // very slight
    )
    : base_col(base),
      m_scale(mottling_scale), m_amt(mottling_amt),
      f_scale(fiber_scale),    f_amt(fiber_amt) {}

    __device__ vec3 value(float /*u*/, float /*v*/, const vec3& p) const override {
        // --- isotropic fine mottling ---
        float m = perlin::noise(p * m_scale);
        #ifdef FELT_PERLIN_IS_SIGNED
            // Map [-1,1] -> [0,1]
            m = 0.5f * (m + 1.0f);
        #endif

        // --- faint directional fibers (along X), slightly perturbed by turbulence ---
        // Keep this in [0,1] using sin()
        float phase = p.x() * f_scale + 2.0f * perlin::turb(p * 0.5f, 2);
        float fibers = 0.5f * (1.0f + __sinf(phase));

        // Combine (keep subtle)
        float gain = 1.0f + m_amt * (m - 0.5f) + f_amt * (fibers - 0.5f);

        // Clamp to a sane range so the felt doesn’t blow out or go black
        gain = fminf(fmaxf(gain, 0.7f), 1.2f);

        return base_col * gain;
    }

private:
    vec3  base_col;
    float m_scale, m_amt;
    float f_scale, f_amt;
};

// texture_uv_offset.cuh
class uv_offset_texture : public texture {
public:
    __device__ uv_offset_texture(texture* base, float u_offset_turns, float v_offset = 0.f)
        : base_(base), du(u_offset_turns), dv(v_offset) {}

    __device__ vec3 value(float u, float v, const vec3& p) const override {
        float uu = u + du;  uu -= floorf(uu);            // wrap to [0,1)
        float vv = v + dv;  vv = fminf(fmaxf(vv,0.f),1.f); // (keep v clamped)
        return base_->value(uu, vv, p);
    }
private:
    texture* base_;
    float du, dv; // du in “turns” (1.0 = 360°)
};
