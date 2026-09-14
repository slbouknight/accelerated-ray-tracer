#pragma once

#include "cuda_compat.hpp"
#include "vec3.hpp"

class ray
{
    public:
        RT_HD ray() {}
        RT_HD ray(const vec3& a, const vec3& b, float time) { A=a; B=b; tm=time;}
        RT_HD ray(const vec3& a, const vec3& b) { A=a; B=b; tm=0.0f;}

        RT_HD vec3 origin() const { return A; }
        RT_HD vec3 direction() const { return B; }
        RT_HD float time() const {return tm; }

        // float, not double: sphere::hit calls this once per intersection test,
        // and a double `t` promoted the whole expression to FP64.
        RT_HD vec3 point_at_parameter(float t) const { return A + t*B; } 

        vec3 A;
        vec3 B;
        float tm;
};

