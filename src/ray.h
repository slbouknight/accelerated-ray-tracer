#ifndef RAYH
#define RAYH
#include "vec3.h"

class ray
{
    public:
        __host__ __device__ ray() {}
        __host__ __device__ ray(const vec3& a, const vec3& b, double time) { A=a; B=b; tm=time;}
        __host__ __device__ ray(const vec3& a, const vec3& b) { A=a; B=b; tm=0;}

        __host__ __device__ vec3 origin() const { return A; }
        __host__ __device__ vec3 direction() const { return B; }
        __host__ __device__ double time() const {return tm; }

        __host__ __device__ vec3 point_at_parameter(double t) const { return A + t*B; } 

        vec3 A;
        vec3 B;
        double tm;
};

#endif