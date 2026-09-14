#pragma once

// Lets the pure-math headers in src/core compile with a plain host C++ compiler
// as well as with nvcc.
//
// Why bother, when nvcc compiles host code perfectly well? Because it turns a
// naming convention into something the build actually enforces. src/core is
// compiled twice: once by nvcc into the renderer, and once by MSVC/gcc into the
// host test binary. The moment a device-only construct leaks into a core header
// -- a __device__-only intrinsic like __sinf, a curandState, a device `new` --
// the host build stops compiling. Without this, "this header is pure math" is
// just a comment that slowly stops being true.
//
// The split:
//   src/core/*.hpp    pure math, no RNG state, no allocation, no virtuals.
//                     Dual-compiled, unit-tested on the host.
//   src/scene/*.cuh   the object model: virtual hittables/materials/textures
//                     built with device `new`. nvcc only.
//   src/io/*.hpp      host-side file and image handling.

#if defined(__CUDACC__)
  #include <cuda_runtime.h>
#else
  // A host compiler has never heard of these. Defining them away is enough:
  // every annotated function in src/core is ordinary C++ underneath.
  #define __host__
  #define __device__
  #define __global__
  #define __forceinline__ inline
  #define __restrict__

  // fminf/fmaxf/sqrtf/floorf/fabsf are standard C and need no shim. Device-only
  // intrinsics (__sinf, __fdividef, __float_as_uint, ...) are deliberately NOT
  // shimmed -- using one in src/core should be a compile error, not a silent
  // fallback to a different-precision host implementation.
  #include <cmath>
  #include <cstdlib>
#endif

// Marks a function as callable from both sides. Shorter than writing both
// annotations, and greppable when you want to audit the boundary.
#define RT_HD __host__ __device__

// `#pragma unroll` is an nvcc directive; MSVC warns C4068 (unknown pragma) and
// gcc/clang want their own spelling. Wrapping it in _Pragma keeps the hint for
// the device compiler without making the host build noisy.
#if defined(__CUDACC__)
  #define RT_UNROLL _Pragma("unroll")
#else
  #define RT_UNROLL
#endif
