# CUDA Accelerated Ray Tracer

A CUDA implementation of the **_Ray Tracing in One Weekend_** Book Series with per-pixel RNG, thin-lens depth of field, metal/dielectric materials, procedural textures, <br>
quads, instancing, light sources, Cornell box, texture mapping, and the randomized final scene. The code mirrors the books’ progression, adapted for GPU kernels and<br>
CUDA memory/launch patterns.

<p align="left">
  <img src="images/alfredo2.png" alt="Demo" height="800"/>
</p>

> References  
> • Peter Shirley, _Ray Tracing in One Weekend_ Book Series — https://raytracing.github.io/  
> • NVIDIA Developer Blog, “Accelerated Ray Tracing in One Weekend in CUDA” — https://developer.nvidia.com/blog/accelerated-ray-tracing-cuda/

---

## Features

### Rendering pipeline
- **Antialiasing via stochastic supersampling** (per-pixel jitter using cuRAND)  
  <p align="left">
    <img src="images/anti-aliasing.png" alt="Anti-aliasing" width="600"/>
  </p>

- **Gamma correction** with configurable values
  <p align="left">
    <img src="images/gamma-correction.png" alt="Gamma correction" width="600"/>
  </p>

### Materials
- **Metal** with **fuzz** parameter for rough/microfacet reflections  
- **Lambertian** (diffuse) with random cosine-ish scattering  
- **Dielectric** (glass): **refraction**, **total internal reflection**, and **Schlick** reflectance  
  <p align="left">
    <img src="images/materials.png" alt="Materials" width="600"/>
  </p>

### Camera
- **Thin-lens defocus blur (depth of field)** (aperture + focus distance)
- Proper **camera basis** (`u/v/w`) with configurable FOV and aspect ratio
  <p align="left">
      <img src="images/defocus.png" alt="Depth of Field" width="600"/>
  </p>

### Procedural Textures
- **Checkered planes and spheres**
- **Perlin noise textures** with turbulence & marble patterns  
<p align="left">
  <img src="images/checkered.png" alt="Checkered texture" width="400"/>
  <img src="images/perlin.png" alt="Perlin texture" width="400"/>
</p>

### Motion Blur
- **Shutter interval sampling**: each ray carries a randomized time in `[time0, time1]`
- **Animated primitives**: object positions are interpolated across the shutter window
- Produces natural blur trails when geometry moves during exposure
<p align="left">
    <img src="images/utk.png" alt="Motion blur" width="400"/>
</p>

### Light Sources
- **Emissive materials** for area lights
- Example: Cornell box ceiling quad + glowing sphere  
<p align="left">
  <img src="images/poolBall.png" alt="Simple Light" width="400"/>
</p>

### Quads & Rectangular Geometry
- General **quad primitive**
- Used for walls, floors, ceilings, and light sources  
<p align="left">
  <img src="images/quads.png" alt="Quads" width="400"/>
</p>

### Instancing & Object Transforms
- Translate / rotate geometry without duplicating vertex data
- Used to place rotated blocks in the Cornell Box  
<p align="left">
  <img src="images/redBlue.png" alt="Instancing" width="400"/>
</p>

### Texture Mapping
- Spherical coordinate texture mapping
- Example: Earth texture on a sphere  
<p align="left">
  <img src="images/textureWrap.png" alt="Texture Mapping" width="400"/>
</p>

---

## Requirements

- **Windows** with **Visual Studio 2022** (MSVC toolset)
- **CUDA Toolkit 12+** (13.x tested)
- **CMake 3.25+**
- **Python 3** (optional — for the golden-image test and the PPM tools)
- Any NVIDIA GPU supported by your CUDA toolkit

`CMAKE_CUDA_ARCHITECTURES` defaults to `native`, so nvcc targets the GPU it
finds at configure time. For a portable fat binary, override it:
`cmake -S . -B build -DRT_CUDA_ARCH="75;86;89;120"`.

---

## Build & Run

### Quick start (Windows)

Configures, builds **Release**, runs the tests, then prompts for a scene and an
output name:

```bat
setup.bat
```

### Manual CMake build

```bat
cmake -S . -B build
cmake --build build --config Release --parallel
ctest --test-dir build -C Release --output-on-failure
```

> Build **Release**. A Debug CUDA build compiles device code with `-G`, which
> disables almost all device-side optimisation — useful under `cuda-gdb`, never
> for timing.

### Running

```bat
build\bin\Release\rayTracer.exe --list
build\bin\Release\rayTracer.exe --scene cornell --out cornell.ppm
build\bin\Release\rayTracer.exe --scene original --width 400 --height 400 --spp 500 --out preview.ppm
```

| flag | meaning |
|---|---|
| `--scene NAME` | which scene to render (`--list` to see them) |
| `--width N` `--height N` | override the scene's default resolution |
| `--spp N` | samples per pixel |
| `--max-depth N` | maximum ray bounces (default 50) |
| `--gamma F` | gamma exponent (default 2.2) |
| `--seed N` | RNG seed — same seed + same params gives a bit-identical image |
| `--block X Y` | thread-block dimensions (default `8 8`) |
| `--batch N` | samples per kernel launch (default 64, `0` = all at once) |
| `--out FILE` | write here instead of stdout |
| `--stats FILE` | write timing JSON (`build_ms`, `render_ms`, …) |
| `--ascii` | emit P3 text PPM instead of P6 binary |
| `--render-stack N` | shrink the per-thread stack before rendering (default `0` = off) |
| `--quiet` | suppress the progress line |

The renderer writes **binary PPM (P6)** by default; `--ascii` gives the P3 text
form the book uses. Progress and timing go to `stderr`, so `rayTracer.exe
--scene cornell > out.ppm` still works.

`--batch` exists because Windows kills any kernel that runs longer than the
display watchdog timeout (TDR, 2 s by default). Issuing the sample budget as
several short launches is mathematically identical to one long one — the RNG
state carries across launches — and it gives a progress indicator for free.

### Viewing the output

```bat
python tools\ppm_to_png.py output.ppm              REM no dependencies
python tools\ppm_to_png.py output.ppm --scale 4    REM nearest-neighbour zoom
```

or open the `.ppm` directly in GIMP, or `magick convert output.ppm output.png`.

---

## Testing & benchmarking

```bat
ctest --test-dir build -C Release --output-on-failure    REM unit + golden image
python bench\run_bench.py --config bench\targets.json    REM timing matrix
```

See [tests/README.md](tests/README.md) and [bench/README.md](bench/README.md).

---

## How it works (GPU notes)

- **Bounded depth instead of recursion**: the book’s recursive `color()` is turned into a loop (default max depth = 50) to avoid device stack overflows.
- **Per-pixel RNG**: Each thread has a `curandState`. We copy the state to a local variable, sample multiple times, then write it back.
- **Unified memory for the framebuffer** (`cudaMallocManaged`) to simplify host readout (`stdout` → PPM).
- **Device-side scene build**: A small kernel constructs the world and camera once, then the main render kernel traces rays. This is forced by the design rather than chosen: a vtable pointer is a device address, so a polymorphic object cannot be built on the host and copied over.
- **Iterative BVH traversal**: an explicit stack rather than recursion. Recursing cost two `hit_record`s per level in *local* memory (which is global memory with a per-thread address), and could not use a hit in the near child to prune the far one until after the far one had been traversed. Worth 7-13x on scenes with real geometry.
- **Batched sample accumulation**: `render_accumulate` adds samples into a buffer; `resolve` normalises and gamma-encodes once at the end. Identical math to one long launch, but each launch stays under the Windows display watchdog.
- **Thin-lens DOF**: `random_in_unit_disk` samples the aperture; `lower_left_corner`, `horizontal`, and `vertical` are scaled by the **focus distance**; `lens_radius = aperture/2`.

---

## Tuning

- **Samples per pixel** (`--spp`): higher → cleaner images, time ∝ spp.
- **Resolution** (`--width`/`--height`): increase for detail.
- **Max depth** (`--max-depth`): 50 is a good default; raising it gives diminishing returns.
- **Aperture**: small (`0.1`) = subtle blur; large (`2.0`) = strong DOF (needs more spp). Per-scene, in `src/main.cu`.
- **Build config**: **Release**, always, for anything you intend to time.
- **Block size** (`--block`): `8 8` (64 threads) measured fastest on an RTX 5070; `16 8` and `8 4` were within 1%, `16 16` was 9% slower. Worth re-sweeping on a different GPU, but the default is already a good one.
- **Batch size** (`--batch`): only affects watchdog headroom and progress granularity, not the result. Lower it if a single launch still trips TDR.

---

## Known limitations

Honest list of what the current implementation still does badly, in rough order
of cost. (Traversal was the big one and is fixed -- see the layout note below.)

- **Scene construction runs on one CUDA thread.** `create_world_*` launches as
  `<<<1,1>>>`, and the BVH *build* still uses an O(n^2) selection sort inside a
  recursive constructor. Measured: 10 objects -> 21 ms, 488 -> 163 ms,
  1409 -> **1590 ms**. For the `final` scene the build is now a larger share of
  total time than it was, because the render got 12x faster and the build did
  not move at all.
- **Everything is a `__device__` virtual allocated with device `new`.** Objects
  land scattered across the device malloc heap, so traversal is a pointer chase
  through uncoalesced global memory, and every `hit()` is an indirect call that
  cannot inline and serialises when lanes in a warp hit different types.
- **The recursive BVH build still forces a deep stack reservation**
  (`cudaLimitStackSize`, 16-64 KB per thread). Traversal no longer needs it --
  `--render-stack` can shrink it before the render kernel -- but on an RTX 5070
  that measured inside the noise, and going below ~2 KB faults.
- **Instancing wrappers leak.** `translate` and `rotate_y` do not delete the
  object they wrap, and materials shared between primitives are marked
  non-owning to avoid a double free, so neither is ever freed.
  `compute-sanitizer --leak-check full` reports ~19 leaked allocations for the
  Cornell scene, and **0 invalid accesses**.
- **The instancing wrappers drop the RNG.** `translate`/`rotate_y`/`with_material`
  forward their 5-argument `hit` to the 4-argument one. No current scene nests a
  `constant_medium` under an instance wrapper, so this is latent rather than
  live, but it would silently degrade a volume that was.

---

## Source layout

```
src/
  core/     pure math. Dual-compiled: nvcc for the renderer, and a plain host
            C++ compiler for tests/test_math.cpp. No cuRAND state, no device
            allocation, no virtuals. cuda_compat.hpp defines the annotations
            away for host builds, so this boundary is enforced by the build
            rather than by convention.
  scene/    the device-only object model: hittables, materials, textures,
            camera, BVH. Virtual dispatch and device `new` live here. nvcc only.
  io/       host-side image loading and asset path resolution.
  main.cu   kernels, scene table, CLI, render driver.
```

---

## Acknowledgements

- Peter Shirley et al. for the **Ray Tracing in One Weekend** series.  
- NVIDIA Developer Blog for the CUDA adaptation guidance.

---
