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
- **Host-side scene build**: geometry is POD in flat arrays, so the scene and its BVH are built on the CPU and memcpy'd over. This is only possible because the primitives are not polymorphic -- a vtable pointer is a device address, so an object with virtual functions cannot be built on the host and copied to the device. That single constraint is what forced the original design to build everything in a `<<<1,1>>>` kernel.
- **Tagged dispatch**: `hit_prim` switches on a small enum instead of making an indirect call, so intersection inlines and a warp straddling two primitive types costs a predicated branch rather than two serialised call targets.
- **The camera is passed by value**, not through a `camera**`. It has no virtual functions, so there was never a reason to allocate it on the device.
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

- **Materials and textures are still a device-only virtual hierarchy.** They are
  built by a small kernel from host descriptors and referenced by index, so they
  no longer block host-side scene construction, but shading still pays an
  indirect call. There are tens of them per scene rather than thousands of
  primitives, so this is not where the time goes.
- **The BVH uses a median split, not a surface-area heuristic.** That is what the
  book does. SAH would give better traversal on the box-heavy scenes; it would
  also stop the tree matching the book's.
- **One primitive per leaf.** Small leaves mean more nodes and more box tests.
  Packing 2-4 primitives per leaf is usually a win and is easy from here.
- **Instancing is baked, not shared.** A rotated box becomes six transformed
  quads, so N copies of a mesh cost N copies of the geometry. Fine at this scale
  (the largest scene is 3409 primitives); it would not be for a real asset.
- **`bouncing` uses a host xorshift** rather than cuRAND for scene layout, so its
  sphere placement differs from the pre-refactor version. The scene is now
  reproducible from `--seed` without a GPU, which it was not before.

---

## Source layout

```
src/
  core/     Pure math and POD, dual-compiled. vec3/ray/aabb, primitives and
            their intersection routines, the flattened BVH and its traversal,
            the camera, Perlin noise. Compiled by nvcc for the renderer and by
            a plain host compiler for the tests -- cuda_compat.hpp defines the
            __host__/__device__ annotations away when nvcc is not driving the
            build, so the boundary is enforced by the build rather than by
            convention.
  host/     Scene authoring. Book-style C++ that emits flat arrays: the scene
            descriptions, the BVH builder, and the material/texture descriptor
            records.
  scene/    What genuinely has to live on the device: the material and texture
            class hierarchy, the kernel that instantiates it from descriptors,
            and the scene upload.
  io/       Image loading and asset path resolution.
  main.cu   Kernels, CLI, render driver.
```

### How a scene becomes pixels

1. `host/scenes.hpp` builds `std::vector<Sphere|Quad|Medium>` plus a material
   descriptor list, on the CPU. Instancing transforms are applied to the
   geometry here, once, instead of to every ray at trace time.
2. `SceneBuilder::build_bvh()` builds a flattened BVH with `std::nth_element`.
3. `upload_scene()` memcpys the arrays over -- possible only because they are
   POD -- and runs one small kernel to instantiate the materials.
4. `render_accumulate` traverses with an explicit stack and dispatches on a
   primitive tag.

---

## Acknowledgements

- Peter Shirley et al. for the **Ray Tracing in One Weekend** series.  
- NVIDIA Developer Blog for the CUDA adaptation guidance.

---
