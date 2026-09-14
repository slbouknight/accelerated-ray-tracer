# Tests

```bat
cmake -S . -B build
cmake --build build --config Release --parallel
ctest --test-dir build -C Release --output-on-failure
```

Three tiers, ordered cheapest-and-most-specific first. When something breaks
you want the failure to come from the lowest tier that can see it, because that
is the one that tells you *where*.

| tier | file | needs a GPU | runtime | what it protects |
|---|---|---|---|---|
| host math | `test_math.cpp` | no | <1 s | `vec3`, `ray`, `aabb`, reflect/refract/schlick, sphere UV, Perlin |
| host geometry | `test_geometry.cpp` | no | <1 s | sphere/quad/box/medium intersection, baked instancing, **BVH vs brute force**, camera, every scene's structure |
| device | `test_device.cu` | yes | <1 s | materials, textures, the host-descriptor to device-object material table |
| golden image | `run_golden.py` | yes | ~3 s | all ten scenes, end to end |

```bat
ctest --test-dir build -C Release -L unit          REM skip the golden test
buildin\Release	est_geometry.exe bvh          REM filter by substring
```

Most of this used to require a GPU. Geometry, instancing, BVH traversal and the
camera were all `__device__` virtuals built with device `new`, so a kernel
launch was the only way to reach them. They are POD now, and the same code the
renderer runs is exercised by plain C++ binaries -- which is why the suite went
from ~10 s to ~3 s, and why a failure gives you a debugger instead of a trap.

---

## Tier 1 and 2 — host (`test_math.cpp`, `test_geometry.cpp`)

Plain C++ targets: not compiled by nvcc, not linked against cudart.
`src/core/cuda_compat.hpp` defines `__host__`/`__device__` away when nvcc is not
driving the build, so every header under `src/core` has to be genuinely free of
device-only constructs. If a `__sinf`, a `curandState` or a device `new` leaks
into one, these targets stop compiling and name the file. The boundary is
enforced by the build rather than by a comment.

They are also the only targets built with `/W4` (or `-Wall -Wextra`), which has
already caught an unguarded `#pragma unroll` and two dead internal-linkage
functions.

The load-bearing test is `bvh.traversal_agrees_with_brute_force`. A BVH is an
acceleration structure: it is only allowed to make the *same* answer arrive
faster. It builds 96 spheres, fires 512 rays, and checks that traversal returns
the same closest-hit `t` and material as a linear scan. It also asserts that a
useful fraction of rays actually hit, so it cannot pass trivially by missing
everything.

`scenes.every_scene_builds_a_consistent_bvh` checks all ten scenes for the
invariants that would otherwise fail silently on the GPU: exactly `2n-1` nodes,
every primitive referenced by exactly one leaf, and every material id in range.

## Tier 3 — device (`test_device.cu`)

Only what has to be here: materials and textures are still a virtual hierarchy,
and a vtable pointer is a device address, so they can only be built and called
on the device.

A single-thread kernel computes and writes primitive floats into a managed
buffer; **the host does the asserting**, with the same harness as the host
tiers, so a failure prints real values and a line number.

`material_table.*` covers the bridge between host descriptors and device
objects. If an index is mishandled there, every primitive in the scene gets the
wrong look.

## Tier 4 — golden images (`run_golden.py`)

Renders all ten scenes at 128px, 24 spp, fixed seed, and diffs against
`tests/golden/*.ppm`.

This works because **the renderer is bit-for-bit reproducible for a given seed,
resolution, sample count and binary**:

```bat
rayTracer --scene cornell --width 120 --height 120 --spp 16 --seed 99 --out a.ppm
rayTracer --scene cornell --width 120 --height 120 --spp 16 --seed 99 --out b.ppm
python tools\ppm_compare.py a.ppm b.ppm        REM PSNR inf, max delta 0
```

**What this will not survive, by design:** changing the RNG, changing how many
random draws happen per bounce, reordering samples, switching float/double, or
moving to a different GPU whose compiler makes different fusion choices. Those
are genuine changes to the output. Re-baseline with

```bat
python tests
un_golden.py --exe buildin\Release
ayTracer.exe --update
python tools\ppm_to_png.py tests\golden\*.ppm --scale 3    REM then look at them
```

`--update` accepts whatever it renders. Always eyeball the images before
committing, or the "regression test" quietly becomes a record of the bug.

A caution learned the hard way: at 24 spp these references are extremely noisy,
and a low-contrast feature (the smoke volumes in `cornell_smoke`) can look
*missing* when it is merely buried in noise. Judge correctness at a few hundred
spp; use the goldens to detect *change*, not to assess quality.

---

## The other safety net: compute-sanitizer

The test suite cannot see a leak or a use-after-free that does not change the
image. This can:

```bat
compute-sanitizer --tool memcheck --leak-check full ^
  build\bin\Release\rayTracer.exe --scene cornell --width 64 --height 64 --spp 4 --quiet --out nul.ppm
```

Current expected state: **0 errors, 0 bytes leaked**, on every scene. Geometry
is POD in flat arrays freed with `cudaFree`, and every material and texture is
owned by exactly one table, so there is no ownership question to get wrong. Any
non-zero number here is a regression, not a known issue.

Also worth running periodically:

```bat
compute-sanitizer --tool racecheck  build\bin\Release\rayTracer.exe --scene quads --width 64 --height 64 --spp 2 --quiet --out nul.ppm
compute-sanitizer --tool initcheck  build\bin\Release\rayTracer.exe --scene quads --width 64 --height 64 --spp 2 --quiet --out nul.ppm
```

## Adding a test

```cpp
TEST(suite_name, what_it_asserts) {
    RT_CHECK(cond);                          // non-fatal
    RT_CHECK_NEAR(got, want, tol);
    RT_CHECK_VEC(v, x, y, z, tol);           // anything with .x()/.y()/.z()
    RT_REQUIRE(ptr != nullptr);              // fatal: abandons this case
}
```

Checks are non-fatal on purpose, so one broken invariant does not hide the five
behind it. The harness (`test_harness.h`) has no dependencies and compiles with
both `nvcc` and a plain host C++17 compiler.

Put a test on the host if you can. If the thing under test needs `curandState`,
device `new`, or virtual dispatch, it has to go in `test_device.cu` — and that
is a signal worth noticing, not just a routing decision. After the flattening
work the only things left on that side are materials and textures.
