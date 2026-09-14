# Tests

```bat
cmake -S . -B build
cmake --build build --config Release --parallel
ctest --test-dir build -C Release --output-on-failure
```

Three tiers, deliberately ordered cheapest-and-most-specific first. When
something breaks you want the failure to come from the lowest tier that can
see it, because that is the one that tells you *where*.

| tier | file | needs a GPU | runtime | what it protects |
|---|---|---|---|---|
| host math | `test_math.cpp` | no | <1 s | `vec3`, `ray`, `aabb`, reflect/refract/schlick, sphere UV, Perlin |
| device | `test_device.cu` | yes | ~1 s | `sphere`/`quad`/`box` intersection, instancing, **BVH vs brute force**, materials, camera |
| golden image | `run_golden.py` | yes | ~10 s | all ten scenes, end to end |

```bat
ctest --test-dir build -C Release -L unit          REM skip the slow golden test
build\bin\Release\test_device.exe bvh              REM filter by substring
```

---

## Tier 1 — host math (`test_math.cpp`)

A **plain C++ target** -- not compiled by nvcc, not linked against cudart.
`src/core/cuda_compat.hpp` defines `__host__`/`__device__` away when nvcc is not
driving the build, so every header under `src/core` has to be genuinely free of
device-only constructs. If a `__sinf`, a `curandState` or a device `new` leaks
into one, this target stops compiling and names the file. The boundary is
enforced by the build rather than by a comment.

It is also built with `/W4` (or `-Wall -Wextra`), which the nvcc targets are
not -- that alone caught an unguarded `#pragma unroll` and two dead
internal-linkage functions in `aabb.hpp`.

This file being small relative to `test_device.cu` is itself a finding: most of
the renderer is still not host-testable, because the geometry and material
hierarchies are `__device__`-only virtuals built with device `new`. Moving scene
construction host-side should migrate most of tier 2 into tier 1.

## Tier 2 — device (`test_device.cu`)

Pattern: a single-thread kernel computes and writes primitive floats into a
managed buffer; **the host does the asserting**, using the same harness as tier
1. Keeping assertions host-side means a failure prints real values and a line
number instead of a device-side trap.

The important one is `bvh.traversal_agrees_with_brute_force`. A BVH is an
acceleration structure — it is only allowed to make the *same* answer arrive
faster. It builds 96 spheres, fires 256 rays, and checks that BVH traversal
returns the same closest-hit `t` as a linear scan over every object. Any rewrite
(iterative traversal, SAH splits, flattened nodes, host-side construction) has
to keep it green. It also asserts that a useful fraction of rays actually hit,
so it cannot pass trivially by missing everything.

## Tier 3 — golden images (`run_golden.py`)

Renders all ten scenes at 128px, 24 spp, fixed seed, and diffs against
`tests/golden/*.ppm`.

This works because **the renderer is bit-for-bit reproducible for a given seed,
resolution, sample count and binary** — verified above, and worth re-verifying
if it ever seems not to be:

```bat
rayTracer --scene cornell --width 120 --height 120 --spp 16 --seed 99 --out a.ppm
rayTracer --scene cornell --width 120 --height 120 --spp 16 --seed 99 --out b.ppm
python tools\ppm_compare.py a.ppm b.ppm        REM PSNR inf, max delta 0
```

**What this will not survive, by design:** changing the RNG, changing how many
random draws happen per bounce, reordering samples, switching float/double, or
moving to a different GPU architecture whose compiler makes different fusion
choices. Those are all genuine changes to the output. Re-baseline with

```bat
python tests\run_golden.py --exe build\bin\Release\rayTracer.exe --update
python tools\ppm_to_png.py tests\golden\*.ppm --scale 3    REM then look at them
```

`--update` accepts whatever it renders. Always eyeball the images before
committing, or the "regression test" quietly becomes a record of the bug.

---

## The other safety net: compute-sanitizer

The test suite cannot see a leak or a use-after-free that does not change the
image. This can:

```bat
compute-sanitizer --tool memcheck --leak-check full ^
  build\bin\Release\rayTracer.exe --scene cornell --width 64 --height 64 --spp 4 --quiet --out nul.ppm
```

Current expected state: **0 invalid accesses, ~19 leaked allocations** in the
Cornell scene. Those leaks are known and documented in the source — `translate`
and `rotate_y` do not delete the object they wrap, and materials shared between
primitives are deliberately marked non-owning to avoid a double free. Both
disappear when scene ownership moves into flat host-side arrays. If the leak
count goes *up*, something new is wrong; if an invalid access appears, stop.

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

Put a test in tier 1 if you can. If the thing under test needs `curandState`,
device `new`, or virtual dispatch, it has to go in tier 2 — and that is a signal
worth noticing, not just a routing decision.
