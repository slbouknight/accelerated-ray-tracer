# Benchmarking

```bat
python bench\run_bench.py --config bench\targets.json --csv results.csv
python bench\run_bench.py --exe build\bin\Release\rayTracer.exe --quick
```

The driver talks to renderers through the **command line**, not through shared
source. Any binary that accepts

```
--scene NAME --width N --height N --spp N --seed N --quiet --out FILE [--stats FILE]
```

can be measured, which is what lets the same script compare the CUDA branch,
the serial branch, and two revisions of either against each other.

---

## What is actually being measured

| source | what it includes |
|---|---|
| `--stats` `render_ms` | GPU time between the first accumulate launch and the resolve, via `cudaEvent`. **Preferred.** |
| `--stats` `build_ms` | Scene construction + BVH build. Reported separately because it is single-threaded and scales badly. |
| process wall time | Everything: process start, CUDA context creation (~200 ms), texture upload, render, PPM write. Used only when a target has no `--stats`. |

`min` over the timed runs is the headline number. Contention only ever makes a
run slower, so the fastest observed run is closest to the machine's real
capability; a large gap between `min` and `median` means something else was
using the GPU. One warmup run is always discarded — the first launch of a CUDA
binary pays context creation, and PTX JIT if the binary was not compiled for
the installed architecture.

### Before trusting a number

- **Build Release.** A Debug CUDA build compiles device code with `-G`, which
  disables essentially all device-side optimisation.
- **Check the architecture.** `CMAKE_CUDA_ARCHITECTURES` now defaults to
  `native`. If it is set to something older, every kernel is JIT-compiled from
  PTX at startup and may generate worse code.
- **Pin the clocks** if you want numbers comparable across days:
  `nvidia-smi -lgc <min>,<max>` (needs admin), then check
  `nvidia-smi -q -d PERFORMANCE` for throttle reasons afterwards.
- **Close anything else using the GPU**, including the browser.

---

## Comparing against the serial branch

Check out both branches side by side so they can be measured in one run:

```bat
git worktree add ..\rt-serial serial
cmake -S ..\rt-serial -B ..\rt-serial\build
cmake --build ..\rt-serial\build --config Release
```

Then add a target to `bench/targets.json`.

### Style `legacy` — works today, coarse

The serial `main()` takes no arguments and prints a PPM to stdout, so the
driver can only time the whole process, and the scene, resolution and sample
count are whatever it was compiled with. The matrix entries are ignored. Useful
for a rough order-of-magnitude figure; not for a like-for-like comparison.

### Style `cli` — worth the 30 minutes

Three changes to the serial branch make it directly comparable:

**1. Accept the flags.** `camera` already exposes everything needed as public
fields, so after each scene function builds its `camera cam`, override:

```cpp
cam.image_width       = opt.width;
cam.aspect_ratio      = double(opt.width) / opt.height;   // camera derives height from this
cam.samples_per_pixel = opt.spp;
cam.max_depth         = opt.max_depth;
```

Replace the `switch (9)` in `main()` with a lookup on `--scene`, using the same
names as `rayTracer.exe --list` so one matrix drives both branches.

**2. Make it deterministic.** `rtweekend.h` currently seeds from
`std::random_device`, so no two runs match and golden images are impossible:

```cpp
inline double random_double() {
    static std::uniform_real_distribution<double> distribution(0.0, 1.0);
    static std::mt19937 generator(std::random_device{}());   // <-- non-deterministic
    return distribution(generator);
}
```

Take the seed from the command line instead. Note this also means the serial
`bouncing_spheres` scene currently generates *different geometry every run*,
which alone makes its timings non-comparable.

**3. Emit `--stats`.** Wrap `cam.render(world)` in `std::chrono::steady_clock`
and write the same JSON keys (`render_ms`, `build_ms`, `scene`, `width`,
`height`, `spp`). Then the driver compares GPU render time against CPU render
time with process startup excluded on both sides.

---

## Reading a CUDA-vs-CPU speedup honestly

The two renderers are not doing identical work, and the difference matters more
than most speedup numbers admit:

- **Precision.** Serial is `double` throughout; CUDA is mostly `float`. On a
  GeForce card FP64 runs at 1/64 the FP32 rate, so some of the "speedup" is a
  precision change, not parallelism. Say so, or build a float serial variant.
- **Sampling.** Different RNGs and different scatter routines mean the images
  converge to the same result but never match sample-for-sample. Compare at
  high spp with `tools/ppm_compare.py --min-psnr 30`, not with a tight
  tolerance.
- **Scene contents.** The CUDA `bouncing` and `final` scenes have diverged from
  their serial counterparts (different palettes, emissive spheres, a 20×20 box
  grid with a deterministic height function). Ray counts differ, so
  paths/second is the comparable metric, not wall time.
- **Thread count.** The serial branch is single-threaded. A fair "what did CUDA
  buy me" figure compares against an OpenMP'd CPU version too — otherwise a
  good chunk of the speedup is just "1 core versus a GPU".

`Mpaths/s` (`width × height × spp / render_time`) is the most portable metric
here. It is still not rays/second — each path traces up to `--max-depth`
bounces and terminates early on a light or an absorbed ray — but it is
identical work on both sides for a given scene and depth.
