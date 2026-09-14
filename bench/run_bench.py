#!/usr/bin/env python3
"""Benchmark driver for the ray tracer.

Runs a scene matrix against one or more binaries and reports timings, so you
can compare CUDA against serial, before against after a refactor, or one block
size against another.

The seam between this script and the renderers is the *command line*, not the
source: any binary that accepts --scene/--width/--height/--spp/--seed/--out and
optionally writes a --stats JSON can be benchmarked. That is what lets the same
driver measure the CUDA branch and the serial branch without either knowing
about the other.

    python bench/run_bench.py --config bench/targets.json
    python bench/run_bench.py --exe build/bin/Release/rayTracer.exe --quick
    python bench/run_bench.py --config bench/targets.json --csv results.csv

Methodology notes, because they change the numbers more than most code does:

  * The first run of any CUDA binary pays context creation (~200 ms) and, if
    the binary was not compiled for the installed architecture, PTX JIT of every
    kernel (can be seconds). One warmup run per target is discarded.
  * `min` is reported alongside `median`. For throughput benchmarking the
    minimum is usually the better estimator: contention only ever makes a run
    slower, so the fastest observed run is the closest to the machine's actual
    capability. A large min/median gap means the machine was busy.
  * Render time is taken from the --stats JSON (measured with cudaEvent on the
    GPU) when available, and falls back to process wall time otherwise. Wall
    time includes process startup, scene build and PPM writing; on short renders
    that is most of the measurement.
  * GPU clocks throttle. For numbers you can compare across days, pin them:
        nvidia-smi -lgc <min>,<max>      (needs admin)
    and check `nvidia-smi -q -d PERFORMANCE` for throttle reasons afterwards.
"""

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

DEFAULT_MATRIX = [
    # (scene, width, height, spp) -- moderate sizes so a full sweep is minutes,
    # not hours. Override with --matrix or in the config file.
    ("cornell",       400, 400, 200),
    ("bouncing",      400, 200, 200),
    ("final",         400, 400, 100),
    ("quads",         400, 200, 200),
]

QUICK_MATRIX = [
    ("cornell",       200, 200, 64),
    ("bouncing",      200, 100, 64),
]


class Target:
    """One binary under test."""

    def __init__(self, spec):
        self.name = spec["name"]
        self.exe = os.path.abspath(spec["exe"])
        # "cli"    -> accepts the full flag set and writes --stats JSON
        # "legacy" -> takes no arguments; we can only time the whole process,
        #             and the scene/size/spp are whatever it was compiled with
        self.style = spec.get("style", "cli")
        self.extra = spec.get("extra_args", [])
        self.label = spec.get("label", self.name)

    def available(self):
        return os.path.exists(self.exe)


def run_once(target, scene, width, height, spp, seed, tmpdir):
    """Run one render. Returns (wall_s, stats_dict_or_None)."""
    out = os.path.join(tmpdir, "out.ppm")
    stats_path = os.path.join(tmpdir, "stats.json")
    for stale in (out, stats_path):
        if os.path.exists(stale):
            os.remove(stale)

    if target.style == "cli":
        cmd = [target.exe, "--scene", scene, "--width", str(width),
               "--height", str(height), "--spp", str(spp), "--seed", str(seed),
               "--quiet", "--out", out, "--stats", stats_path, *target.extra]
        stdout = subprocess.DEVNULL
    else:
        # Legacy binary: no flags, PPM straight to stdout.
        cmd = [target.exe, *target.extra]
        stdout = open(out, "wb")

    t0 = time.perf_counter()
    r = subprocess.run(cmd, stdout=stdout, stderr=subprocess.PIPE,
                       cwd=os.path.dirname(target.exe))
    wall = time.perf_counter() - t0
    if stdout is not subprocess.DEVNULL:
        stdout.close()

    if r.returncode != 0:
        raise RuntimeError(f"{target.name} failed ({r.returncode}): "
                           f"{r.stderr.decode(errors='replace')[-400:]}")

    stats = None
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
    return wall, stats


def bench_cell(target, scene, width, height, spp, seed, reps, tmpdir):
    # Warmup, discarded: CUDA context creation, PTX JIT, filesystem cache.
    run_once(target, scene, width, height, spp, seed, tmpdir)

    walls, renders, builds = [], [], []
    device = None
    for _ in range(reps):
        wall, stats = run_once(target, scene, width, height, spp, seed, tmpdir)
        walls.append(wall)
        if stats:
            renders.append(stats["render_ms"] / 1000.0)
            builds.append(stats["build_ms"])
            device = stats.get("device")

    # Prefer the GPU-side measurement; fall back to wall time.
    times = renders if renders else walls
    return {
        "target": target.label,
        "scene": scene, "width": width, "height": height, "spp": spp,
        "device": device,
        "measured": "render_ms" if renders else "wall",
        "min_s": min(times),
        "median_s": statistics.median(times),
        "max_s": max(times),
        "wall_median_s": statistics.median(walls),
        "build_ms": statistics.median(builds) if builds else None,
        "mpaths_s": (width * height * spp) / min(times) / 1e6,
    }


def print_table(rows, baseline_label):
    by_key = {}
    for r in rows:
        by_key.setdefault((r["scene"], r["width"], r["height"], r["spp"]), []).append(r)

    name_w = max(len(r["target"]) for r in rows)
    header = (f"{'scene':<16} {'resolution':>11} {'spp':>6}  {'target':<{name_w}} "
              f"{'min s':>9} {'median s':>9} {'Mpaths/s':>10} {'build ms':>9} {'speedup':>8}")
    print(header)
    print("-" * len(header))

    for key, group in by_key.items():
        scene, w, h, spp = key
        base = next((g for g in group if g["target"] == baseline_label), None)
        for i, r in enumerate(group):
            if base is r:
                speedup = "1.00x"
            elif base is not None and r["min_s"] > 0:
                speedup = f"{base['min_s'] / r['min_s']:.2f}x"
            else:
                speedup = ""

            build = "-" if r["build_ms"] is None else f"{r['build_ms']:.1f}"
            # Only the first row of each group repeats the scene columns.
            scene_col = scene if i == 0 else ""
            res_col = f"{w}x{h}" if i == 0 else ""
            spp_col = str(spp) if i == 0 else ""

            print(f"{scene_col:<16} {res_col:>11} {spp_col:>6}  "
                  f"{r['target']:<{name_w}} "
                  f"{r['min_s']:>9.3f} {r['median_s']:>9.3f} "
                  f"{r['mpaths_s']:>10.2f} {build:>9} {speedup:>8}")
        print()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="JSON file describing the targets")
    ap.add_argument("--exe", action="append", default=[],
                    help="benchmark this executable (repeatable); shorthand for --config")
    ap.add_argument("--reps", type=int, default=3, help="timed runs per cell (default 3)")
    ap.add_argument("--seed", type=int, default=1984)
    ap.add_argument("--quick", action="store_true", help="use the small scene matrix")
    ap.add_argument("--scene", action="append", help="limit to these scenes")
    ap.add_argument("--csv", help="also write raw results here")
    ap.add_argument("--json", help="also write raw results here as JSON")
    ap.add_argument("--baseline", help="target label to compute speedups against")
    args = ap.parse_args()

    specs = []
    matrix = QUICK_MATRIX if args.quick else DEFAULT_MATRIX
    if args.config:
        with open(args.config) as f:
            cfg = json.load(f)
        specs += cfg.get("targets", [])
        if "matrix" in cfg and not args.quick:
            matrix = [tuple(m) for m in cfg["matrix"]]
        if args.baseline is None:
            args.baseline = cfg.get("baseline")
    for e in args.exe:
        specs.append({"name": os.path.basename(e), "exe": e})

    if not specs:
        print("error: give --config or at least one --exe", file=sys.stderr)
        return 2

    targets = [Target(s) for s in specs]
    for t in targets:
        if not t.available():
            print(f"error: no such executable: {t.exe}", file=sys.stderr)
            return 2

    if args.scene:
        matrix = [m for m in matrix if m[0] in args.scene]
        if not matrix:
            print("error: --scene filtered out every matrix entry", file=sys.stderr)
            return 2

    baseline = args.baseline or targets[0].label
    tmpdir = tempfile.mkdtemp(prefix="rt_bench_")
    rows = []

    total = len(matrix) * len(targets)
    n = 0
    try:
        for scene, w, h, spp in matrix:
            for t in targets:
                n += 1
                print(f"[{n}/{total}] {t.label:<20} {scene} {w}x{h} {spp}spp ...",
                      end="", flush=True, file=sys.stderr)
                try:
                    row = bench_cell(t, scene, w, h, spp, args.seed, args.reps, tmpdir)
                except RuntimeError as e:
                    print(f" FAILED\n  {e}", file=sys.stderr)
                    continue
                rows.append(row)
                print(f" {row['min_s']:.3f}s", file=sys.stderr)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    if not rows:
        print("no results", file=sys.stderr)
        return 1

    device = next((r["device"] for r in rows if r["device"]), None)
    print()
    if device:
        print(f"device: {device}")
    measured = {r["measured"] for r in rows}
    print(f"timing source: {', '.join(sorted(measured))}   reps: {args.reps} (min of, after 1 warmup)")
    print()
    print_table(rows, baseline)

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            wtr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wtr.writeheader()
            wtr.writerows(rows)
        print(f"wrote {args.csv}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
