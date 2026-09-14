#!/usr/bin/env python3
"""Golden-image regression test.

Renders every scene at a small fixed configuration and compares against a
committed reference image.

Why this works: with a fixed --seed the renderer is bit-for-bit reproducible on
a given GPU and build, so the reference can be compared with a very tight
tolerance. That makes it a real safety net for refactoring -- flattening the
BVH, replacing virtual dispatch, moving scene construction to the host -- none
of which is allowed to change the picture.

What it will NOT survive, by design:
  * changing the RNG, the sample ordering, or the number of random draws per
    bounce (the sampling pattern moves, so every pixel moves)
  * a different GPU architecture, if the compiler makes different fusion or
    fast-math choices
  * switching float <-> double

Those are all real changes to the output, so re-baselining is the correct
response -- but it must be a deliberate `--update` with the diff eyeballed,
never an automatic refresh.

Usage:
    python tests/run_golden.py --exe build/bin/Release/rayTracer.exe
    python tests/run_golden.py --exe ... --update      # re-baseline
    python tests/run_golden.py --exe ... --scene cornell
"""

import argparse
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import ppm_compare  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN_DIR = os.path.join(HERE, "golden")

# Aspect ratios mirror each scene's default resolution in src/main.cu. The
# renders are deliberately tiny and low-spp: this suite is checking that the
# image did not *change*, not that it converged.
SCENES = {
    #  name              aspect (w, h)
    "bouncing":        (2, 1),
    "checkered":       (2, 1),
    "earth":           (2, 1),
    "perlin":          (2, 1),
    "quads":           (2, 1),
    "simple_light":    (2, 1),
    "cornell":         (1, 1),
    "cornell_smoke":   (1, 1),
    "final":           (1, 1),
    "original":        (1, 1),
}

WIDTH = 128
SPP = 24
MAX_DEPTH = 12
SEED = 1984

# A same-seed render on the same binary is bit-identical, so in principle this
# could be 0. A hair of slack absorbs driver-level nondeterminism (e.g. a
# different SM scheduling order interacting with denormal flushing) without
# being loose enough to hide a real regression: MAE 0.5/255 is invisible, and
# any structural change moves MAE by whole units.
MAX_MAE = 0.5
MAX_ABS = 12


def render(exe, scene, width, height, out, extra=()):
    cmd = [exe, "--scene", scene, "--width", str(width), "--height", str(height),
           "--spp", str(SPP), "--max-depth", str(MAX_DEPTH), "--seed", str(SEED),
           "--quiet", "--out", out, *extra]
    r = subprocess.run(cmd, capture_output=True, text=True,
                       cwd=os.path.dirname(os.path.abspath(exe)))
    if r.returncode != 0:
        raise RuntimeError(f"render failed ({r.returncode}):\n{r.stdout}\n{r.stderr}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True, help="path to rayTracer executable")
    ap.add_argument("--update", action="store_true", help="rewrite the reference images")
    ap.add_argument("--scene", action="append", help="limit to these scenes (repeatable)")
    args = ap.parse_args()

    exe = os.path.abspath(args.exe)
    if not os.path.exists(exe):
        print(f"error: no such executable: {exe}", file=sys.stderr)
        return 2

    os.makedirs(GOLDEN_DIR, exist_ok=True)
    scenes = args.scene if args.scene else list(SCENES)

    # Textures resolve relative to the executable's directory, so renders run
    # from there; the output path must therefore be absolute.
    tmpdir = tempfile.mkdtemp(prefix="rt_golden_")
    failures, missing, updated = [], [], []

    for scene in scenes:
        if scene not in SCENES:
            print(f"error: unknown scene '{scene}'", file=sys.stderr)
            return 2
        aw, ah = SCENES[scene]
        width, height = WIDTH, WIDTH * ah // aw
        ref = os.path.join(GOLDEN_DIR, f"{scene}.ppm")

        if args.update:
            render(exe, scene, width, height, ref)
            print(f"  updated  {scene:<16} {width}x{height}")
            updated.append(scene)
            continue

        if not os.path.exists(ref):
            print(f"  MISSING  {scene:<16} (no reference; run with --update)")
            missing.append(scene)
            continue

        cand = os.path.join(tmpdir, f"{scene}.ppm")
        render(exe, scene, width, height, cand)
        try:
            r = ppm_compare.compare(ref, cand)
        except ValueError as e:
            print(f"  FAIL     {scene:<16} {e}")
            failures.append(scene)
            continue

        bad = r["mae"] > MAX_MAE or r["max_abs"] > MAX_ABS
        status = "FAIL    " if bad else "ok      "
        print(f"  {status} {scene:<16} MAE {r['mae']:7.4f}  max {r['max_abs']:3d}  "
              f"PSNR {r['psnr']:6.2f} dB" if r["psnr"] != float("inf") else
              f"  {status} {scene:<16} identical")
        if bad:
            failures.append(scene)

    print()
    if args.update:
        print(f"re-baselined {len(updated)} scene(s) in {GOLDEN_DIR}")
        print("review the images before committing -- --update accepts whatever it renders")
        return 0
    if missing:
        print(f"{len(missing)} scene(s) have no reference image: {', '.join(missing)}")
    if failures:
        print(f"FAIL: {len(failures)} scene(s) regressed: {', '.join(failures)}")
        return 1
    print(f"PASS: {len(scenes) - len(missing)} scene(s) match their reference")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
