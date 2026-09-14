#!/usr/bin/env python3
"""Compare two PPM images and report perceptual + numerical difference.

Used for two different jobs, which need two different tolerances:

  1. Golden-image regression on one backend. A refactor that only reorders
     float operations still changes low bits, so exact equality is not a
     usable criterion -- but with a fixed seed the sampling pattern is
     identical, so the difference should be tiny. Use a strict tolerance.

  2. Cross-backend comparison (CUDA vs. serial). The two renderers have
     completely unrelated RNG streams and the serial branch works in double
     precision, so the images differ by Monte Carlo noise everywhere. They
     should still *converge* to the same picture, so compare at high spp with
     a loose tolerance, and read mean error rather than max.

Exit status is 0 when every requested tolerance holds, 1 otherwise.
"""

import argparse
import math
import sys


def read_ppm(path):
    """Read a binary (P6) or ASCII (P3) PPM. Returns (width, height, bytes)."""
    with open(path, "rb") as f:
        data = f.read()

    # The header is ASCII in both formats: magic, width, height, maxval,
    # separated by whitespace, with '#' comments allowed anywhere in it.
    fields, i = [], 0
    while len(fields) < 4:
        while i < len(data) and data[i : i + 1].isspace():
            i += 1
        if i < len(data) and data[i : i + 1] == b"#":
            while i < len(data) and data[i : i + 1] != b"\n":
                i += 1
            continue
        start = i
        while i < len(data) and not data[i : i + 1].isspace():
            i += 1
        fields.append(data[start:i])
    i += 1  # exactly one whitespace byte terminates the header for P6

    magic = fields[0].decode()
    w, h, maxval = int(fields[1]), int(fields[2]), int(fields[3])
    if maxval != 255:
        raise ValueError(f"{path}: only maxval 255 supported, got {maxval}")

    if magic == "P6":
        px = data[i : i + w * h * 3]
        if len(px) != w * h * 3:
            raise ValueError(f"{path}: truncated, expected {w*h*3} bytes, got {len(px)}")
        return w, h, px
    if magic == "P3":
        vals = data[i:].split()
        if len(vals) < w * h * 3:
            raise ValueError(f"{path}: truncated P3 data")
        return w, h, bytes(int(v) for v in vals[: w * h * 3])
    raise ValueError(f"{path}: unsupported magic '{magic}'")


def compare(a_path, b_path):
    wa, ha, pa = read_ppm(a_path)
    wb, hb, pb = read_ppm(b_path)
    if (wa, ha) != (wb, hb):
        raise ValueError(f"size mismatch: {wa}x{ha} vs {wb}x{hb}")

    n = len(pa)
    total = 0
    sq = 0
    worst = 0
    # Histogram of per-channel absolute differences, for the "how many pixels
    # moved at all" figure that catches small localised regressions that a mean
    # over a large image would bury.
    over_1 = over_4 = over_16 = 0

    for x, y in zip(pa, pb):
        d = x - y if x > y else y - x
        total += d
        sq += d * d
        if d > worst:
            worst = d
        if d > 1:
            over_1 += 1
            if d > 4:
                over_4 += 1
                if d > 16:
                    over_16 += 1

    mae = total / n
    rmse = math.sqrt(sq / n)
    # PSNR against an 8-bit full-scale signal; >40 dB is visually identical,
    # >30 dB is "same image, different noise".
    psnr = float("inf") if sq == 0 else 10 * math.log10((255.0**2) * n / sq)

    return {
        "width": wa, "height": ha, "channels": n,
        "mae": mae, "rmse": rmse, "psnr": psnr, "max_abs": worst,
        "pct_over_1": 100.0 * over_1 / n,
        "pct_over_4": 100.0 * over_4 / n,
        "pct_over_16": 100.0 * over_16 / n,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reference")
    ap.add_argument("candidate")
    ap.add_argument("--max-mae", type=float, default=None,
                    help="fail if mean absolute error exceeds this (0-255 scale)")
    ap.add_argument("--max-abs", type=int, default=None,
                    help="fail if any single channel differs by more than this")
    ap.add_argument("--min-psnr", type=float, default=None,
                    help="fail if PSNR drops below this (dB)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    try:
        r = compare(args.reference, args.candidate)
    except (OSError, ValueError) as e:
        print(f"ppm_compare: {e}", file=sys.stderr)
        return 2

    if not args.quiet:
        print(f"{args.reference} vs {args.candidate}")
        print(f"  size        {r['width']}x{r['height']}")
        print(f"  MAE         {r['mae']:.4f}  (0-255 scale)")
        print(f"  RMSE        {r['rmse']:.4f}")
        print(f"  PSNR        {r['psnr']:.2f} dB" if r["psnr"] != float("inf")
              else "  PSNR        inf (bit-identical)")
        print(f"  max |delta| {r['max_abs']}")
        print(f"  channels differing by >1/>4/>16: "
              f"{r['pct_over_1']:.2f}% / {r['pct_over_4']:.2f}% / {r['pct_over_16']:.2f}%")

    failures = []
    if args.max_mae is not None and r["mae"] > args.max_mae:
        failures.append(f"MAE {r['mae']:.4f} > {args.max_mae}")
    if args.max_abs is not None and r["max_abs"] > args.max_abs:
        failures.append(f"max|delta| {r['max_abs']} > {args.max_abs}")
    if args.min_psnr is not None and r["psnr"] < args.min_psnr:
        failures.append(f"PSNR {r['psnr']:.2f} < {args.min_psnr}")

    if failures:
        print("FAIL: " + "; ".join(failures), file=sys.stderr)
        return 1
    if not args.quiet:
        print("  OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
