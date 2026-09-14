#!/usr/bin/env python3
"""Convert PPM (P3 or P6) to PNG using only the standard library.

The renderer emits PPM because that is what the book does and it needs no
encoder. Viewing it normally means installing ImageMagick; this is a 60-line
substitute with no dependencies, which also makes it usable from CI.

    python tools/ppm_to_png.py render.ppm            -> render.png
    python tools/ppm_to_png.py a.ppm b.ppm c.ppm     -> a.png b.png c.png
    python tools/ppm_to_png.py in.ppm -o out.png
    python tools/ppm_to_png.py in.ppm --scale 4      -> nearest-neighbour zoom
"""

import argparse
import os
import struct
import sys
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ppm_compare import read_ppm  # noqa: E402


def write_png(path, width, height, rgb):
    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    # PNG scanlines are each prefixed with a filter-type byte; 0 means "none",
    # which costs a little compression ratio and saves all the filter logic.
    raw = bytearray()
    stride = width * 3
    for y in range(height):
        raw.append(0)
        raw += rgb[y * stride:(y + 1) * stride]

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit truecolour
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", ihdr))
        f.write(chunk(b"IDAT", zlib.compress(bytes(raw), 9)))
        f.write(chunk(b"IEND", b""))


def upscale(width, height, rgb, factor):
    """Nearest-neighbour, so the pixels of a small test render stay legible."""
    out = bytearray()
    stride = width * 3
    for y in range(height):
        row = bytearray()
        for x in range(width):
            row += rgb[y * stride + x * 3: y * stride + x * 3 + 3] * factor
        out += row * factor
    return width * factor, height * factor, bytes(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("-o", "--output", help="output path (only with a single input)")
    ap.add_argument("--scale", type=int, default=1, help="integer upscale factor")
    args = ap.parse_args()

    if args.output and len(args.inputs) > 1:
        print("error: -o only makes sense with one input", file=sys.stderr)
        return 2

    for src in args.inputs:
        try:
            w, h, px = read_ppm(src)
        except (OSError, ValueError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        if args.scale > 1:
            w, h, px = upscale(w, h, px, args.scale)
        dst = args.output or os.path.splitext(src)[0] + ".png"
        write_png(dst, w, h, px)
        print(f"{src} -> {dst}  ({w}x{h})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
