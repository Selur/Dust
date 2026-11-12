#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Speed of every instruction set tier (opt) for typical calls, 1920x1080 4:2:0, 8 and 16 bit, VapourSynth.

Source frames are random noise prepared once and looped, so decoding costs nothing. One thread by default
(--threads), to compare the kernels themselves. Prints frames per second per call and tier.

Usage: python test/bench.py [path/to/Dust plugin] [--frames 200] [--threads 1]
"""
import argparse
import os
import time

import numpy as np
import vapoursynth as vs

core = vs.core
CALLS = [
    ("PixieDust", {}),
    ("PixieDust", {"tr": 2, "temporal": 1}),
    ("FaeryDust", {"tr": 5, "temporal": 1}),
    ("GoldDust", {"tr": 3}),
    ("SpaceDust", {}),
    ("PixieDust", {"limit": 16, "limit_edge": 3, "edge_thresh": 24}),
]


def noise_clip(fmt, frames=12):
    fmt_info = core.get_video_format(fmt)
    rng = np.random.default_rng(1)
    blank = core.std.BlankClip(width=1920, height=1080, format=fmt, length=frames)
    maxv = (1 << fmt_info.bits_per_sample) - 1

    def fill(n, f):
        fo = f.copy()
        for p in range(fmt_info.num_planes):
            a = np.asarray(fo[p])
            a[...] = np.clip(rng.normal(maxv * 0.5, maxv * 0.05, a.shape), 0, maxv)
        return fo

    return core.std.ModifyFrame(blank, blank, fill)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("plugin", nargs="?")
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    if a.plugin:
        core.std.LoadPlugin(os.path.abspath(a.plugin))
    core.num_threads = a.threads
    opts = []
    probe = core.std.BlankClip(format=vs.GRAY8, width=8, height=8, length=1)
    for o in (0, 1, 2, 3, 4):
        try:
            core.dust.SpaceDust(probe, opt=o).get_frame(0)
            opts.append(o)
        except vs.Error:
            pass
    names = {0: "C", 1: "SSE4.1", 2: "AVX", 3: "AVX2", 4: "NEON"}
    print(f"{'call':52}" + "".join(f"{names[o]:>9}" for o in opts) + "   fps, 1920x1080, %d thread(s)" % a.threads)
    for fmt in (vs.YUV420P8, vs.YUV420P16):
        # Materialize the noise frames once so that only the filter is timed.
        src = noise_clip(fmt)
        frames = [src.get_frame(i) for i in range(src.num_frames)]
        held = core.std.BlankClip(src, length=src.num_frames)
        held = core.std.ModifyFrame(held, held, lambda n, f: frames[n].copy())
        loop = core.std.Loop(held, times=a.frames // held.num_frames + 1)[:a.frames]
        base = time.perf_counter()
        for _ in loop.frames():
            pass
        overhead = time.perf_counter() - base
        for func, kw in CALLS:
            row = f"{core.get_video_format(fmt).name + ' ' + func + str(kw):52}"
            for o in opts:
                clip = getattr(core.dust, func)(loop, opt=o, **kw)
                t = time.perf_counter()
                for _ in clip.frames():
                    pass
                dt = max(time.perf_counter() - t - overhead, 1e-9)
                row += f"{a.frames / dt:9.1f}"
            print(row, flush=True)


if __name__ == "__main__":
    main()
