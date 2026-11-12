#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compares Dust with the DustPort 1.0.0 DLL (AviSynth+, Windows) and reports how much the output differs.

Dust deliberately computes other formulas than DustPort 1.0 (see README, differences), so this is a report, not an
equality test. Checked hard: both outputs stay within the limit of the source, so they differ by at most 2 * limit.
Dust runs with scthresh=-1 (DustPort has no scene change handling). Get the old DLL with
    git show e2128ac:DustPort_build/DustPort.dll > test/work/DustPort-1.0.0.dll

Usage: F:\\Python4Claude\\python.exe test/compare_original.py [--original test/work/DustPort-1.0.0.dll]
       [--plugin build/Dust.dll] [CLIP.y4m]   (default: the synthetic 4:2:0 clip of test_avs.py)
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_avs  # noqa: E402

HERE = test_avs.HERE
CASES = [  # (function, arguments)
    ("FaeryDust", ""), ("PixieDust", ""), ("GoldDust", ""), ("SpaceDust", ""),
    ("PixieDust", "tr=2"), ("PixieDust", "tr=2, temporal=1"), ("PixieDust", "tr=4, temporal=1"),
    ("PixieDust", "tr=5, temporal=1"), ("FaeryDust", "tr=3, temporal=1"),
    ("PixieDust", "limit=16, limit_edge=3, edge_thresh=24"), ("SpaceDust", "limit=6, planes=\"y\""),
]
FORMATS = [("YV12", "", 8), ("YUV420P16", ".ConvertBits(16)", 16), ("YUY2", ".ConvertToYUY2()", 8)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--original", default=os.path.join(HERE, "work", "DustPort-1.0.0.dll"))
    ap.add_argument("--plugin", default=os.path.join(test_avs.ROOT, "build", "Dust.dll"))
    ap.add_argument("--avs2yuv", default=os.path.join(test_avs.AVS_DIR, "avs2yuv64.exe"))
    ap.add_argument("--bestsource", default=os.path.join(test_avs.AVS_DIR, "avisynthPlugins", "BestSource.dll"))
    ap.add_argument("clip", nargs="?")
    args = ap.parse_args()
    args.only, args.verbose = "", False
    r = test_avs.Runner(args)
    src = args.clip or r.clip("420p8", 720, 480)[0]

    def run(name, plugin, call, fmt_conv, back):
        script = (f'LoadPlugin("{test_avs.fwd(args.bestsource)}")\nLoadPlugin("{test_avs.fwd(plugin)}")\n'
                  f'BSVideoSource("{test_avs.fwd(src)}", cachemode=0){fmt_conv}.{call}{back}\n')
        rc, out, err = r.run(name, script)
        if rc != 0:
            raise SystemExit(f"{name}: {err.strip()[-300:]}")
        return out

    bad = 0
    print(f"{'format':10} {'call':48} {'changed':>8} {'max':>5} {'mean':>6}")
    for fmt, conv, bits in FORMATS:
        back = ".ConvertToYV16()" if fmt == "YUY2" else ""
        dt = "<u2" if bits > 8 else "u1"
        for func, a in CASES:
            if fmt == "YUY2" and ("planes" in a or "limit_edge" in a):
                continue
            limit = int(a.split("limit=")[1].split(",")[0]) if "limit=" in a else {"FaeryDust": 2, "PixieDust": 5,
                                                                                    "GoldDust": 8, "SpaceDust": 6}[func]
            sep = ", " if a else ""
            old = run("orig", args.original, f"{func}Port({a})", conv, back)
            sc = "" if func == "SpaceDust" else sep + "scthresh=-1"
            new = run("new", args.plugin, f"{func}({a}{sc})", conv, back)
            x = np.frombuffer(old, dt).astype(np.int64)
            y = np.frombuffer(new, dt).astype(np.int64)
            d = np.abs(x - y)
            share = np.count_nonzero(d) / d.size
            mx = int(d.max()) >> (bits - 8)
            ok = d.max() <= 2 * (limit << (bits - 8))
            bad += not ok
            print(f"{fmt:10} {func + '(' + a + ')':48} {share:8.1%} {mx:5} {d.mean() / (1 << (bits - 8)):6.3f}"
                  + ("" if ok else "  OUT OF BOUND"))
    print("max and mean on the 8-bit scale")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
