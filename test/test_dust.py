#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Checks the VapourSynth interface of Dust bit for bit against the numpy model (model.py); runs in CI too.

Same case matrix as test_avs.py (cases.py; the AviSynth-only YUY2/YUVA variants are skipped), once with the
automatic instruction set and once for every available opt value, plus VapourSynth
specific checks: planes as an index list, frame properties kept, results independent of the request order, and the
parameter errors.

Usage: python test/test_dust.py [path/to/Dust plugin] [--only SUBSTRING] [-v]
Without a plugin path the plugin must be autoloaded (installed wheel). Needs numpy.
Windows with Hybrid: F:\\Hybrid\\64bit\\Vapoursynth\\python.exe test\\test_dust.py build\\Dust.dll
"""
import argparse
import os
import sys

import numpy as np
import vapoursynth as vs

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cases  # noqa: E402
import model  # noqa: E402

core = vs.core
VS_FORMATS = {
    "420p8": vs.YUV420P8, "422p8": vs.YUV422P8, "444p8": vs.YUV444P8, "y8": vs.GRAY8,
    "420p10": vs.YUV420P10, "444p12": vs.YUV444P12, "420p16": vs.YUV420P16, "y16": vs.GRAY16,
}


def to_vs(frames, fmt):
    h, w = frames[0].planes[0].shape
    blank = core.std.BlankClip(width=w, height=h, format=VS_FORMATS[fmt], length=len(frames))

    def fill(n, f):
        fo = f.copy()
        for p, arr in enumerate(frames[n].planes):
            np.asarray(fo[p])[...] = arr
        for k, v in frames[n].props.items():
            fo.props[k] = v
        return fo

    return core.std.ModifyFrame(blank, blank, fill)


def from_vs(clip, order=None):
    out = [None] * clip.num_frames
    for n in order or range(clip.num_frames):
        f = clip.get_frame(n)
        out[n] = model.Frame([np.asarray(f[p]).astype(np.int64) for p in range(f.format.num_planes)], dict(f.props))
    return out


def vs_args(kwargs):
    a = {}
    for k, v in kwargs.items():
        if k == "planes":
            a[k] = [i for i, ch in enumerate("yuv") if ch in v]
        elif isinstance(v, bool):
            a[k] = int(v)
        else:
            a[k] = v
    return a


class Runner:
    def __init__(self, args):
        self.args = args
        self.opt = None
        self.count = 0
        self.failures = 0

    def ok(self, cond, name, detail=""):
        self.count += 1
        if not cond:
            self.failures += 1
            print(f"FAIL {name}: {detail}")
        elif self.args.verbose:
            print(f"ok   {name}")

    def check(self, name, fmt, width, height, func, kwargs, props=None, mc=0, avs=None):
        if avs or (self.args.only and self.args.only not in name):
            return
        frames = cases.make_clip(fmt, width, height)
        for n, pr in (props or {}).items():
            frames[n].props.update(pr)
        src = to_vs(frames, fmt)
        a = vs_args(kwargs)
        if self.opt is not None:
            a["opt"] = self.opt
        if mc:
            a["prev"] = [to_vs(cases.transform(frames, cases.MC_PREV[i]), fmt) for i in range(mc)]
            a["next"] = [to_vs(cases.transform(frames, cases.MC_NEXT[i]), fmt) for i in range(mc)]
        try:
            got = from_vs(getattr(core.dust, func)(src, **a))
        except vs.Error as e:
            return self.ok(False, f"{name} opt={self.opt}", str(e).strip().splitlines()[-1])
        exp = cases.expected(frames, fmt, func, kwargs, mc)
        for n, (g, e) in enumerate(zip(got, exp)):
            for p, (gp, ep) in enumerate(zip(g.planes, e.planes)):
                if not np.array_equal(gp, ep):
                    d = np.argwhere(gp != ep)
                    y, x = d[0]
                    return self.ok(False, f"{name} opt={self.opt}", f"frame {n} plane {p}: {len(d)} samples differ, first at "
                                                f"({x},{y}): {gp[y, x]} vs {ep[y, x]}")
        self.ok(True, name)

    def expect_error(self, name, fn, text):
        if self.args.only and self.args.only not in name:
            return
        try:
            fn()
        except vs.Error as e:
            return self.ok(text in str(e), name, f"wrong error: {str(e).strip()}")
        self.ok(False, name, f"no error, expected {text!r}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("plugin", nargs="?")
    ap.add_argument("--only", default="")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    if args.plugin:
        core.std.LoadPlugin(os.path.abspath(args.plugin))
    if not hasattr(core, "dust"):
        print("FAIL: plugin dust not loaded")
        return 1
    print(f"{core.dust.name} {core.dust.version} ({core.dust.identifier}), VapourSynth {core.core_version}")
    r = Runner(args)
    blank = core.std.BlankClip(format=vs.GRAY8, width=8, height=8, length=1)
    opts = []
    for o in (0, 1, 2, 3, 4):
        try:
            core.dust.SpaceDust(blank, opt=o).get_frame(0)
            opts.append(o)
        except vs.Error:
            pass
    print("instruction sets (opt):", opts)
    for r.opt in [None] + opts:
        for name, fmt, w, h, func, kwargs, extras in cases.cases():
            r.check(name, fmt, w, h, func, kwargs, **extras)
    r.opt = None

    if not args.only or "vs-" in args.only:
        frames = cases.make_clip("420p8", 40, 24)
        frames[2].props["Custom"] = 7
        src = to_vs(frames, "420p8")
        out = core.dust.PixieDust(src, tr=2)
        fwd = from_vs(out)
        bwd = from_vs(out, order=range(src.num_frames - 1, -1, -1))
        r.ok(all(np.array_equal(a, b) for f, g in zip(fwd, bwd) for a, b in zip(f.planes, g.planes)),
             "vs-request-order", "backward request order gives other frames")
        r.ok(fwd[2].props.get("Custom") == 7, "vs-props", f"props {fwd[2].props}")
        same = from_vs(core.dust.PixieDust(src, tr=2, planes=[2, 0, 1]))
        r.ok(all(np.array_equal(a, b) for f, g in zip(fwd, same) for a, b in zip(f.planes, g.planes)),
             "vs-planes-all", "planes=[2, 0, 1] differs from the default")
        u = from_vs(core.dust.PixieDust(src, planes=[1]))
        r.ok(all(np.array_equal(f.planes[0], g.planes[0]) and np.array_equal(f.planes[2], g.planes[2])
                 for f, g in zip(u, frames)), "vs-planes-copy", "unselected planes changed")

        r.expect_error("vs-err-plane-range", lambda: core.dust.PixieDust(src, planes=[3]), "out of range")
        r.expect_error("vs-err-plane-twice", lambda: core.dust.PixieDust(src, planes=[0, 0]), "specified twice")
        r.expect_error("vs-err-gray-plane", lambda: core.dust.PixieDust(src.std.ShufflePlanes(0, vs.GRAY),
                                                                        planes=[1]), "out of range")
        r.expect_error("vs-err-rgb", lambda: core.dust.PixieDust(src.resize.Point(format=vs.RGB24)),
                       "unsupported format")
        r.expect_error("vs-err-float", lambda: core.dust.PixieDust(src.resize.Point(format=vs.YUV420PS)), "float")
        r.expect_error("vs-err-limit", lambda: core.dust.PixieDust(src, limit=65), "limit must be 1..64")
        r.expect_error("vs-err-tr", lambda: core.dust.GoldDust(src, tr=0), "tr must be 1..5")
        r.expect_error("vs-err-limit-edge", lambda: core.dust.PixieDust(src, limit=4, limit_edge=5), "<= limit")
        r.expect_error("vs-err-prev", lambda: core.dust.PixieDust(src, tr=2, prev=[src]), "prev array size")
        r.expect_error("vs-err-mc-format", lambda: core.dust.PixieDust(src, prev=[src.resize.Point(
            format=vs.YUV420P10)]), "format mismatch")
        r.expect_error("vs-err-mc-length", lambda: core.dust.PixieDust(src, next=[src[:5]]), "shorter")
        r.expect_error("vs-err-scthresh", lambda: core.dust.PixieDust(src, scthresh=256), "scthresh must be")
        r.expect_error("vs-err-opt", lambda: core.dust.PixieDust(src, opt=9), "instruction set not available")
        try:
            core.dust.SpaceDust(src, tr=2)
            r.ok(False, "vs-space-no-tr", "SpaceDust accepted tr")
        except vs.Error:
            r.ok(True, "vs-space-no-tr")

    print(f"{r.count - r.failures} of {r.count} checks passed")
    return 1 if r.failures else 0


if __name__ == "__main__":
    sys.exit(main())
