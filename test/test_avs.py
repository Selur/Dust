#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Checks the AviSynth+ interface of Dust bit for bit against the numpy model (model.py) on synthetic clips.

The clips (cases.py) are written as y4m and read with BestSource; the output comes from avs2yuv64 -raw. Covers every
function, mode, temporal setting, tr 1..5, edge-aware limits, planes, scene changes (detector and frame properties),
clip ends, borders (odd and tiny sizes), field mode (parameter and _FieldBased), motion compensated prev/next,
8/10/12/16 bit, Y, 4:2:0/4:2:2/4:4:4, YUVA and YUY2, plus the parameter errors.

Usage (Windows): F:\\Python4Claude\\python.exe test/test_avs.py [--plugin build/Dust.dll] [--only SUBSTRING]
"""
import argparse
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cases  # noqa: E402
import model  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
AVS_DIR = r"F:\Hybrid\64bit\Avisynth"


def fwd(p):
    return os.path.abspath(p).replace("\\", "/")


def write_y4m(path, frames, fmt, width, height):
    tag, bits, _ = cases.FORMATS[fmt]
    with open(path, "wb") as f:
        f.write(f"YUV4MPEG2 W{width} H{height} F25:1 Ip A1:1 C{tag}\n".encode())
        for fr in frames:
            f.write(b"FRAME\n")
            for p in fr.planes:
                f.write(p.astype("<u2" if bits > 8 else "u1").tobytes())


def serialize(frames, bits):
    return b"".join(p.astype("<u2" if bits > 8 else "u1").tobytes() for fr in frames for p in fr.planes)


def avs_value(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return f'"{v}"'
    return str(v)


def describe(out, expect, bits):
    a = np.frombuffer(out, dtype="<u2" if bits > 8 else "u1").astype(np.int64)
    b = np.frombuffer(expect, dtype="<u2" if bits > 8 else "u1").astype(np.int64)
    if a.size != b.size:
        return f"size {a.size} != {b.size}"
    d = np.nonzero(a != b)[0]
    return f"{d.size} samples differ, first at {d[0]}: {a[d[0]]} vs {b[d[0]]}"


class Runner:
    def __init__(self, args):
        self.args = args
        self.work = os.path.join(HERE, "work", "avs")
        os.makedirs(self.work, exist_ok=True)
        self.clips = {}
        self.count = 0
        self.failures = 0

    def clip(self, fmt, width, height):
        key = (fmt, width, height)
        if key not in self.clips:
            frames = cases.make_clip(fmt, width, height)
            path = os.path.join(self.work, f"src_{fmt}_{width}x{height}.y4m")
            write_y4m(path, frames, fmt, width, height)
            self.clips[key] = (path, frames)
        return self.clips[key]

    def header(self, path):
        return (f'LoadPlugin("{fwd(self.args.bestsource)}")\nLoadPlugin("{fwd(self.args.plugin)}")\n'
                f'c = BSVideoSource("{fwd(path)}", cachemode=0)\n')

    def run(self, name, script):
        path = os.path.join(self.work, name + ".avs")
        with open(path, "w", encoding="utf-8") as f:
            f.write(script)
        p = subprocess.run([self.args.avs2yuv, "-raw", path, "-o", "-"], capture_output=True)
        return p.returncode, p.stdout, p.stderr.decode(errors="replace")

    def check(self, name, fmt, width, height, func, kwargs, props=None, mc=0, avs=None):
        if self.args.only and self.args.only not in name:
            return
        self.count += 1
        path, frames = self.clip(fmt, width, height)
        bits = cases.FORMATS[fmt][1]
        frames = [model.Frame([p.copy() for p in f.planes], dict(f.props)) for f in frames]
        script = self.header(path)
        if avs == "yuy2":
            script += "c = c.ConvertToYUY2()\n"
        elif avs == "yuva":
            script += "c = c.AddAlphaPlane(300)\n"
        for n, pr in (props or {}).items():
            frames[n].props.update(pr)
            seg = [f"c.Trim(0, {n - 1})"] if n > 0 else []
            seg.append(f"c.Trim({n}, -1)" + "".join(f'.propSet("{k}", {v})' for k, v in pr.items()))
            if n < cases.NUM_FRAMES - 1:
                seg.append(f"c.Trim({n + 1}, 0)")
            script += "c = " + " ++ ".join(seg) + "\n"
        args = [f"{k}={avs_value(v)}" for k, v in kwargs.items()]
        if mc:
            script += "".join(f"p{i} = c.{cases.MC_PREV[i]}\nq{i} = c.{cases.MC_NEXT[i]}\n" for i in range(mc))
            args.append("prev=[" + ",".join(f"p{i}" for i in range(mc)) + "]")
            args.append("next=[" + ",".join(f"q{i}" for i in range(mc)) + "]")
        post = {"yuy2": ".ConvertToYV16()", "yuva": ".RemoveAlphaPlane()"}.get(avs, "")
        script += f"c.{func}({', '.join(args)}){post}\n"
        rc, out, err = self.run(name, script)
        expect = serialize(cases.expected(frames, fmt, func, kwargs, mc), bits)
        if rc != 0 or out != expect:
            self.failures += 1
            detail = (err.strip().splitlines() or [""])[-1] if rc != 0 else describe(out, expect, bits)
            print(f"FAIL {name}: {detail}")
        elif self.args.verbose:
            print(f"ok   {name}")

    def expect_error(self, name, script_tail, text):
        if self.args.only and self.args.only not in name:
            return
        self.count += 1
        path, _ = self.clip("420p8", 32, 16)
        rc, _, err = self.run(name, self.header(path) + script_tail)
        if rc == 0 or text not in err:
            self.failures += 1
            print(f"FAIL {name}: expected error containing {text!r}, got rc={rc}: {err.strip()[-200:]}")
        elif self.args.verbose:
            print(f"ok   {name}")


ERRORS = [
    ("err-limit", "c.PixieDust(limit=0)", "limit must be 1..64"),
    ("err-mode", "c.PixieDust(mode=4)", "mode must be 0..3"),
    ("err-tr", "c.PixieDust(tr=6)", "tr must be 1..5"),
    ("err-temporal", "c.PixieDust(temporal=2)", "temporal must be 0"),
    ("err-limit-edge", "c.PixieDust(limit=4, limit_edge=5)", "must be <= limit"),
    ("err-edge-thresh", "c.PixieDust(limit_edge=1, edge_thresh=0)", "edge_thresh must be 1..255"),
    ("err-planes", 'c.PixieDust(planes="yx")', "invalid characters"),
    ("err-planes-none", 'c.PixieDust(planes="")', "selects no planes"),
    ("err-scthresh", "c.PixieDust(scthresh=-2)", "scthresh must be"),
    ("err-prev", "c.PixieDust(tr=2, prev=[c])", "prev array size"),
    ("err-mc-size", "c.PixieDust(prev=[c.Crop(0,0,-2,0)])", "size mismatch"),
    ("err-rgb", "c.ConvertToPlanarRGB().PixieDust()", "unsupported format"),
    ("err-float", "c.ConvertBits(32).PixieDust()", "float"),
    ("err-opt", "c.PixieDust(opt=7)", "instruction set not available"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plugin", default=os.path.join(ROOT, "build", "Dust.dll"))
    ap.add_argument("--avs2yuv", default=os.path.join(AVS_DIR, "avs2yuv64.exe"))
    ap.add_argument("--bestsource", default=os.path.join(AVS_DIR, "avisynthPlugins", "BestSource.dll"))
    ap.add_argument("--only", default="")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    r = Runner(args)
    for name, fmt, w, h, func, kwargs, extras in cases.cases():
        r.check(name, fmt, w, h, func, kwargs, **extras)
    for name, tail, text in ERRORS:
        r.expect_error(name, tail + "\n", text)
    print(f"{r.count - r.failures} of {r.count} checks passed")
    return 1 if r.failures else 0


if __name__ == "__main__":
    sys.exit(main())
