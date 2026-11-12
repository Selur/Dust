#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Runs DustDenoise() of scripts/Dust.avsi (AviSynth+) and scripts/Dust.py (VapourSynth) on a real clip (Windows).

For every setting (mode, tr 1..5, mean/median, planes, edge-aware, field mode, no MC, 8 and 16 bit) both wrappers
must run, keep frame count and size, and stay within `limit` of the source in every sample (the guarantee of the
plugin, which also proves the motion compensated clips reached it in the right format). The two hosts are compared
byte for byte: with mc=false they must match; with motion compensation the result depends on the MVTools builds of
the two hosts, so differences there are only reported.

Usage: F:\\Hybrid\\64bit\\Vapoursynth\\python.exe test\\test_wrappers.py [CLIP]
"""
import argparse
import os
import subprocess
import sys

import numpy as np
import vapoursynth as vs

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
AVS_DIR = r"F:\Hybrid\64bit\Avisynth"
VS_PLUGINS = r"F:\Hybrid\64bit\Vapoursynth\Lib\site-packages\vapoursynth\plugins2"
CLIP = r"G:\TestClips&Co\files\interlaceAndTelecineSamples\funkyMixedInterlacing.m2v"
FRAMES = 24

# (name, AviSynth arguments, VapourSynth arguments, limit)
SETTINGS = [
    ("pixie", "", {}, 5),
    ("faery-tr3-median", 'mode="faery", tr=3, temporal=1', dict(mode="faery", tr=3, temporal=1), 2),
    ("gold-tr5", 'mode="gold", tr=5', dict(mode="gold", tr=5), 8),
    ("pixie-tr2-edge", "tr=2, limit=16, limit_edge=3, edge_thresh=24",
     dict(tr=2, limit=16, limit_edge=3, edge_thresh=24), 16),
    ("pixie-tr4-y", 'tr=4, planes="y"', dict(tr=4, planes=[0]), 5),
    ("pixie-uv", 'planes="uv"', dict(planes=[1, 2]), 5),
    ("space", 'mode="space"', dict(mode="space"), 6),
    ("fields-tr2", "interlaced=true, tr=2", dict(interlaced=True, tr=2), 5),
    ("nomc-tr2", "mc=false, tr=2, temporal=1", dict(mc=False, tr=2, temporal=1), 5),
]


def fwd(p):
    return os.path.abspath(p).replace("\\", "/")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip", nargs="?", default=CLIP)
    ap.add_argument("--plugin", default=os.path.join(ROOT, "build", "Dust.dll"))
    a = ap.parse_args()
    core = vs.core
    for dll in ("libbestsource.dll", "mvtools.dll"):
        core.std.LoadPlugin(os.path.join(VS_PLUGINS, dll))
    core.std.LoadPlugin(os.path.abspath(a.plugin))
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from Dust import DustDenoise

    work = os.path.join(HERE, "work", "wrappers")
    os.makedirs(work, exist_ok=True)
    base = core.bs.VideoSource(a.clip)[:FRAMES]
    base = core.std.SetFieldBased(base, 2)
    failures = 0
    for bits in (8, 16):
        src = base if bits == 8 else core.std.Expr(base, "x 256 *", format=base.format.replace(bits_per_sample=16).id)
        conv = "" if bits == 8 else ".ConvertBits(16, fulls=false, fulld=false)"
        ref = [[np.asarray(f[p]).astype(np.int64) for p in range(3)] for f in src.frames()]
        for name, avs_args, vs_args, limit in SETTINGS:
            lim = limit << (bits - 8)
            script = (f'LoadPlugin("{fwd(os.path.join(AVS_DIR, "avisynthPlugins", "BestSource.dll"))}")\n'
                      f'LoadPlugin("{fwd(os.path.join(AVS_DIR, "avisynthPlugins", "mvtools2.dll"))}")\n'
                      f'LoadPlugin("{fwd(a.plugin)}")\nImport("{fwd(os.path.join(ROOT, "scripts", "Dust.avsi"))}")\n'
                      f'BSVideoSource("{fwd(a.clip)}").Trim(0, {FRAMES - 1}).AssumeTFF(){conv}'
                      f'.DustDenoise({avs_args})\n')
            path = os.path.join(work, f"{name}-{bits}.avs")
            with open(path, "w", encoding="utf-8") as f:
                f.write(script)
            p = subprocess.run([os.path.join(AVS_DIR, "avs2yuv64.exe"), "-raw", path, "-o", "-"], capture_output=True)
            label = f"{bits:2} bit {name:18}"
            if p.returncode != 0:
                failures += 1
                print(f"FAIL {label} AviSynth: {p.stderr.decode(errors='replace').strip()[-300:]}")
                continue
            try:
                out = DustDenoise(src, **vs_args)
                vsb = bytearray()
                worst = 0
                for n, fr in enumerate(out.frames()):
                    for pl in range(3):
                        arr = np.asarray(fr[pl])
                        vsb += arr.tobytes()
                        worst = max(worst, int(np.abs(arr.astype(np.int64) - ref[n][pl]).max()))
            except vs.Error as e:
                failures += 1
                print(f"FAIL {label} VapourSynth: {str(e).strip()[-300:]}")
                continue
            dt = "<u2" if bits > 8 else "u1"
            av = np.frombuffer(p.stdout, dt).astype(np.int64)
            vv = np.frombuffer(bytes(vsb), dt).astype(np.int64)
            srcb = np.concatenate([pl.ravel() for fr in ref for pl in fr])
            ok = av.size == vv.size == srcb.size and worst <= lim and int(np.abs(av - srcb).max()) <= lim
            same = ok and np.array_equal(av, vv)
            must_match = not vs_args.get("mc", True) or vs_args.get("mode") == "space"
            if not ok or (must_match and not same):
                failures += 1
            share = np.count_nonzero(av != vv) / max(av.size, 1) if av.size == vv.size else 1.0
            print(f"{'ok  ' if ok and (same or not must_match) else 'FAIL'} {label} "
                  f"{'identical' if same else f'hosts differ in {share:.2%} of samples'}"
                  + ("" if ok else " (size or limit violated)"))
    print(f"{failures} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
