#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compares the VapourSynth and the AviSynth+ interface of the same Dust build on real clips (Windows).

Both hosts read the same clip with BestSource (first --frames frames); 10 and 16 bit are made by a bit shift in both
hosts (forced as a shift, also for full range clips), Gray/Y by taking the luma plane. Every output byte must match. Run it with a VapourSynth Python that has the
BestSource plugin, e.g. Hybrid's:

  F:\\Hybrid\\64bit\\Vapoursynth\\python.exe test\\compare_hosts.py [CLIP...]
"""
import argparse
import hashlib
import os
import subprocess
import sys

import vapoursynth as vs

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
AVS_DIR = r"F:\Hybrid\64bit\Avisynth"
VS_PLUGINS = r"F:\Hybrid\64bit\Vapoursynth\Lib\site-packages\vapoursynth\plugins2"
CLIPS = [r"G:\TestClips&Co\files\interlaceAndTelecineSamples\funkyMixedInterlacing.m2v",
         r"G:\TestClips&Co\LightningStormVeracruz_denoiseTest.mp4"]

# (function, AviSynth arguments)
CALLS = [
    ("FaeryDust", ""), ("PixieDust", ""), ("GoldDust", ""), ("SpaceDust", ""),
    ("PixieDust", "tr=2, temporal=1"), ("GoldDust", "tr=5"), ("FaeryDust", "tr=3, temporal=1, scthresh=5"),
    ("PixieDust", "limit=16, limit_edge=3, edge_thresh=24"), ("SpaceDust", 'limit=10, limit_edge=2, planes="y"'),
    ("PixieDust", 'planes="uv", scthresh=-1'), ("GoldDust", "interlaced=true, tr=2"),
    ("PixieDust", "mode=0, limit=20, scthresh=0"),
]
# name -> (AviSynth conversion, bit shift, luma only)
# ConvertBits with fulls/fulld=false: a plain shift, even when the clip is flagged as full range.
SHIFT = "ConvertBits({}, fulls=false, fulld=false)"
FORMATS = {
    "8bit": ("", 0, False),
    "10bit": ("." + SHIFT.format(10), 2, False),
    "16bit": ("." + SHIFT.format(16), 8, False),
    "y8": (".ConvertToY()", 0, True),
    "y16": (".ConvertToY()." + SHIFT.format(16), 8, True),
}


def fwd(p):
    return os.path.abspath(p).replace("\\", "/")


def vs_kwargs(args):
    kw = {}
    for part in filter(None, (s.strip() for s in args.split(","))):
        k, v = part.split("=", 1)
        if v in ("true", "false"):
            kw[k] = int(v == "true")
        elif v.startswith('"'):
            kw[k] = [i for i, ch in enumerate("yuv") if ch in v.strip('"')]
        else:
            kw[k] = int(v)
    return kw


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plugin", default=os.path.join(ROOT, "build", "Dust.dll"))
    ap.add_argument("--avs2yuv", default=os.path.join(AVS_DIR, "avs2yuv64.exe"))
    ap.add_argument("--avs-bestsource", default=os.path.join(AVS_DIR, "avisynthPlugins", "BestSource.dll"))
    ap.add_argument("--vs-bestsource", default=os.path.join(VS_PLUGINS, "libbestsource.dll"))
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("clips", nargs="*", default=CLIPS)
    a = ap.parse_args()
    work = os.path.join(HERE, "work", "hosts")
    os.makedirs(work, exist_ok=True)
    core = vs.core
    core.std.LoadPlugin(a.vs_bestsource)
    core.std.LoadPlugin(os.path.abspath(a.plugin))
    failures = total = 0
    for ci, clip in enumerate(a.clips):
        base = core.bs.VideoSource(clip)[:a.frames]
        print(f"clip{ci}: {clip} ({base.format.name} {base.width}x{base.height})")
        for fname, (conv, shift, luma) in FORMATS.items():
            src = core.std.ShufflePlanes(base, 0, vs.GRAY) if luma else base
            if shift:
                fmt = src.format.replace(bits_per_sample=8 + shift)
                src = core.std.Expr(src, f"x {1 << shift} *", format=fmt.id)
            for func, args in CALLS:
                if luma and "planes" in args:
                    continue
                total += 1
                script = (f'LoadPlugin("{fwd(a.avs_bestsource)}")\nLoadPlugin("{fwd(a.plugin)}")\n'
                          f'BSVideoSource("{fwd(clip)}").Trim(0, {a.frames - 1}){conv}.{func}({args})\n')
                path = os.path.join(work, f"h{total:04d}.avs")
                with open(path, "w", encoding="utf-8") as f:
                    f.write(script)
                p = subprocess.run([a.avs2yuv, "-raw", path, "-o", "-"], capture_output=True)
                if p.returncode != 0 or not p.stdout:
                    failures += 1
                    print(f"FAIL clip{ci} {fname} {func}({args}): {p.stderr.decode(errors='replace')[-300:]}")
                    continue
                out = bytearray()
                for fr in getattr(core.dust, func)(src, **vs_kwargs(args)).frames():
                    for pl in range(fr.format.num_planes):
                        out += bytes(fr[pl])
                ok = hashlib.md5(p.stdout).digest() == hashlib.md5(bytes(out)).digest()
                failures += not ok
                print(f"{'ok  ' if ok else 'FAIL'} clip{ci} {fname:6} {func}({args})"
                      + ("" if ok else f" ({len(p.stdout)} vs {len(out)} bytes)"))
    print(f"{total - failures} of {total} cases identical")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
