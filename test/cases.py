# SPDX-License-Identifier: MIT
"""Synthetic test clips and the test case matrix shared by test_avs.py (AviSynth+) and test_dust.py (VapourSynth).

A case is (name, format, width, height, function, arguments, extras). Arguments use the AviSynth+ spelling
(planes as a "yuv" string, interlaced as bool); the VapourSynth runner converts them. Extras:
  props: {frame: {property: value}} set on the source frames before filtering
  mc:    number of motion compensated clips for prev and next (simple transforms of the source, see MC_PREV/MC_NEXT)
  avs:   "yuy2" / "yuva" variants, AviSynth+ only
"""
import numpy as np

import model

# name -> (y4m colorspace tag, bits, chroma subsampling (log2 w, log2 h) or None for Y/Gray)
FORMATS = {
    "420p8": ("420jpeg", 8, (1, 1)),
    "422p8": ("422", 8, (1, 0)),
    "444p8": ("444", 8, (0, 0)),
    "y8": ("mono", 8, None),
    "420p10": ("420p10", 10, (1, 1)),
    "444p12": ("444p12", 12, (0, 0)),
    "420p16": ("420p16", 16, (1, 1)),
    "y16": ("mono16", 16, None),
}
NUM_FRAMES = 14
CUT = 7  # first frame of the second scene

MC_PREV = ["Reverse()", "FlipHorizontal()", "Reverse().FlipHorizontal()", "FlipVertical()", "Reverse().FlipVertical()"]
MC_NEXT = ["FlipVertical()", "Reverse().FlipVertical()", "FlipHorizontal()", "Reverse()",
           "FlipHorizontal().FlipVertical()"]


def make_clip(fmt, width, height, seed=1):
    """Two scenes with noise, spikes, a moving square and gradients; returns a list of model.Frame."""
    _, bits, ss = FORMATS[fmt]
    rng = np.random.default_rng(seed)
    maxv = (1 << bits) - 1
    scale = 1 << (bits - 8)
    frames = []
    for n in range(NUM_FRAMES):
        planes = []
        dims = [(height, width)]
        if ss:
            dims += [(height >> ss[1], width >> ss[0])] * 2
        for pi, (h, w) in enumerate(dims):
            yy, xx = np.mgrid[0:h, 0:w]
            if n < CUT:
                base = 60 + 120 * xx / max(w - 1, 1) + 30 * yy / max(h - 1, 1)
            else:
                base = 200 - 100 * yy / max(h - 1, 1)
            if pi:
                base = 128 + (base - 128) * 0.3
            sq = (abs(xx - (w // 4 + n * w // 20)) < max(w // 8, 1)) & (abs(yy - h // 2) < max(h // 6, 1))
            base = np.where(sq, 235 - pi * 60, base)
            noise = rng.normal(0, 6, (h, w))
            spikes = rng.random((h, w)) < 0.01
            v = (base + noise) * scale + spikes * 80 * scale
            planes.append(np.clip(np.rint(v), 0, maxv).astype(np.int64))
        frames.append(model.Frame(planes))
    return frames


def transform(frames, chain):
    """Applies an AviSynth style chain of Reverse()/FlipHorizontal()/FlipVertical() to model frames."""
    out = [model.Frame([p.copy() for p in f.planes], f.props) for f in frames]
    for step in chain.split("."):
        if step == "Reverse()":
            out = out[::-1]
        elif step == "FlipHorizontal()":
            out = [model.Frame([p[:, ::-1] for p in f.planes], f.props) for f in out]
        elif step == "FlipVertical()":
            out = [model.Frame([p[::-1, :] for p in f.planes], f.props) for f in out]
    return out


def model_args(kwargs):
    """Plugin arguments -> model.dust() arguments."""
    m = {}
    for k, v in kwargs.items():
        if k == "planes":
            m["planes"] = tuple(ch in v for ch in "yuv")
        elif k != "opt":
            m[k] = v
    return m


def expected(frames, fmt, func, kwargs, mc=0):
    """Model output for a case; frames already carry the case's props."""
    mk = model_args(kwargs)
    if mc:
        mk["prev"] = [transform(frames, MC_PREV[i]) for i in range(mc)]
        mk["next"] = [transform(frames, MC_NEXT[i]) for i in range(mc)]
    return model.dust(frames, FORMATS[fmt][1], func, **mk)


def cases():
    c = []

    def add(name, fmt, w, h, func, kwargs, **extras):
        c.append((name, fmt, w, h, func, kwargs, extras))

    for fmt in ("420p8", "420p10", "420p16", "y8", "444p12"):
        for func in ("FaeryDust", "PixieDust", "GoldDust", "SpaceDust"):
            add(f"{func}-{fmt}", fmt, 48, 32, func, {})
    for fmt in ("420p8", "420p16"):
        for tr in range(1, 6):
            for temporal in (0, 1):
                add(f"pixie-tr{tr}-t{temporal}-{fmt}", fmt, 40, 24, "PixieDust", {"tr": tr, "temporal": temporal})
        add(f"gold-limit64-{fmt}", fmt, 40, 24, "GoldDust", {"limit": 64, "tr": 2})
        add(f"faery-limit1-{fmt}", fmt, 40, 24, "FaeryDust", {"limit": 1, "temporal": 1, "tr": 3})
        for mode in range(4):
            add(f"pixie-mode{mode}-{fmt}", fmt, 40, 24, "PixieDust", {"mode": mode})
        add(f"space-mode1-{fmt}", fmt, 40, 24, "SpaceDust", {"mode": 1})
    for fmt in ("420p8", "420p10", "y16"):
        add(f"edge-{fmt}", fmt, 40, 24, "PixieDust", {"limit": 16, "limit_edge": 3, "edge_thresh": 24})
        add(f"edge0-{fmt}", fmt, 40, 24, "SpaceDust", {"limit": 10, "limit_edge": 0, "edge_thresh": 1})
        add(f"edge255-{fmt}", fmt, 40, 24, "GoldDust", {"limit": 20, "limit_edge": 20, "edge_thresh": 255})
    for pl in ("y", "u", "v", "uv", "yv"):
        add(f"planes-{pl}", "420p8", 40, 24, "PixieDust", {"planes": pl, "tr": 2})
    for sc in (-1, 0, 5, 20, 100):
        add(f"scthresh{sc}", "420p8", 40, 24, "PixieDust", {"scthresh": sc, "tr": 3})
    add("scprops-prev", "420p8", 40, 24, "PixieDust", {"scthresh": 0, "tr": 3}, props={4: {"_SceneChangePrev": 1}})
    add("scprops-next", "420p16", 40, 24, "PixieDust", {"scthresh": 0, "tr": 2}, props={9: {"_SceneChangeNext": 1}})
    add("scprops-off", "420p8", 40, 24, "PixieDust", {"scthresh": -1, "tr": 2}, props={4: {"_SceneChangePrev": 1}})
    for w, h in ((17, 13), (3, 3), (1, 5), (6, 1), (2, 2)):
        for func in ("PixieDust", "SpaceDust"):
            add(f"border-{func}-{w}x{h}", "444p8", w, h, func, {"limit_edge": 1, "limit": 9})
    add("border-y16-odd", "y16", 19, 11, "GoldDust", {"limit_edge": 2, "limit": 12, "tr": 2})
    for fmt in ("420p8", "420p10", "444p8"):
        add(f"fields-{fmt}", fmt, 40, 24, "PixieDust", {"interlaced": True, "limit_edge": 2, "limit": 8})
    add("fields-odd", "444p8", 21, 15, "GoldDust", {"interlaced": True, "tr": 2})
    add("fields-off", "420p8", 40, 24, "PixieDust", {"interlaced": False}, props={3: {"_FieldBased": 2}})
    add("fields-prop", "420p8", 40, 24, "PixieDust", {}, props={3: {"_FieldBased": 2}, 8: {"_FieldBased": 1}})
    add("fields-space", "420p16", 40, 24, "SpaceDust", {"interlaced": True})
    for tr in (1, 2, 5):
        add(f"mc-tr{tr}", "420p8", 40, 24, "PixieDust", {"tr": tr}, mc=tr)
        add(f"mc-tr{tr}-median-16", "420p16", 40, 24, "GoldDust", {"tr": tr, "temporal": 1}, mc=tr)
    add("mc-nosc", "420p8", 40, 24, "FaeryDust", {"tr": 2, "scthresh": -1}, mc=2)
    for func, kw in (("PixieDust", {"tr": 2}), ("SpaceDust", {"limit_edge": 2}), ("GoldDust", {"planes": "uv"})):
        add(f"yuy2-{func}", "422p8", 40, 24, func, kw, avs="yuy2")
    add("yuva", "420p10", 40, 24, "PixieDust", {"tr": 2}, avs="yuva")
    add("opt0", "420p8", 40, 24, "PixieDust", {"opt": 0})
    return c
