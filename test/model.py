#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Reference model of Dust in numpy, written directly from ALGORITHM.md (not from the C code).

A clip is a list of frames; a frame is a list of 2D integer numpy arrays (planes: Y or Y, U, V) plus a dict of
frame properties. Used by test_avs.py (and later the VapourSynth tests) to check the plugin bit for bit.
"""
import numpy as np

DEFAULTS = {  # function -> (mode, limit)
    "FaeryDust": (0, 2),
    "PixieDust": (1, 5),
    "GoldDust": (2, 8),
    "SpaceDust": (3, 6),
}


class Frame:
    def __init__(self, planes, props=None):
        self.planes = [np.asarray(p, dtype=np.int64) for p in planes]
        self.props = dict(props or {})


def _pad(a):
    """Mirrors one sample at every border without repeating the edge sample (a single sample mirrors to itself)."""
    mode_y = "reflect" if a.shape[0] > 1 else "edge"
    a = np.pad(a, ((1, 1), (0, 0)), mode=mode_y)
    mode_x = "reflect" if a.shape[1] > 1 else "edge"
    return np.pad(a, ((0, 0), (1, 1)), mode=mode_x)


def limit_map(c, bits, limit, limit_edge, edge_thresh):
    p = _pad(c >> (bits - 8))
    a, b, cc = p[:-2, :-2], p[:-2, 1:-1], p[:-2, 2:]
    d, f = p[1:-1, :-2], p[1:-1, 2:]
    g, h, i = p[2:, :-2], p[2:, 1:-1], p[2:, 2:]
    sx = (cc + 2 * f + i) - (a + 2 * d + g)
    sy = (g + 2 * h + i) - (a + 2 * b + cc)
    grad = np.minimum(np.maximum(np.abs(sx), np.abs(sy)), edge_thresh)
    return limit - (grad * (limit - limit_edge) + edge_thresh // 2) // edge_thresh


def filter_plane(win, centre, bits, mode, temporal, limit, limit_edge, edge_thresh):
    """win: list of 2D arrays (the window), centre index. Returns the filtered plane."""
    c = win[centre]
    n = len(win)
    t = s = None
    if mode != 3:
        stack = np.stack(win)
        if temporal == 1:
            t = np.sort(stack, axis=0)[n // 2]
        else:
            t = (stack.sum(axis=0) + n // 2) // n
    if mode != 0:
        p = _pad(c)
        s = (4 * c + p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:] + 4) >> 3
    f = {0: t, 1: None, 2: None, 3: s}[mode]
    if mode == 1:
        f = (2 * t + s + 1) // 3
    elif mode == 2:
        f = (t + 2 * s + 1) // 3
    shift = bits - 8
    lim = limit_map(c, bits, limit, limit_edge, edge_thresh) if limit_edge >= 0 else np.full(c.shape, limit)
    lim = lim << shift
    lo = np.maximum(c - lim, 0)
    hi = np.minimum(c + lim, (1 << bits) - 1)
    return np.minimum(np.maximum(f, lo), hi)


def scene_cut(fa, fb, bits, scthresh):
    """Scene change between frame fa and the following frame fb."""
    if scthresh < 0:
        return False
    if fb.props.get("_SceneChangePrev") == 1 or fa.props.get("_SceneChangeNext") == 1:
        return True
    if scthresh == 0:
        return False
    a = fa.planes[0][::2, ::2] >> (bits - 8)
    b = fb.planes[0][::2, ::2] >> (bits - 8)
    return int(np.abs(a - b).sum()) // a.size > scthresh


def window_offsets(n, num_frames, tr, cut):
    """Offsets -tr..tr -> frame offset actually used (ALGORITHM.md stage 1). cut(f) = scene change between f, f+1."""
    offs = {0: 0}
    for side in (-1, 1):
        valid, blocked = 0, False
        for k in range(1, tr + 1):
            f = n + side * k
            if not blocked:
                if f < 0 or f >= num_frames or cut(f if side < 0 else f - 1):
                    blocked = True
                else:
                    valid = k
            offs[side * k] = side * valid
    return [offs[i] for i in range(-tr, tr + 1)]


def dust(clip, bits, func="PixieDust", limit=None, mode=None, tr=1, temporal=0, limit_edge=-1, edge_thresh=32,
         planes=(True, True, True), scthresh=20, interlaced=None, prev=None, next=None):
    """Returns the filtered clip (list of Frame). prev/next: lists of tr clips or None."""
    dmode, dlimit = DEFAULTS[func]
    mode = dmode if mode is None else mode
    limit = dlimit if limit is None else limit
    if func == "SpaceDust" and mode != 3:
        tr, temporal, prev, next = 1, 0, None, None
    out = []
    nf = len(clip)
    for n in range(nf):
        centre = clip[n]
        if mode == 3:
            offs = [0]
        else:
            offs = window_offsets(n, nf, tr, lambda f: scene_cut(clip[f], clip[f + 1], bits, scthresh))

        def frame_at(o):
            if o == 0:
                return centre
            if o < 0:
                return prev[-o - 1][n] if prev else clip[n + o]
            return next[o - 1][n] if next else clip[n + o]

        win = [frame_at(o) for o in offs]
        fields = interlaced if interlaced is not None else centre.props.get("_FieldBased", 0) > 0
        res = []
        for pi, plane in enumerate(centre.planes):
            if not planes[pi]:
                res.append(plane.copy())
                continue
            o = np.empty_like(plane)
            for rows in ((slice(0, None, 2), slice(1, None, 2)) if fields else (slice(None),)):
                w = [fr.planes[pi][rows] for fr in win]
                if w[0].shape[0] == 0:
                    continue
                o[rows] = filter_plane(w, len(offs) // 2, bits, mode, temporal, limit, limit_edge, edge_thresh)
            res.append(o)
        out.append(Frame(res, centre.props))
    return out
