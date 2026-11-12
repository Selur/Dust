# Algorithm details

This document describes what Dust computes, exactly. Every instruction set (C, SSE4.1, AVX, AVX2, NEON) gives
bit-identical results to these formulas, in AviSynth+ and VapourSynth alike. Read it to understand why the plugin
behaves the way it does, to tune parameters beyond the defaults, or to reimplement it.

Progressive and interlaced (woven) input are both supported; for interlaced input the spatial stages work per
field, see [Interlaced input](#interlaced-input).

## Overview

For every output frame `n`, each processed plane goes through these stages:

```
 clip / prev / next  --> 1. window assembly (clip ends, scene changes) --+
                                                                        v
                                                     2. temporal aggregate (mean or median) = t
 centre frame ------------------------------------> 3. spatial 3x3 blend                     = s
                                                                        v
                                                     4. combiner (mode)                      = f
 centre frame --> 0. limit map (optional) -------> 5. limit-clipping: out = clamp(f, c - L, c + L)
```

`c` is the source pixel of frame `n`. Stages 2 and 3 are skipped when the mode does not use them: mode 0 (FaeryDust)
uses only `t`, mode 3 (SpaceDust) uses only `s` and fetches no other frame.

All level parameters (`limit`, `limit_edge`, `edge_thresh`, `scthresh`) are given on the 8-bit scale, whatever the
bit depth of the clip (see [Bit depth](#bit-depth)).

## Stage 1 — Window assembly

The window holds `2 * tr + 1` frames, the centre frame `n` in the middle:

- Position `-k` (k = 1..tr): `prev[k-1]` at frame `n` if `prev` is given, else the clip at frame `n - k`.
- Position `+k`: `next[k-1]` at frame `n` if `next` is given, else the clip at frame `n + k`.

`prev[k-1]` must be frame `n - k` motion compensated onto frame `n`. With MVTools that is
`MCompensate(clip, super, MAnalyse(super, isb=false, delta=k))`; `next[k-1]` uses `isb=true`. (The order matters
only for documentation: mean and median do not depend on it.)

**Invalid positions are replaced by the nearest valid frame in the same direction.** A position is invalid if

- it lies outside the clip (`n - k < 0` or `n + k > last frame`), or
- a scene change lies between it and frame `n`.

"Nearest valid frame in the same direction" is position `±j` with the largest `j < k` that is still valid, which is
the centre frame itself if the cut (or the clip end) is right next to `n`. The same rule holds with and without
`prev`/`next`: an invalid MC position is replaced by the nearest valid MC position on that side, or by the centre.
So no frame from another scene ever enters the window, and the kernels always see `2 * tr + 1` frames. The price is
less noise reduction near cuts and clip ends, because the repeated frames carry more weight.

### Scene changes

A scene change lies between frames `f` and `f + 1` if any of these holds:

1. frame `f + 1` has the frame property `_SceneChangePrev` = 1,
2. frame `f` has the frame property `_SceneChangeNext` = 1,
3. the built-in detector fires: `scthresh > 0` and the mean absolute difference of the luma of `f` and `f + 1`,
   on the 8-bit scale, is greater than `scthresh`.

The detector compares every second sample of every second row of the luma plane (Y of YUV and Gray, the luma bytes
of YUY2) of the source clip, never of `prev`/`next`:

```
mad = sum(|(a >> shift) - (b >> shift)|) / count        (integer division, count = number of compared samples)
cut = mad > scthresh
```

`scthresh`: 1..255, default 20; 0 = only frame properties, -1 = no scene change handling at all (the window rule then
only handles the clip ends). Properties are read in both hosts (AviSynth+ from interface version 8; older versions
only use the detector). VapourSynth's `misc.SCDetect` or `mv.SCDetection` set them, so a script can use a better
detector than the built-in one and pass `scthresh=0`.

Mode 3 (SpaceDust) has no window and ignores all of this.

## Stage 2 — Temporal aggregate

Over the `N = 2 * tr + 1` values `v[i]` of the window at the pixel position:

- `temporal=0`, mean: `t = (sum(v) + N / 2) / N` (integer division, so rounded to nearest, halves up).
- `temporal=1`, median: `t` = the middle value of the sorted `v` (N is always odd).

SIMD: the mean sums in 16-bit lanes for 8-bit input (at most 11 * 255) and in 32-bit lanes for 9–16 bit, then
divides as `(x * m) >> k`; the constants `m`, `k` are searched when the plugin loads and accepted only if they give
the exact quotient for every possible sum (the same holds for the divisions by 3 of stage 4 and by `edge_thresh`
of stage 0). The median uses selection networks for 3, 5, 7, 9 and 11 values (Batcher's merge sort, pruned),
each verified exhaustively with the 0-1 principle.

The first and last column of every row use the scalar formula (they need mirrored neighbours); the interior runs in
vectors (8 to 32 samples, depending on instruction set and bit depth), the last vector of a row overlapping the one
before instead of a scalar tail.

Mean reduces independent noise by up to `sqrt(N)` on well aligned frames; median rejects outliers (dust specks,
dead pixels) better. For clean digital sources with only grain, mean is usually preferable.

## Stage 3 — Spatial 3x3 blend

```
s = (4 * c + up + down + left + right + 4) >> 3
```

i.e. `0.5 * c + 0.125 * (up + down + left + right)`, rounded to nearest. The corners are not used. In 9–16 bit the
sum needs 32 bits (`8 * 65535`), the SIMD code widens accordingly.

The kernel keeps DC and reduces noise while keeping the centre dominant; the real safety net is stage 5.

## Stage 4 — Combiner

| Mode | Function | Formula | Character |
|:----:|---|---|---|
| 0 | FaeryDust | `f = t` | temporal only |
| 1 | PixieDust | `f = (2 * t + s + 1) / 3` | temporal-weighted |
| 2 | GoldDust | `f = (t + 2 * s + 1) / 3` | spatial-weighted |
| 3 | SpaceDust | `f = s` | spatial only |

Integer division. Each function sets its own default `mode` and `limit`; `mode` can override it (SpaceDust has no
temporal parameters, so with `mode < 3` it uses `tr = 1` without motion compensation).

## Stage 5 — Limit-clipping

```
lo  = max(c - L, 0)
hi  = min(c + L, max_value)
out = min(max(f, lo), hi)
```

`L` is `limit` (scaled to the bit depth), or the per-pixel value of the limit map when edge-aware mode is on.
Whatever the earlier stages produce, the output never leaves `c ± L`: no drift, no invented detail, at the cost of
leaving noise that is larger than `L`.

## Stage 0 — Edge-aware limit map (optional)

Runs when `limit_edge >= 0`, for every frame and every processed plane, on the centre frame, before stage 5.

```
x'   = x >> shift                                   (all 9 samples of the 3x3 neighbourhood)
Sx   = (c' + 2 f' + i') - (a' + 2 d' + g')          a b c
Sy   = (g' + 2 h' + i') - (a' + 2 b' + c')          d e f
grad = min(max(|Sx|, |Sy|), edge_thresh)            g h i
L    = limit - (grad * (limit - limit_edge) + edge_thresh / 2) / edge_thresh
```

So `L` falls linearly from `limit` (flat area) to `limit_edge` (gradient at or above `edge_thresh`).

`edge_thresh` is a **Sobel magnitude**, not a difference between two pixel values: a sharp step of height `h`
gives `|S| = 4 h`. The default 32 therefore treats a step of 8 code values (8-bit scale) as a full edge; a soft
edge spread over several pixels gives less. The magnitude is `max(|Sx|, |Sy|)` (L-infinity), not the Euclidean
length: diagonal edges count slightly less than with `sqrt(Sx² + Sy²)`.

The map is stored per pixel on the 8-bit scale and scaled with the clip (`L << shift`) when used. Sobel works on
the down-shifted samples so that `edge_thresh` means the same at every bit depth; gradients below one 8-bit step
are ignored.

The map is computed per call in a buffer owned by that call, so the filter is thread-safe.

## Borders

Every pixel, including the first and last row and column, gets the same formulas. Neighbours outside the plane (the
field, in field mode) are **mirrored** at the border without repeating the edge sample (row `-1` is row `1`, column
`W` is column `W - 2`), as in VapourSynth's `std.Convolution`. A plane with a single row or column uses the centre for
the missing neighbours. This holds for the spatial blend and for the edge detector; on a border the mirrored Sobel has
no component across the border, so edges parallel to it are found and edges hitting it at a right angle are not.

YUY2 mirrors per component: luma neighbours are 2 bytes apart, chroma neighbours 4 bytes.

Temporal: the clip ends are handled by the window rule of stage 1.

## Interlaced input

With field mode on, every plane is processed as two independent fields: the even rows (0, 2, 4, …) and the odd
rows (1, 3, 5, …), each like a plane of its own with twice the row stride and half the height (the even field gets
the extra row when the height is odd). Chroma planes are split the same way, which matches interlaced 4:2:0, where
chroma rows alternate between the fields too.

- Spatial blend and edge detector: `up`/`down` are the rows above and below **in the same field** (two frame rows
  apart); borders are mirrored at the field's first and last row.
- Temporal aggregate: unchanged. It works per pixel position, so it combines the same field of the window frames
  (top with top, bottom with bottom).
- Scene changes: unchanged, measured on whole frames.
- Field order (`TFF`/`BFF`) does not matter, the two fields never mix.

Field mode is selected per frame:

- parameter `interlaced` given: `true` = field mode, `false` = progressive, for every frame;
- not given: from the frame property `_FieldBased` of the centre frame (1 = bottom field first, 2 = top field
  first -> field mode; 0 or missing -> progressive). AviSynth+ reads it from interface version 8; older versions
  treat the clip as progressive.

Motion compensated `prev`/`next` must be made per field for interlaced input: separate the fields, run MVTools on
the field clip with `delta = 2 * k` (the same field of frame `n ± k`), weave the compensated fields back. The
wrapper functions in `Dust.avsi` / `Dust.py` do this when `interlaced` is set.

## Bit depth

Integer formats with 8–16 bits per sample (`max_value = 2^bits - 1`, `shift = bits - 8`):

- `limit`, `limit_edge` and the map values are used as `value << shift`.
- `edge_thresh` and `scthresh` are compared with values computed on `sample >> shift`.

So the same script behaves the same at every bit depth. 32-bit float is not supported in this version.

## Planes

`planes` selects the processed planes (AviSynth+: a string of `y`, `u`, `v`; VapourSynth: a list of plane
indices); the others are copied unchanged. Skipped planes cost neither the kernel nor the limit map. Each plane is
processed on its own (chroma planes in their own resolution, their limit map from their own content). Alpha is
copied. YUY2 is processed as three components (Y, U, V) in their own resolution, exactly like planar 4:2:2:
`planes` and the edge-aware mode work there too, and the output equals that of the same clip in 4:2:2 planar.

## Threading and frame properties

The filter keeps no state between frames: AviSynth+ `MT_NICE_FILTER`, VapourSynth `fmParallel`, results independent
of thread count and request order. Frame properties are copied from the source frame.
