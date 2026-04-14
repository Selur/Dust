# Algorithm details

This document describes what DustPort does at each internal stage. Read this if
you want to understand why the plugin behaves the way it does, tune parameters
beyond the defaults, or adapt the code for other projects.

## Overview

For every output frame, DustPort produces each pixel via the following path:

```
                  +--- prev frames (raw or MC) ---+
source frame -----+                               +---> temporal aggregate ---+
                  +--- next frames (raw or MC) ---+                           |
                                                                              v
                                              +----- spatial 3x3 blend -----> combiner
                                              |                               |
                                              +---- source frame neighbours   |
                                                                              v
                                                                  limit-clipping --> out
                                                                              ^
                                                                              |
                                                         per-pixel limit map (optional)
```

## Stage 1 — Temporal aggregate

A window of `2 * tr + 1` frames is collected. If `prev`/`next` are supplied,
the motion-compensated versions are used; otherwise raw `child[n ± k]` with
boundary clamping.

Two aggregation modes:

- `temporal=0` (mean): average of all frames in window. Vectorised via u16 sum
  followed by multiply-by-reciprocal for division.
- `temporal=1` (median): sorting-network median via SSE2 (8-bit) or SSE4.1
  (16-bit) `_mm_min_epu8/16` and `_mm_max_epu8/16` operations. Hand-coded
  networks for window sizes 3, 5, 7, 9, 11.

**Trade-off**: mean reduces white Gaussian noise by √N (optimal for shot
noise). Median rejects outliers (better for dead pixels, film dust specks).
For clean digital sources with only grain, mean is usually preferable.

## Stage 2 — Spatial 3x3 blend

Only applied when `mode != 0` (i.e. not FaeryDust) and `mode != 3` (not
SpaceDust-only). The formula is a weighted average:

```
spatial = 0.5 * centre + 0.125 * (top + bottom + left + right)
```

Implemented in SSE2 as three `_mm_avg_epu8` chains for 8-bit, or via
`avg_epu16` macro (no native u16 avg in SSE) for 16-bit.

**Why this specific kernel.** Preserves DC and reduces noise uniformly while
keeping centre weight dominant. Detail preservation is adequate for its
purpose — the real safety net is Stage 4 (limit-clipping).

## Stage 3 — Combiner

The temporal estimate and spatial estimate are combined according to mode:

| Mode | Combiner formula               | Character              |
|:----:|--------------------------------|------------------------|
| 0    | `out = temporal`               | Temporal-only (Faery)  |
| 1    | `out = (2 * temporal + spatial) / 3` | Balanced (Pixie) |
| 2    | `out = (temporal + 2 * spatial) / 3` | Spatial-dominant (Gold) |
| 3    | `out = spatial`                | Spatial-only (Space)   |

For 8-bit, implemented via two `_mm_avg_epu8` for modes 1 and 2. For 16-bit,
via chained `avg_epu16`.

## Stage 4 — Limit-clipping

The key architectural guarantee. Each output pixel is clamped:

```
out = clamp(stage3_result, source - limit, source + limit)
```

Where `limit` is either:
- A global scalar (`limit` parameter) if edge-aware is disabled
- Per-pixel value from the limit map (see Stage 0) if enabled

Implementation uses saturating arithmetic:
- 8-bit: `_mm_subs_epu8` / `_mm_adds_epu8` / `_mm_min_epu8` / `_mm_max_epu8`
- 16-bit: same intrinsics with `_u16` suffix (SSE4.1 for min/max)

**Why this matters.** The filter can only do what limit-clipping allows. If
the upstream kernel produces a terrible estimate, limit-clipping caps the
damage at `±limit` from the source. This prevents drift, hallucinated detail,
and destructive artifacts — at the cost of sometimes leaving visible noise
when the limit is below the noise amplitude.

## Stage 0 — Edge-aware limit map (optional)

Computed once per plane, before Stage 4. Runs only when `limit_edge >= 0`.

For each pixel, Sobel 3×3 gradient magnitude is computed (L∞ approximation
`max(|Sx|, |Sy|)` rather than L2 — 40% cheaper and visually equivalent for
our purposes). Then mapped to per-pixel limit:

```
t = min(grad, edge_thresh) / edge_thresh     # in [0, 1]
per_pixel_limit = limit_flat - t * (limit_flat - limit_edge)
```

Result stored in an 8-bit buffer with pitch = width, regardless of source bit
depth. For 16-bit sources, the map is scaled up at point-of-use by the kernel.

**Why 8-bit map for 16-bit source.** The limit parameter is always expressed
in 8-bit scale (1..64 for `limit`). A 16-bit source uses an effective limit of
`limit << (bitdepth - 8)`. Storing the map in 8-bit saves 2× memory
bandwidth and is precision-sufficient — sub-unit gradations in limit have no
visible effect.

**Why Sobel on down-shifted source for 16-bit.** Preserves the semantics of
`edge_thresh` (always in 8-bit scale) and is one shift cheaper. The high bits
of 16-bit data contain the same structural information as the 8-bit
equivalent for gradient detection purposes.

## Bit depth handling

The plugin supports 8/10/12/14/16 bit planar YUV. Internal representation:

- 8-bit: u8 throughout
- 10..16-bit: u16 throughout with max_val = `(1 << bitdepth) - 1`

All parameters (`limit`, `limit_edge`, `edge_thresh`) stay in 8-bit scale.
Internal scaling by `(1 << (bitdepth - 8))` is applied once at kernel call site.
This means the same script works correctly across bit depths without parameter
adjustment.

## Per-plane processing

The `planes` parameter selects which planes are processed. Unselected planes
are copied from source using `env->BitBlt`.

Edge-aware limit map is computed only for processed planes (skipping a plane
saves both the kernel run and the Sobel computation).

## Memory layout

Scratch buffers for limit maps are allocated per-plane and reused across
frames. First frame allocates, subsequent frames reuse. On resolution change
(from AviSynth plugin perspective this is unusual), buffers grow to fit.

Temporal window frames are collected as `std::vector<PVideoFrame>` per
GetFrame call. Each AVS+ frame is reference-counted, so collecting up to 11
frames is cheap.

## Threading

`SetCacheHints` returns `MT_SERIALIZED`. DustPort uses stateful limit-map
buffers that would be unsafe under `MT_MULTI_INSTANCE` without additional
locking. Since the plugin is mostly memory-bandwidth bound on SSE2 paths,
the serialisation cost is usually acceptable. If throughput matters,
`Prefetch(4)` upstream can effectively parallelise the surrounding pipeline.
