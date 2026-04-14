# DustPort

**A limit-clipping spatiotemporal denoiser for AviSynth+, in the philosophy of Steady's DustV5 (2003).**

DustPort reconstructs the four classic Dust modes (FaeryDust/PixieDust/GoldDust/SpaceDust) with modern extensions: larger temporal radius, median option, external motion compensation, edge-aware limit map, 10–16 bit planar YUV support, and per-plane processing.

The defining property is **limit-clipping**: every output pixel is guaranteed to stay within a bounded distance from the source — `out = clamp(filter, src ± limit)`. This makes DustPort well-suited for film restoration where drift, hallucinated detail, and over-smoothing must be avoided.

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Filter names and modes](#filter-names-and-modes)
- [Parameters](#parameters)
- [Usage examples](#usage-examples)
- [Architecture and philosophy](#architecture-and-philosophy)
- [Relation to the original DustV5](#relation-to-the-original-dustv5)
- [Performance tips](#performance-tips)
- [Build from source](#build-from-source)
- [Credits and license](#credits-and-license)

---

## Installation

1. Download `DustPort.dll` from Releases.
2. Put it in your AviSynth+ plugins64 directory (usually `C:\Program Files (x86)\AviSynth+\plugins64`).
3. Optional: put `DustPort.avsi` in the same directory for convenience wrappers.

Requirements: AviSynth+ 3.4 or newer, 64-bit build, x86-64 CPU with SSE4.1 support.

## Quick start

Simplest usage, spatial only, no motion compensation:

```
LoadPlugin("DustPort.dll")
src = LWLibavVideoSource("clip.mkv")
src.SpaceDustPort(limit=6)
```

Full spatiotemporal with motion compensation (classic Pixie strength):

```
LoadPlugin("DustPort.dll")
src = LWLibavVideoSource("clip.mkv").ConvertBits(16)

# Build MC clips with MVTools
sup = src.MSuper(pel=2, sharp=2, hpad=16, vpad=16)
bv1 = sup.MAnalyse(isb=true,  delta=1, blksize=8, overlap=4, search=5, truemotion=true)
fv1 = sup.MAnalyse(isb=false, delta=1, blksize=8, overlap=4, search=5, truemotion=true)
p1  = src.MCompensate(sup, bv1, thSAD=300)
n1  = src.MCompensate(sup, fv1, thSAD=300)

src.PixieDustPort(limit=5, prev=[p1], next=[n1])
```

## Filter names and modes

Four filter names preserve the classic Dust naming. Each corresponds to an internal `mode` with a different default strength:

| Filter name      | Mode | Default limit | Behaviour                               |
|------------------|:----:|:-------------:|-----------------------------------------|
| `FaeryDustPort`  |  0   |       2       | Temporal only (conservative)            |
| `PixieDustPort`  |  1   |       5       | Temporal + spatial, balanced            |
| `GoldDustPort`   |  2   |       8       | Temporal + spatial, spatial-dominant    |
| `SpaceDustPort`  |  3   |       6       | Spatial only (fastest, no frames fetched) |

All four share the same parameter set (except `SpaceDustPort` which omits temporal parameters).

## Parameters

### Common parameters

| Parameter     | Type    | Default | Description                                                         |
|---------------|---------|:-------:|---------------------------------------------------------------------|
| `limit`       | int     | varies  | Maximum per-pixel deviation from source. Range: 1..64 (8-bit scale, auto-scaled internally for 16-bit clips) |
| `mode`        | int     | varies  | Override the default mode (0..3). Normally not needed              |
| `limit_edge`  | int     |   -1    | If >=0, enables edge-aware limit map. Limit on edge pixels         |
| `edge_thresh` | int     |   32    | Sobel gradient magnitude threshold above which a pixel is an edge   |
| `planes`      | string  | "yuv"   | Which planes to process. Any of `y`, `u`, `v`, `yu`, `yv`, `uv`, `yuv` |

### Temporal parameters (Faery/Pixie/Gold only, not Space)

| Parameter   | Type                | Default | Description                                              |
|-------------|---------------------|:-------:|----------------------------------------------------------|
| `tr`        | int                 |    1    | Temporal radius. Window size = 2×tr+1 frames. Range: 1..5 |
| `temporal`  | int                 |    0    | 0 = mean, 1 = median                                     |
| `prev`      | array of clips      |   nil   | Motion-compensated previous frames. Size must equal `tr` |
| `next`      | array of clips      |   nil   | Motion-compensated next frames. Size must equal `tr`     |

If `prev`/`next` are omitted, the plugin uses un-compensated `clip[n±k]` with boundary clamping. This is usually worse than external MC but requires no MVTools setup.

## Usage examples

### Edge-aware, aggressive on flat areas

```
src.PixieDustPort( \
    limit=16, limit_edge=3, edge_thresh=24, \
    tr=1, temporal=0, prev=[p1], next=[n1])
```

`limit=16` gives strong denoising on flat areas, `limit_edge=3` protects contours, `edge_thresh=24` defines where "edge" starts.

### Temporal radius = 2 with median

```
src.PixieDustPort( \
    limit=8, tr=2, temporal=1, \
    prev=[p1,p2], next=[n1,n2])
```

Median over a 5-frame window (prev-2, prev-1, cur, next-1, next-2). Better at rejecting temporal outliers than mean.

### Luma only (fastest useful configuration)

```
src.SpaceDustPort(limit=6, planes="y")
```

Spatial-only on luma, chroma passes through. Useful when chroma is clean and Y has shot noise.

### Multiple passes

```
src.PixieDustPort(limit=4, tr=1, prev=[p1], next=[n1]).\
    PixieDustPort(limit=4, tr=1, prev=[p1], next=[n1])
```

Two gentle passes often give better character than one aggressive pass (classic Fizick advice, works here too).

## Architecture and philosophy

**What makes DustPort different from modern denoisers (BM3D, KNLMeansCL, DFTTest):**

- **Bounded output.** Every pixel is guaranteed within `src ± limit`. No drift. No hallucinated detail. What goes out is always "close to" what went in, by a measurable amount.
- **Honest MC.** The plugin does no motion estimation itself. External motion-compensated clips are supplied by the user via `prev`/`next` arrays. This is visible, tunable, reproducible. MVTools is the usual source.
- **Simple kernels.** Temporal mean or median over an MC window. Spatial 3×3 weighted blend. Edge-aware per-pixel limit. Each component does one thing that can be reasoned about.
- **No quality-restoration claim.** DustPort cleans noise. It does not sharpen, enhance, invent detail, or de-ring edges. Those are separate problems with separate tools.

**The trade-off.** Modern denoisers often produce cleaner-looking output on clean sources. But on restoration material with compression artifacts, dead pixels, or scan noise, they can produce plausible-but-wrong output that is indistinguishable from real detail in the target frame. Limit-clipping architectures cannot do this — by construction.

## Relation to the original DustV5

Original `DustV5.dll` by Steady (2003) was a 57 KB 32-bit MMX filter for AviSynth 2.0.x, handling YUY2 only. The source code was never published.

DustPort is an **independent reimplementation** based on:
- Reverse-engineering of the 2003 binary via `pefile` + `capstone` (mode dispatch, limit defaults, class structure — 12 specialized MMX kernels)
- The public ReadMe by Steady
- Published forum discussion on doom9 (thread ID 80834 and surrounding)

Confirmed findings:
- Faery/Pixie/Gold/Space are one class with a mode selector (0/1/2/3), not four separate filters
- The undocumented `mode` parameter in the original signature overrides the default per filter name
- SpaceDust (mode=3) is spatial-only; the other three share a spatiotemporal path with `tr=1`
- Limit defaults: Faery=2, Pixie=5, Gold=8, Space=6

DustPort **adds**:
- `tr` up to 5, `temporal` (mean/median) choice
- External motion compensation via `prev`/`next` arrays
- Edge-aware per-pixel limit map (Sobel-based)
- Planar YUV support (YV12/YV16/YV24), bit depths 8–16
- Per-plane processing (`planes` parameter)
- SSE2/SSE4.1 implementation (MMX retired)

DustPort is **not bit-exact** with the original. It reproduces the same philosophy.

## Performance tips

Ordered by usefulness:

1. **`planes="y"`** — if your noise is in luma only, skip chroma processing. Typically 1.4–2.5× speedup depending on subsampling.
2. **`SpaceDustPort`** — if you don't need temporal filtering, use spatial-only. Fetches only one frame per output, no MC setup required.
3. **`tr=1, temporal=0`** — the default. `tr=2+` and `temporal=1` cost more.
4. **Disable edge-aware** (`limit_edge=-1`, the default) when not needed. Saves one Sobel pass per plane.
5. **MVTools setup matters more than DustPort settings.** MC usually dominates runtime. Consider lower `pel`, larger `blksize`, `search=4` (hexagon) instead of 5 (exhaustive).

## Build from source

Requires MSVC (not MinGW — AviSynth+ has vtable layout incompatibility with GCC).

```
cl /LD /EHsc /O2 /arch:AVX2 ^
   /I "C:\Program Files (x86)\AviSynth+\FilterSDK\include" ^
   DustPort.cpp /link /OUT:DustPort.dll
```

`/arch:AVX2` is required — it enables SSE4.1 intrinsics used for 16-bit min/max operations.

A `build.bat` is provided for convenience.

## Credits and license

Original DustV5 © 2003 by Steady. DustPort is an independent implementation.

DustPort © 2026 Yuri Pronkin.
Released under the MIT License. See `LICENSE`.

Reverse-engineering made possible by the `pefile` and `capstone` projects.

Feedback and patches welcome via GitHub issues or the doom9 thread.
