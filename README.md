# Dust

**A limit-clipping spatiotemporal denoiser in the philosophy of Steady's DustV5 (2003).** One plugin binary for
**AviSynth+** and **VapourSynth** (API 4), built for Windows x64, Linux x86_64 and macOS arm64, with SSE4.1, AVX,
AVX2 and NEON code paths. All of them give exactly the same output as the plain C code, so the result does not
depend on the CPU. The output differs from DustPort 1.0.0 (see
[Differences to DustPort 1.0.0](#differences-to-dustport-100)).

Every output sample stays within a fixed distance of the source sample: `out = clamp(filter, src ± limit)`. Dust
cannot drift, invent detail or smear structure by more than `limit`, which makes it a safe denoiser for film
restoration, scans and compressed sources.

- DustV5 by Steady (2003): FaeryDust, PixieDust, GoldDust, SpaceDust; 32-bit, MMX, YUY2 only, source never published
- DustPort 1.0.0 by Yuri Pronkin (2026): independent reimplementation for AviSynth+ (64 bit, planar YUV, 10-16 bit,
  larger temporal radius, median, external motion compensation, edge-aware limits)
- Dust 1.0.0: AviSynth+ and VapourSynth from one source, exact formulas, scene change handling, mirrored borders,
  field mode for interlaced video, Y/Gray and YUVA, multithreading, SSE4.1/AVX/AVX2/NEON, Linux and macOS builds

How it works, formula by formula: [ALGORITHM.md](ALGORITHM.md).

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Functions and modes](#functions-and-modes)
- [Parameters](#parameters)
- [Formats](#formats)
- [Wrapper: DustDenoise](#wrapper-dustdenoise)
- [Usage examples](#usage-examples)
- [Performance](#performance)
- [Differences to DustPort 1.0.0](#differences-to-dustport-100)
- [Build from source](#build-from-source)
- [Credits and license](#credits-and-license)

## Installation

Release files: `Dust-<version>-<platform>.zip` (plugin, `Dust.avsi`, `Dust.py`, docs) and a VapourSynth wheel.

| | AviSynth+ | VapourSynth |
|---|---|---|
| Plugin | `Dust.dll` into the `plugins64` folder, or `LoadPlugin("Dust.dll")` | `pip install vapoursynth_dust-<version>-py3-none-<platform>.whl` (installs into `site-packages/vapoursynth/plugins`), or the plugin file into a plugin folder / `core.std.LoadPlugin()` |
| File | `Dust.dll` (Windows) | `Dust.dll`, `libDust.so`, `libDust.dylib` |
| Wrapper | `Dust.avsi` (needs MVTools2) | `Dust.py` on the Python path (needs MVTools, namespace `mv`) |
| Needs | AviSynth+; `Dust.avsi` needs arrays and `ArrayAdd` (3.7.3+, tested with 3.7.6); frame properties from interface version 8 | API 4 (R55 or newer) |

The best instruction set is chosen at runtime; any x86-64 CPU works (C fallback without SSE4.1).

## Quick start

AviSynth+:

```
LoadPlugin("Dust.dll")
src = LWLibavVideoSource("clip.mkv")
src.SpaceDust(limit=6)                      # spatial only, no other frames
src.PixieDust()                             # temporal + spatial, plain neighbour frames
```

With motion compensation (`Dust.avsi` builds the MVTools clips):

```
LoadPlugin("Dust.dll")
LoadPlugin("mvtools2.dll")
Import("Dust.avsi")
LWLibavVideoSource("clip.mkv").ConvertBits(16).DustDenoise(tr=2, temporal=1)
```

VapourSynth:

```python
import vapoursynth as vs
core = vs.core
clip = core.dust.PixieDust(clip, tr=2)

from Dust import DustDenoise                 # Dust.py, motion compensated
clip = DustDenoise(clip, limit=16, limit_edge=3, edge_thresh=24, tr=2, temporal=1, planes=[0])
```

## Functions and modes

The four names of DustV5 are one filter with different defaults:

| AviSynth+ | VapourSynth | `mode` | `limit` | Behaviour |
|---|---|:--:|:--:|---|
| `FaeryDust` | `dust.FaeryDust` | 0 | 2 | temporal only (conservative) |
| `PixieDust` | `dust.PixieDust` | 1 | 5 | temporal + spatial, temporal-weighted |
| `GoldDust` | `dust.GoldDust` | 2 | 8 | temporal + spatial, spatial-weighted |
| `SpaceDust` | `dust.SpaceDust` | 3 | 6 | spatial only (fastest, fetches no other frame) |

`mode` overrides the default of the function. SpaceDust has no temporal parameters; with `mode < 3` it uses `tr=1`
without motion compensation.

## Parameters

All levels (`limit`, `limit_edge`, `edge_thresh`, `scthresh`) are on the 8-bit scale at every bit depth.

| Parameter | Type | Default | Range | Meaning |
|---|---|:--:|---|---|
| `limit` | int | per function | 1..64 | maximum change of a sample |
| `mode` | int | per function | 0..3 | see above |
| `limit_edge` | int | -1 | -1, 0..`limit` | limit on edges; -1 = edge-aware limit off |
| `edge_thresh` | int | 32 | 1..255 | Sobel magnitude counted as a full edge (a sharp step of height h gives 4h) |
| `planes` | AviSynth+: string, VapourSynth: int list | all | `"y"`, `"uv"`, … / `[0]`, `[1, 2]`, … | planes to process, the others are copied |
| `scthresh` | int | 20 | -1, 0, 1..255 | scene change threshold (mean absolute luma difference); 0 = only frame properties, -1 = no scene change handling |
| `interlaced` | AviSynth+: bool, VapourSynth: int | from `_FieldBased` | | field mode: spatial neighbours from the same field |
| `opt` | int | -1 | -1..4 | instruction set: -1 best, 0 C, 1 SSE4.1, 2 AVX, 3 AVX2, 4 NEON; error if not available |

Temporal parameters (FaeryDust, PixieDust, GoldDust):

| Parameter | Type | Default | Range | Meaning |
|---|---|:--:|---|---|
| `tr` | int | 1 | 1..5 | temporal radius, window of `2*tr+1` frames |
| `temporal` | int | 0 | 0, 1 | 0 = mean, 1 = median |
| `prev` | clip array | none | `tr` clips | `prev[k-1]` = frame `n-k` motion compensated onto `n` (MVTools `isb=false`, `delta=k`) |
| `next` | clip array | none | `tr` clips | `next[k-1]` = frame `n+k` motion compensated onto `n` (MVTools `isb=true`, `delta=k`) |

Without `prev`/`next` the plain neighbour frames are used. The motion compensated clips must have the format and
size of the input and at least as many frames.

**Scene changes:** a window position behind a cut (or beyond the clip end) is replaced by the nearest frame of the
same scene, so frames of different scenes are never averaged. Cuts come from `_SceneChangePrev` /
`_SceneChangeNext` (e.g. VapourSynth `misc.SCDetect`, `mv.SCDetection`) and from the built-in detector
(`scthresh`).

**Borders** are mirrored, so every sample, including the first and last row and column, gets the same filter.

**Interlaced video:** with `interlaced=true` (or `_FieldBased` = 1/2 when `interlaced` is not given) the spatial
blend and the edge detector work per field; the temporal part already combines the same field of the neighbour
frames. Motion compensated clips for interlaced video must be made per field (`DustDenoise(interlaced=true)` does
this).

## Formats

| | AviSynth+ | VapourSynth |
|---|---|---|
| Supported | Y, planar YUV and YUVA (alpha copied), 8-16 bit; YUY2 | Gray, YUV, 8-16 bit integer |
| Not supported | RGB, 32 bit float | RGB, float |

YUY2 gives exactly the result of the same clip as planar 4:2:2. Frame properties are kept. Thread-safe and
independent of request order (AviSynth+ `MT_NICE_FILTER`, VapourSynth `fmParallel`). Float support is planned for
1.1.0.

## Wrapper: DustDenoise

`Dust.avsi` (AviSynth+, MVTools2 by pinterf) and `Dust.py` (VapourSynth, MVTools `core.mv`) build the motion
compensated clips and call the filter:

```
DustDenoise(clip, mode="pixie", limit, limit_edge=-1, edge_thresh=32, tr=1, temporal=0, planes, scthresh=20,
            interlaced=false, mc=true, thSAD=300, blksize=8, overlap=4, search=4, pel=2, truemotion=true, opt=-1)
```

- `mode`: `"faery"`, `"pixie"`, `"gold"` or `"space"` (no motion compensation); `limit` defaults per mode.
- `mc=false`: no motion compensation, the filter uses the plain neighbour frames.
- `interlaced=true`: field mode; motion compensation per field (`SeparateFields`, `delta = 2k`, `Weave`). Set the
  field order first (AviSynth+ `AssumeTFF()` / `AssumeBFF()`, VapourSynth `_FieldBased`).
- MVTools settings: `thSAD` (`thsad` in Python), `blksize`, `overlap`, `search`, `pel`, `truemotion`; chroma is
  used for the motion search when `planes` includes U or V.
- `planes`: string in AviSynth+, list of indices in Python.

The wrapper rebuilds the motion vectors on every call; for several calls on the same clip build `prev`/`next` once
and call the filter directly.

## Usage examples

Edge-aware: strong on flat areas, gentle on contours:

```
src.PixieDust(limit=16, limit_edge=3, edge_thresh=24, prev=[p1], next=[n1])
```

Median over five motion compensated frames (rejects dust specks and dead pixels):

```
src.PixieDust(limit=8, tr=2, temporal=1, prev=[p1, p2], next=[n1, n2])
```

Luma only:

```
src.SpaceDust(limit=6, planes="y")
```

```python
core.dust.SpaceDust(clip, limit=6, planes=[0])
```

Two gentle passes often look better than one strong pass:

```
src.PixieDust(limit=4, prev=[p1], next=[n1]).PixieDust(limit=4, prev=[p1], next=[n1])
```

Better scene detection than the built-in one (VapourSynth):

```python
clip = core.misc.SCDetect(clip, threshold=0.1)
clip = core.dust.PixieDust(clip, tr=3, scthresh=0)
```

## Performance

1920x1080 4:2:0, one thread, frames per second (VapourSynth R82, synthetic noise, AMD Ryzen 9 7950X):

| Call | C | SSE4.1 | AVX | AVX2 |
|---|---:|---:|---:|---:|
| 8 bit `PixieDust()` | 46 | 706 | 765 | 1259 |
| 8 bit `PixieDust(tr=2, temporal=1)` | 13 | 337 | 341 | 527 |
| 8 bit `FaeryDust(tr=5, temporal=1)` | 5 | 123 | 120 | 187 |
| 8 bit `SpaceDust()` | 86 | 1301 | 1383 | 2274 |
| 8 bit `PixieDust(limit=16, limit_edge=3, edge_thresh=24)` | 27 | 356 | 376 | 589 |
| 16 bit `PixieDust()` | 39 | 329 | 328 | 452 |
| 16 bit `GoldDust(tr=3)` | 38 | 201 | 199 | 271 |
| 16 bit `SpaceDust()` | 79 | 773 | 755 | 663 |

Tips, by effect: `planes="y"` / `planes=[0]` if only luma is noisy; SpaceDust if no temporal filtering is needed;
mean and small `tr` are cheaper than median and large `tr`; the edge-aware limit costs one Sobel pass per plane; with
motion compensation, MVTools usually takes more time than Dust (larger `blksize`, `search=4`, lower `pel` are
faster). Measure with `test/bench.py`.

## Differences to DustPort 1.0.0

The output is **not** identical to DustPort 1.0.0; Dust computes the formulas that DustPort documented, everywhere:

- Names: `FaeryDust` / `PixieDust` / `GoldDust` / `SpaceDust` (the DustV5 names) instead of `…DustPort`; the
  wrapper file is `Dust.avsi`.
- DustPort's SIMD and scalar paths computed different formulas, so a result depended on the column (the first and
  the last 1-15 columns of each row differed). Dust uses `(4c + up + down + left + right + 4) >> 3` for the
  spatial blend and `(2t + s + 1) / 3`, `(t + 2s + 1) / 3` for PixieDust / GoldDust everywhere.
- DustPort's temporal mean rounded down too far (three equal frames of value 100 gave 99), so static areas got
  darker by one code value per pass; Dust rounds to nearest.
- DustPort's median for 9 and 11 frames (`tr=4`, `tr=5`) was not a median; Dust uses verified networks.
- The edge-aware limit was rounded differently per column; Dust rounds to nearest everywhere.
- New: scene change handling (`scthresh`, frame properties), mirrored borders (DustPort copied the first and last row
  and column in SpaceDust and used only the temporal part there in the other modes), field mode, Y/Gray and
  YUVA, edge-aware and `planes` for YUY2, VapourSynth, `opt`.
- DustPort was built for AVX2 as a whole (illegal instruction on older CPUs) and ran single-threaded
  (`MT_SERIALIZED`) and dropped frame properties; Dust selects the instruction set at runtime, runs multithreaded
  and keeps the properties.
- `DustPort.avsi` took `prev` from the `isb=true` vectors and `next` from `isb=false` (swapped; no effect on mean and
  median, but wrong); `Dust.avsi` uses the right ones.

`test/compare_original.py` reports the differences per setting: FaeryDust with median is identical, FaeryDust with
mean differs by at most 1, all other settings stay within `2 * limit` of DustPort's output.

## Build from source

Meson and a C++17 compiler (MSVC, GCC or Clang):

```
meson setup build            # Windows: meson setup build --vsenv -Db_vscrt=mt   (or run build.bat)
meson compile -C build
meson test -C build          # every SIMD tier against the C reference
```

Wheel: `pip wheel .` (Windows: add `--config-settings=setup-args=--vsenv --config-settings=setup-args=-Db_vscrt=mt`).

Tests in `test/` (Python with numpy):

| Script | Checks |
|---|---|
| `test_dust.py [plugin]` | VapourSynth: bit for bit against the numpy model `model.py` (written from ALGORITHM.md), every case for every instruction set, errors |
| `test_avs.py` | AviSynth+ (Windows, `avs2yuv64`, BestSource): the same cases, plus YUY2 and YUVA |
| `kernel_test.cpp` | `meson test`: every SIMD tier against C on random data |
| `compare_hosts.py` | AviSynth+ and VapourSynth byte-identical on real clips |
| `test_wrappers.py` | `Dust.avsi` and `Dust.py` on a real clip |
| `compare_original.py` | differences to the DustPort 1.0.0 DLL |
| `bench.py` | speed per instruction set |

## Credits and license

- DustV5 © 2003 Steady: the original filter and its four modes.
- DustPort © 2026 Yuri Pronkin: the reimplementation this port is based on (reverse-engineered with `pefile` and
  `capstone`, see the doom9 thread 80834 and the DustV5 ReadMe).
- Dust © 2026 Selur: AviSynth+ / VapourSynth port.

Released under the MIT License, see [LICENSE](LICENSE). The vendored VapourSynth headers in `include/vapoursynth`
are LGPL-2.1, see the license file there.
