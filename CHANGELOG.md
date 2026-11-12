# Changelog

All notable changes to Dust are documented here. Dust started as a port of DustPort 1.0.0 (Yuri Pronkin), whose
entry is kept below.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.0.0] — 2026-11-12

First release of Dust: one plugin for AviSynth+ and VapourSynth. Not bit-exact with DustPort 1.0.0 (see README,
"Differences to DustPort 1.0.0").

### Added

- VapourSynth API 4 interface: `core.dust.FaeryDust`, `PixieDust`, `GoldDust`, `SpaceDust`
  (plugin id `io.github.selur.dust`), `planes` as a list of plane indices
- Scene change handling: built-in detector (`scthresh`) and the frame properties `_SceneChangePrev` /
  `_SceneChangeNext`; frames of another scene never enter the temporal window
- Field mode for interlaced video (`interlaced`, or `_FieldBased` when not given)
- Formats: Y/Gray, YUVA (alpha copied); edge-aware mode and `planes` for YUY2
- SSE4.1, AVX, AVX2 (x86-64) and NEON (arm64) code paths, selected at runtime, bit-identical to the C reference;
  `opt` parameter to force one
- Builds for Windows x64, Linux x86_64 and macOS arm64; VapourSynth wheel
- `Dust.avsi` and `Dust.py`: `DustDenoise()` with `mode`, `scthresh`, `interlaced` (motion compensation per field),
  `mc`, `search`, `pel`, `opt`

### Changed

- Functions renamed to the DustV5 names: `FaeryDust`, `PixieDust`, `GoldDust`, `SpaceDust` (were `…DustPort`);
  `DustPort.avsi` is now `Dust.avsi`
- The documented formulas everywhere: spatial blend `(4c + up + down + left + right + 4) >> 3`, PixieDust
  `(2t + s + 1) / 3`, GoldDust `(t + 2s + 1) / 3` (DustPort computed other weights in its SIMD columns)
- Borders mirrored: every sample, including the first and last row and column, gets the same filter
- Multithreaded (`MT_NICE_FILTER`, `fmParallel`); frame properties are kept
- `ALGORITHM.md` rewritten as the exact specification

### Fixed

- Temporal mean rounded down too far (static areas got one code value darker per pass)
- Median of 9 and 11 frames (`tr=4`, `tr=5`) was wrong in the SIMD paths
- Edge-aware limit rounded differently depending on the column
- The DLL required AVX2 although only SSE4.1 was documented
- `DustPort.avsi` used swapped motion vectors for `prev` and `next`

## DustPort 1.0.0 — 2026-04-14 (origin of Dust, by Yuri Pronkin)

Initial public release of DustPort.

#### Features

- Four filter names with classic Dust mode mapping:
  - `FaeryDustPort` (mode 0, default limit 2, temporal only)
  - `PixieDustPort` (mode 1, default limit 5, temporal + spatial)
  - `GoldDustPort`  (mode 2, default limit 8, spatial-dominant)
  - `SpaceDustPort` (mode 3, default limit 6, spatial only)
- Limit-clipping architecture: `out = clamp(filter, src ± limit)` guarantee
- Temporal radius `tr` 1 to 5
- Temporal aggregation choice: mean or median (SSE2/SSE4.1 sorting networks)
- External motion compensation via `prev`/`next` clip arrays
- Edge-aware per-pixel limit map (Sobel 3×3 gradient, linear blend)
- Pixel format support:
  - YUY2 (8-bit packed)
  - YV12, YV16, YV24 (8-bit planar)
  - YUV420P10..P16, YUV422P10..P16, YUV444P10..P16
- Per-plane processing via `planes` parameter
- SSE2 for 8-bit paths, SSE4.1 for 16-bit min/max

#### Known limitations

- Not bit-exact with the original DustV5 DLL
- Edge-aware mode not supported on YUY2
- RGB formats not supported
- Single-threaded per instance (AviSynth+ serializes the filter)

#### Platform

- Windows x86-64 only
- AviSynth+ 3.4 or newer
- Requires MSVC for building (MinGW has vtable incompatibility)
