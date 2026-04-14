# Changelog

All notable changes to DustPort are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.0.0] — 2026-04-14

Initial public release.

### Features

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

### Known limitations

- Not bit-exact with the original DustV5 DLL
- Edge-aware mode not supported on YUY2
- RGB formats not supported
- Single-threaded per instance (AviSynth+ serializes the filter)

### Platform

- Windows x86-64 only
- AviSynth+ 3.4 or newer
- Requires MSVC for building (MinGW has vtable incompatibility)
