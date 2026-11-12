// SPDX-License-Identifier: MIT
// Pixel kernels of Dust. Every instruction set gives bit-identical results to the C reference (see ALGORITHM.md).
#pragma once

#include <cstddef>
#include <cstdint>

namespace dust {

// One plane (or one field of a plane, or one component of YUY2). Sample (x, y) lives at
// data + y * stride + x * xstep * bytes_per_sample. Strides are in bytes, xstep in samples.
struct FilterArgs {
  const uint8_t* const* win = nullptr;  // window planes, count entries, centre at index `centre`
  const ptrdiff_t* win_stride = nullptr;
  int count = 1;                        // 1 for mode 3 (spatial only)
  int centre = 0;
  uint8_t* dst = nullptr;
  ptrdiff_t dst_stride = 0;
  int width = 0, height = 0;
  int xstep = 1;
  int bits = 8;                         // 8..16
  int mode = 1;                         // 0..3
  int temporal = 0;                     // 0 mean, 1 median
  int limit = 5;                        // 8-bit scale
  const uint8_t* map = nullptr;         // per-pixel limit (8-bit scale), width * height, or null
};

struct LimitMapArgs {
  const uint8_t* src = nullptr;
  ptrdiff_t stride = 0;
  int width = 0, height = 0;
  int xstep = 1;
  int bits = 8;
  int limit = 5, limit_edge = 0, edge_thresh = 32;
  uint8_t* map = nullptr;               // width * height
};

struct Kernels {
  void (*filter)(const FilterArgs& a);
  void (*limit_map)(const LimitMapArgs& a);
  // Scene change metric: sum of |(a >> shift) - (b >> shift)| over every second sample of every second row, and the
  // number of compared samples.
  void (*scene_sad)(const uint8_t* a, ptrdiff_t sa, const uint8_t* b, ptrdiff_t sb, int width, int height, int xstep,
                    int bits, uint64_t* sum, uint64_t* count);
};

enum class Isa { C, SSE41, AVX, AVX2, NEON };

const Kernels& kernelsC();
#if defined(_M_X64) || defined(__x86_64__)
#define DUST_X86 1
const Kernels& kernelsSSE41();
const Kernels& kernelsAVX();
const Kernels& kernelsAVX2();
#elif defined(__aarch64__) || defined(_M_ARM64)
#define DUST_NEON 1
const Kernels& kernelsNEON();
#endif

// opt: -1 = best available, 0 = C, 1 = SSE4.1, 2 = AVX, 3 = AVX2, 4 = NEON. Returns false if not available.
bool selectIsa(int opt, Isa* isa);
const Kernels& kernelsFor(Isa isa);
const char* isaName(Isa isa);

}  // namespace dust
