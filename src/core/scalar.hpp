// SPDX-License-Identifier: MIT
// Scalar formulas of ALGORITHM.md shared by the C reference and the border columns of the SIMD kernels, and the
// exact division constants of the SIMD kernels. Everything has internal linkage (static), because the header is
// compiled into translation units with different instruction set flags: a shared inline copy could be AVX code.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include "kernels.hpp"

namespace dust {

// Mirrored neighbour index without repeating the edge sample (-1 -> 1, n -> n - 2); a single sample mirrors to
// itself.
static inline int mirror(int i, int n) {
  if (i < 0) return n > 1 ? 1 : 0;
  if (i >= n) return n > 1 ? n - 2 : n - 1;
  return i;
}

template <typename T>
static inline int loadSample(const uint8_t* row, int x, int xstep) {
  return reinterpret_cast<const T*>(row)[ptrdiff_t(x) * xstep];
}

template <typename T>
static inline void storeSample(uint8_t* row, int x, int xstep, int v) {
  reinterpret_cast<T*>(row)[ptrdiff_t(x) * xstep] = T(v);
}

static inline int medianOf(int* v, int n) {
  for (int i = 1; i < n; i++) {
    const int x = v[i];
    int j = i - 1;
    while (j >= 0 && v[j] > x) {
      v[j + 1] = v[j];
      j--;
    }
    v[j + 1] = x;
  }
  return v[n / 2];
}

// Rows of the centre plane around row y (mirrored) and of every window plane at row y.
struct Rows {
  const uint8_t* c0;
  const uint8_t* c1;
  const uint8_t* c2;
  const uint8_t* win[11];
  uint8_t* d;
  const uint8_t* map;
};

static inline Rows rowsAt(const FilterArgs& a, int y) {
  Rows r;
  const uint8_t* centre = a.win[a.centre];
  const ptrdiff_t cs = a.win_stride[a.centre];
  r.c1 = centre + y * cs;
  r.c0 = centre + mirror(y - 1, a.height) * cs;
  r.c2 = centre + mirror(y + 1, a.height) * cs;
  for (int i = 0; i < a.count; i++) r.win[i] = a.win[i] + y * a.win_stride[i];
  r.d = a.dst + y * a.dst_stride;
  r.map = a.map ? a.map + ptrdiff_t(y) * a.width : nullptr;
  return r;
}

// One output sample (ALGORITHM.md stages 2-5).
template <typename T>
static inline void filterPixel(const FilterArgs& a, const Rows& r, int x) {
  const int shift = a.bits - 8;
  const int maxv = (1 << a.bits) - 1;
  const int c = loadSample<T>(r.c1, x, a.xstep);
  int t = 0;
  if (a.mode != 3) {
    int vals[11];
    for (int i = 0; i < a.count; i++) vals[i] = loadSample<T>(r.win[i], x, a.xstep);
    if (a.temporal == 1) {
      t = medianOf(vals, a.count);
    } else {
      int sum = 0;
      for (int i = 0; i < a.count; i++) sum += vals[i];
      t = (sum + a.count / 2) / a.count;
    }
  }
  int s = 0;
  if (a.mode != 0) {
    const int up = loadSample<T>(r.c0, x, a.xstep);
    const int dn = loadSample<T>(r.c2, x, a.xstep);
    const int l = loadSample<T>(r.c1, mirror(x - 1, a.width), a.xstep);
    const int rr = loadSample<T>(r.c1, mirror(x + 1, a.width), a.xstep);
    s = (4 * c + up + dn + l + rr + 4) >> 3;
  }
  int f;
  switch (a.mode) {
    case 0: f = t; break;
    case 1: f = (2 * t + s + 1) / 3; break;
    case 2: f = (t + 2 * s + 1) / 3; break;
    default: f = s; break;
  }
  const int lim = (r.map ? r.map[x] : a.limit) << shift;
  int lo = c - lim;
  if (lo < 0) lo = 0;
  int hi = c + lim;
  if (hi > maxv) hi = maxv;
  storeSample<T>(r.d, x, a.xstep, f < lo ? lo : (f > hi ? hi : f));
}

// One limit map value (ALGORITHM.md stage 0).
template <typename T>
static inline uint8_t limitPixel(const LimitMapArgs& a, const uint8_t* r0, const uint8_t* r1, const uint8_t* r2,
                                  int x) {
  const int shift = a.bits - 8;
  const int xl = mirror(x - 1, a.width), xr = mirror(x + 1, a.width);
  const int pa = loadSample<T>(r0, xl, a.xstep) >> shift, pb = loadSample<T>(r0, x, a.xstep) >> shift,
            pc = loadSample<T>(r0, xr, a.xstep) >> shift;
  const int pd = loadSample<T>(r1, xl, a.xstep) >> shift, pf = loadSample<T>(r1, xr, a.xstep) >> shift;
  const int pg = loadSample<T>(r2, xl, a.xstep) >> shift, ph = loadSample<T>(r2, x, a.xstep) >> shift,
            pi = loadSample<T>(r2, xr, a.xstep) >> shift;
  const int sx = std::abs((pc + 2 * pf + pi) - (pa + 2 * pd + pg));
  const int sy = std::abs((pg + 2 * ph + pi) - (pa + 2 * pb + pc));
  int grad = sx > sy ? sx : sy;
  if (grad > a.edge_thresh) grad = a.edge_thresh;
  return uint8_t(a.limit - (grad * (a.limit - a.limit_edge) + a.edge_thresh / 2) / a.edge_thresh);
}

// floor(x / d) == (x * m) >> (16 + s) for every 0 <= x <= xmax (m < 2^16), or identity for d == 1.
struct Magic16 {
  uint16_t m = 0;
  int s = 0;
  bool identity = false;
  bool ok = false;
};

static inline Magic16 findMagic16(int d, int xmax) {
  Magic16 r;
  if (d == 1) {
    r.identity = r.ok = true;
    return r;
  }
  for (int s = 0; s < 16; s++) {
    const uint32_t m = uint32_t(((uint64_t(1) << (16 + s)) + d - 1) / d);
    if (m > 0xFFFF) break;
    bool good = true;
    for (int x = 0; x <= xmax && good; x++) good = int((uint32_t(x) * m) >> (16 + s)) == x / d;
    if (good) {
      r.m = uint16_t(m);
      r.s = s;
      r.ok = true;
      return r;
    }
  }
  return r;
}

// floor(x / d) == (x * m) >> (32 + s) for every 0 <= x <= xmax (m < 2^32).
struct Magic32 {
  uint32_t m = 0;
  int s = 0;
  bool ok = false;
};

static inline Magic32 findMagic32(int d, int xmax) {
  Magic32 r;
  for (int s = 0; s < 32; s++) {
    const uint64_t m = ((uint64_t(1) << (32 + s)) + d - 1) / d;
    if (m > 0xFFFFFFFFull) break;
    bool good = true;
    for (int x = 0; x <= xmax && good; x++) good = int((uint64_t(x) * m) >> (32 + s)) == x / d;
    if (good) {
      r.m = uint32_t(m);
      r.s = s;
      r.ok = true;
      return r;
    }
  }
  return r;
}

// Division constants of the SIMD filter kernels, verified for every possible input at first use.
struct DivTables {
  Magic16 mean8[12];  // (sum + n/2) / n, 8 bit: sum <= 11 * 255 + 5
  Magic16 third8;     // (2a + b + 1) / 3, 8 bit
  Magic32 mean16[12]; // 9-16 bit: sum <= 11 * 65535 + 5
  Magic32 third16;
};

static inline const DivTables& divTables() {
  static const DivTables t = [] {
    DivTables d;
    for (int n = 3; n <= 11; n += 2) {
      d.mean8[n] = findMagic16(n, n * 255 + n / 2);
      d.mean16[n] = findMagic32(n, n * 65535 + n / 2);
    }
    d.third8 = findMagic16(3, 3 * 255 + 1);
    d.third16 = findMagic32(3, 3 * 65535 + 1);
    return d;
  }();
  return t;
}

}  // namespace dust
