// SPDX-License-Identifier: MIT
// Compares every available SIMD tier with the C reference kernels on random data: all widths 1..96, heights 1..5,
// 8/10/12/14/16 bit, window sizes 1..11, every mode, mean and median, with and without a limit map, random strides,
// uniform, extreme (0 / max only) and smooth content; limit maps for every edge_thresh; scene SAD sums.
// Run: meson test -C build (or build/kernel_test). Exit code 0 = all tiers identical to C.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "core/kernels.hpp"

using namespace dust;

namespace {

std::mt19937 rng(12345);

int rnd(int lo, int hi) { return std::uniform_int_distribution<int>(lo, hi)(rng); }

// A plane with random stride padding, filled according to `kind` (0 random, 1 extremes, 2 smooth, 3 flat).
struct Plane {
  std::vector<uint8_t> buf;
  ptrdiff_t stride = 0;
  uint8_t* data() { return buf.data() + 32; }
  const uint8_t* data() const { return buf.data() + 32; }
};

Plane makePlane(int w, int h, int bits, int kind) {
  Plane p;
  const int bps = bits == 8 ? 1 : 2;
  p.stride = ptrdiff_t(w + rnd(0, 40)) * bps;
  p.buf.assign(size_t(p.stride) * h + 64 + 64, 0);
  const int maxv = (1 << bits) - 1;
  const int base = rnd(0, maxv);
  for (int y = 0; y < h; y++)
    for (int x = 0; x < w; x++) {
      int v;
      switch (kind) {
        case 1: v = rnd(0, 1) ? maxv : 0; break;
        case 2: v = std::min(maxv, std::max(0, base + (x + y) * rnd(0, 3) * (1 << (bits - 8)) + rnd(-3, 3))); break;
        case 3: v = base; break;
        default: v = rnd(0, maxv); break;
      }
      uint8_t* row = p.data() + y * p.stride;
      if (bps == 1)
        row[x] = uint8_t(v);
      else
        reinterpret_cast<uint16_t*>(row)[x] = uint16_t(v);
    }
  return p;
}

int failures = 0, checks = 0;

void fail(const char* isa, const char* what, int w, int h, int bits, const char* extra) {
  if (failures < 30) std::printf("FAIL %s %s w=%d h=%d bits=%d %s\n", isa, what, w, h, bits, extra);
  failures++;
}

void testFilter(Isa isa, const Kernels& k) {
  const Kernels& c = kernelsC();
  for (int w = 1; w <= 96; w++) {
    for (int iter = 0; iter < 6; iter++) {
      const int h = rnd(1, 5);
      const int bits = std::vector<int>{8, 10, 12, 14, 16}[rnd(0, 4)];
      const int bps = bits == 8 ? 1 : 2;
      const int mode = rnd(0, 3);
      const int count = mode == 3 ? 1 : 2 * rnd(1, 5) + 1;
      const int kind = rnd(0, 3);
      std::vector<Plane> win;
      std::vector<const uint8_t*> ptr;
      std::vector<ptrdiff_t> st;
      for (int i = 0; i < count; i++) {
        win.push_back(makePlane(w, h, bits, i == 0 || rnd(0, 3) ? kind : 0));
      }
      for (auto& p : win) {
        ptr.push_back(p.data());
        st.push_back(p.stride);
      }
      std::vector<uint8_t> map;
      const bool use_map = rnd(0, 1);
      if (use_map) {
        map.resize(size_t(w) * h);
        for (auto& m : map) m = uint8_t(rnd(0, 64));
      }
      FilterArgs a;
      a.win = ptr.data();
      a.win_stride = st.data();
      a.count = count;
      a.centre = count / 2;
      a.width = w;
      a.height = h;
      a.bits = bits;
      a.mode = mode;
      a.temporal = rnd(0, 1);
      a.limit = rnd(0, 3) ? rnd(1, 64) : (rnd(0, 1) ? 1 : 64);
      a.map = use_map ? map.data() : nullptr;
      Plane d1 = makePlane(w, h, bits, 0), d2 = d1;
      a.dst = d1.data();
      a.dst_stride = d1.stride;
      c.filter(a);
      a.dst = d2.data();
      k.filter(a);
      checks++;
      if (d1.buf != d2.buf) {
        char extra[128];
        std::snprintf(extra, sizeof extra, "mode=%d count=%d temporal=%d limit=%d map=%d kind=%d", mode, count,
                      a.temporal, a.limit, int(use_map), kind);
        fail(isaName(isa), "filter", w, h, bits, extra);
      }
      (void)bps;
    }
  }
}

void testLimitMap(Isa isa, const Kernels& k) {
  const Kernels& c = kernelsC();
  for (int th = 1; th <= 255; th++) {
    const int w = rnd(1, 96), h = rnd(1, 5);
    const int bits = std::vector<int>{8, 10, 16}[rnd(0, 2)];
    Plane src = makePlane(w, h, bits, rnd(0, 2));
    LimitMapArgs a;
    a.src = src.data();
    a.stride = src.stride;
    a.width = w;
    a.height = h;
    a.bits = bits;
    a.limit = rnd(0, 2) ? rnd(1, 64) : 64;
    a.limit_edge = rnd(0, a.limit);
    a.edge_thresh = th;
    std::vector<uint8_t> m1(size_t(w) * h), m2(size_t(w) * h);
    a.map = m1.data();
    c.limit_map(a);
    a.map = m2.data();
    k.limit_map(a);
    checks++;
    if (m1 != m2) {
      char extra[96];
      std::snprintf(extra, sizeof extra, "th=%d limit=%d limit_edge=%d", th, a.limit, a.limit_edge);
      fail(isaName(isa), "limit_map", w, h, bits, extra);
    }
  }
}

void testSceneSad(Isa isa, const Kernels& k) {
  const Kernels& c = kernelsC();
  for (int w = 1; w <= 96; w++) {
    for (int bits : {8, 10, 16}) {
      const int h = rnd(1, 6);
      Plane a = makePlane(w, h, bits, rnd(0, 2)), b = makePlane(w, h, bits, rnd(0, 2));
      uint64_t s1, n1, s2, n2;
      c.scene_sad(a.data(), a.stride, b.data(), b.stride, w, h, 1, bits, &s1, &n1);
      k.scene_sad(a.data(), a.stride, b.data(), b.stride, w, h, 1, bits, &s2, &n2);
      checks++;
      if (s1 != s2 || n1 != n2) fail(isaName(isa), "scene_sad", w, h, bits, "");
    }
  }
}

}  // namespace

int main() {
  int tested = 0;
  for (int opt = 1; opt <= 4; opt++) {
    Isa isa;
    if (!selectIsa(opt, &isa)) continue;
    const Kernels& k = kernelsFor(isa);
    testFilter(isa, k);
    testLimitMap(isa, k);
    testSceneSad(isa, k);
    std::printf("%s: tested\n", isaName(isa));
    tested++;
  }
  std::printf("%d instruction sets, %d checks, %d failures\n", tested, checks, failures);
  return failures ? 1 : 0;
}
