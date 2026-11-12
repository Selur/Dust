// SPDX-License-Identifier: MIT
// Generic SIMD kernels of Dust, written once against a small vector interface S (see kernels_sse41.cpp,
// kernels_avx2.cpp, kernels_neon.cpp). Include inside an anonymous namespace after defining S.
//
// Bit-exact to the C reference: every intermediate value is an integer kept in lanes wide enough for it (8 bit
// input: 16-bit lanes, 9-16 bit input: 32-bit lanes where sums exceed 16 bits), divisions by n and 3 use
// multiply-high constants verified for every possible input (scalar.hpp), the median uses verified networks
// (median_nets.hpp). The first and last column of each row run the scalar formula (mirrored neighbours); the
// interior runs in vectors, the last vector overlapping the previous one instead of a scalar tail.
//
// Interface S (V8/V16/V32: vectors of N8 bytes, N16 = N8/2 uint16, N32 = N8/4 uint32; widening and packing may
// reorder lanes as long as pack16(lo8(v), hi8(v)) == v and pack32(lo16(v), hi16(v)) == v):
//   load8/store8, load16/store16 (N16 uint16), lo8/hi8 (V8 -> V16), pack16 (V16, V16 -> V8, unsigned saturation),
//   lo16/hi16 (V16 -> V32), pack32 (V32, V32 -> V16, unsigned saturation), loadHalf8 (N16 bytes -> V16, in
//   order), storeHalf8 (V16 -> N16 bytes, in order, saturating), min8/max8, min16/max16/add16/sub16/adds16/subs16/
//   mullo16/set16/shl16/shr16/mulhi16 ((x * m) >> 16), add32/set32/shl32/shr32/mulshr32 ((x * m) >> (32 + s)),
//   and16/asU16 (V8 -> V16)/sum16to32 (pairwise)/min32/max32/sub32/and32/asU32 (V16 -> V32)/hsum32.

static inline typename S::V16 div16(typename S::V16 x, const Magic16& m) {
  if (m.identity) return x;
  typename S::V16 q = S::mulhi16(x, m.m);
  return m.s ? S::shr16(q, m.s) : q;
}

// Interior columns 1..width-2 in steps of n, the last step overlapping; false if the interior is narrower than n.
template <typename F>
static inline bool forInterior(int width, int n, F&& step) {
  if (width - 2 < n) return false;
  int x = 1;
  for (; x + n <= width - 1; x += n) step(x);
  if (x < width - 1) step(width - 1 - n);
  return true;
}

static void filter8(const FilterArgs& a) {
  using V8 = typename S::V8;
  using V16 = typename S::V16;
  const DivTables& dt = divTables();
  const Magic16 mm = dt.mean8[a.count], m3 = dt.third8;
  const MedianNet net = medianNet(a.count);
  const int n = a.count;
  const bool temporal = a.mode != 3, spatial = a.mode != 0;
  const V16 half = S::set16(n / 2), one = S::set16(1), four = S::set16(4), vmax = S::set16(255),
            vlim = S::set16(a.limit);

  for (int y = 0; y < a.height; y++) {
    const Rows r = rowsAt(a, y);
    filterPixel<uint8_t>(a, r, 0);
    if (a.width > 1) filterPixel<uint8_t>(a, r, a.width - 1);
    const bool done = forInterior(a.width, S::N8, [&](int x) {
      const V8 c8 = S::load8(r.c1 + x);
      const V16 cl = S::lo8(c8), ch = S::hi8(c8);
      V16 tl = cl, th = ch, sl = cl, sh = ch;
      if (temporal) {
        if (a.temporal == 1) {
          V8 v[11];
          for (int i = 0; i < n; i++) v[i] = S::load8(r.win[i] + x);
          for (int k = 0; k < net.count; k++) {
            const int p = net.pairs[k][0], q = net.pairs[k][1];
            const V8 lo = S::min8(v[p], v[q]);
            v[q] = S::max8(v[p], v[q]);
            v[p] = lo;
          }
          tl = S::lo8(v[n / 2]);
          th = S::hi8(v[n / 2]);
        } else {
          V16 suml = half, sumh = half;
          for (int i = 0; i < n; i++) {
            const V8 w = S::load8(r.win[i] + x);
            suml = S::add16(suml, S::lo8(w));
            sumh = S::add16(sumh, S::hi8(w));
          }
          tl = div16(suml, mm);
          th = div16(sumh, mm);
        }
      }
      if (spatial) {
        const V8 up = S::load8(r.c0 + x), dn = S::load8(r.c2 + x), l = S::load8(r.c1 + x - 1),
                 rr = S::load8(r.c1 + x + 1);
        V16 ql = S::add16(S::add16(S::lo8(up), S::lo8(dn)), S::add16(S::lo8(l), S::lo8(rr)));
        V16 qh = S::add16(S::add16(S::hi8(up), S::hi8(dn)), S::add16(S::hi8(l), S::hi8(rr)));
        sl = S::shr16(S::add16(S::add16(S::shl16(cl, 2), ql), four), 3);
        sh = S::shr16(S::add16(S::add16(S::shl16(ch, 2), qh), four), 3);
      }
      V16 fl, fh;
      switch (a.mode) {
        case 0: fl = tl; fh = th; break;
        case 1:
          fl = div16(S::add16(S::add16(S::add16(tl, tl), sl), one), m3);
          fh = div16(S::add16(S::add16(S::add16(th, th), sh), one), m3);
          break;
        case 2:
          fl = div16(S::add16(S::add16(S::add16(sl, sl), tl), one), m3);
          fh = div16(S::add16(S::add16(S::add16(sh, sh), th), one), m3);
          break;
        default: fl = sl; fh = sh; break;
      }
      V16 liml = vlim, limh = vlim;
      if (r.map) {
        const V8 m = S::load8(r.map + x);
        liml = S::lo8(m);
        limh = S::hi8(m);
      }
      fl = S::min16(S::max16(fl, S::subs16(cl, liml)), S::min16(S::add16(cl, liml), vmax));
      fh = S::min16(S::max16(fh, S::subs16(ch, limh)), S::min16(S::add16(ch, limh), vmax));
      S::store8(r.d + x, S::pack16(fl, fh));
    });
    if (!done)
      for (int x = 1; x < a.width - 1; x++) filterPixel<uint8_t>(a, r, x);
  }
}

static void filter16(const FilterArgs& a) {
  using V16 = typename S::V16;
  using V32 = typename S::V32;
  const DivTables& dt = divTables();
  const Magic32 mm = dt.mean16[a.count], m3 = dt.third16;
  const MedianNet net = medianNet(a.count);
  const int n = a.count;
  const int shift = a.bits - 8;
  const bool temporal = a.mode != 3, spatial = a.mode != 0;
  const V32 half = S::set32(n / 2), one = S::set32(1), four = S::set32(4);
  const V16 vmax = S::set16((1 << a.bits) - 1), vlim = S::set16(a.limit << shift);
  auto p16 = [](const uint8_t* row, int x) { return row + ptrdiff_t(x) * 2; };

  for (int y = 0; y < a.height; y++) {
    const Rows r = rowsAt(a, y);
    filterPixel<uint16_t>(a, r, 0);
    if (a.width > 1) filterPixel<uint16_t>(a, r, a.width - 1);
    const bool done = forInterior(a.width, S::N16, [&](int x) {
      const V16 c = S::load16(p16(r.c1, x));
      V16 t = c;
      V32 tl = S::lo16(c), th = S::hi16(c), sl = tl, sh = th;
      if (temporal) {
        if (a.temporal == 1) {
          V16 v[11];
          for (int i = 0; i < n; i++) v[i] = S::load16(p16(r.win[i], x));
          for (int k = 0; k < net.count; k++) {
            const int p = net.pairs[k][0], q = net.pairs[k][1];
            const V16 lo = S::min16(v[p], v[q]);
            v[q] = S::max16(v[p], v[q]);
            v[p] = lo;
          }
          t = v[n / 2];
          tl = S::lo16(t);
          th = S::hi16(t);
        } else {
          V32 suml = half, sumh = half;
          for (int i = 0; i < n; i++) {
            const V16 w = S::load16(p16(r.win[i], x));
            suml = S::add32(suml, S::lo16(w));
            sumh = S::add32(sumh, S::hi16(w));
          }
          tl = S::mulshr32(suml, mm);
          th = S::mulshr32(sumh, mm);
          t = S::pack32(tl, th);
        }
      }
      V16 s = c;
      if (spatial) {
        const V16 up = S::load16(p16(r.c0, x)), dn = S::load16(p16(r.c2, x)), l = S::load16(p16(r.c1, x - 1)),
                  rr = S::load16(p16(r.c1, x + 1));
        V32 ql = S::add32(S::add32(S::lo16(up), S::lo16(dn)), S::add32(S::lo16(l), S::lo16(rr)));
        V32 qh = S::add32(S::add32(S::hi16(up), S::hi16(dn)), S::add32(S::hi16(l), S::hi16(rr)));
        sl = S::shr32(S::add32(S::add32(S::shl32(S::lo16(c), 2), ql), four), 3);
        sh = S::shr32(S::add32(S::add32(S::shl32(S::hi16(c), 2), qh), four), 3);
        s = S::pack32(sl, sh);
      }
      V16 f;
      switch (a.mode) {
        case 0: f = t; break;
        case 1:
          f = S::pack32(S::mulshr32(S::add32(S::add32(S::add32(tl, tl), sl), one), m3),
                        S::mulshr32(S::add32(S::add32(S::add32(th, th), sh), one), m3));
          break;
        case 2:
          f = S::pack32(S::mulshr32(S::add32(S::add32(S::add32(sl, sl), tl), one), m3),
                        S::mulshr32(S::add32(S::add32(S::add32(sh, sh), th), one), m3));
          break;
        default: f = s; break;
      }
      const V16 lim = r.map ? S::shl16(S::loadHalf8(r.map + x), shift) : vlim;
      f = S::min16(S::max16(f, S::subs16(c, lim)), S::min16(S::adds16(c, lim), vmax));
      S::store16(r.d + ptrdiff_t(x) * 2, f);
    });
    if (!done)
      for (int x = 1; x < a.width - 1; x++) filterPixel<uint16_t>(a, r, x);
  }
}

static void filterSimd(const FilterArgs& a) {
  if (a.xstep != 1) return kernelsC().filter(a);
  if (a.bits == 8)
    filter8(a);
  else
    filter16(a);
}

// |pos - neg| for non-negative 16-bit values.
static inline typename S::V16 absdiff16(typename S::V16 p, typename S::V16 q) {
  return S::sub16(S::max16(p, q), S::min16(p, q));
}

static inline typename S::V16 limitVec(typename S::V16 pa, typename S::V16 pb, typename S::V16 pc,
                                       typename S::V16 pd, typename S::V16 pf, typename S::V16 pg,
                                       typename S::V16 ph, typename S::V16 pi, typename S::V16 vth,
                                       typename S::V16 vdiff, typename S::V16 vround, typename S::V16 vlimit,
                                       const Magic16& m) {
  const auto sx = absdiff16(S::add16(S::add16(pc, S::shl16(pf, 1)), pi), S::add16(S::add16(pa, S::shl16(pd, 1)), pg));
  const auto sy = absdiff16(S::add16(S::add16(pg, S::shl16(ph, 1)), pi), S::add16(S::add16(pa, S::shl16(pb, 1)), pc));
  const auto grad = S::min16(S::max16(sx, sy), vth);
  return S::sub16(vlimit, div16(S::add16(S::mullo16(grad, vdiff), vround), m));
}

static void limitMapSimd(const LimitMapArgs& a) {
  using V8 = typename S::V8;
  using V16 = typename S::V16;
  const Magic16 m = findMagic16(a.edge_thresh, a.edge_thresh * (a.limit - a.limit_edge) + a.edge_thresh / 2);
  if (a.xstep != 1 || !m.ok) return kernelsC().limit_map(a);
  const V16 vth = S::set16(a.edge_thresh), vdiff = S::set16(a.limit - a.limit_edge),
            vround = S::set16(a.edge_thresh / 2), vlimit = S::set16(a.limit);
  const int shift = a.bits - 8;
  for (int y = 0; y < a.height; y++) {
    const uint8_t* r0 = a.src + mirror(y - 1, a.height) * a.stride;
    const uint8_t* r1 = a.src + y * a.stride;
    const uint8_t* r2 = a.src + mirror(y + 1, a.height) * a.stride;
    uint8_t* map = a.map + ptrdiff_t(y) * a.width;
    bool done;
    if (a.bits == 8) {
      map[0] = limitPixel<uint8_t>(a, r0, r1, r2, 0);
      if (a.width > 1) map[a.width - 1] = limitPixel<uint8_t>(a, r0, r1, r2, a.width - 1);
      done = forInterior(a.width, S::N8, [&](int x) {
        const V8 va = S::load8(r0 + x - 1), vb = S::load8(r0 + x), vc = S::load8(r0 + x + 1);
        const V8 vd = S::load8(r1 + x - 1), vf = S::load8(r1 + x + 1);
        const V8 vg = S::load8(r2 + x - 1), vh = S::load8(r2 + x), vi = S::load8(r2 + x + 1);
        const V16 lo = limitVec(S::lo8(va), S::lo8(vb), S::lo8(vc), S::lo8(vd), S::lo8(vf), S::lo8(vg), S::lo8(vh),
                                S::lo8(vi), vth, vdiff, vround, vlimit, m);
        const V16 hi = limitVec(S::hi8(va), S::hi8(vb), S::hi8(vc), S::hi8(vd), S::hi8(vf), S::hi8(vg), S::hi8(vh),
                                S::hi8(vi), vth, vdiff, vround, vlimit, m);
        S::store8(map + x, S::pack16(lo, hi));
      });
      if (!done)
        for (int x = 1; x < a.width - 1; x++) map[x] = limitPixel<uint8_t>(a, r0, r1, r2, x);
    } else {
      map[0] = limitPixel<uint16_t>(a, r0, r1, r2, 0);
      if (a.width > 1) map[a.width - 1] = limitPixel<uint16_t>(a, r0, r1, r2, a.width - 1);
      auto ld = [shift](const uint8_t* row, int x) { return S::shr16(S::load16(row + ptrdiff_t(x) * 2), shift); };
      done = forInterior(a.width, S::N16, [&](int x) {
        S::storeHalf8(map + x, limitVec(ld(r0, x - 1), ld(r0, x), ld(r0, x + 1), ld(r1, x - 1), ld(r1, x + 1),
                                        ld(r2, x - 1), ld(r2, x), ld(r2, x + 1), vth, vdiff, vround, vlimit, m));
      });
      if (!done)
        for (int x = 1; x < a.width - 1; x++) map[x] = limitPixel<uint16_t>(a, r0, r1, r2, x);
    }
  }
}

static void sceneSadSimd(const uint8_t* pa, ptrdiff_t sa, const uint8_t* pb, ptrdiff_t sb, int width, int height,
                         int xstep, int bits, uint64_t* sum, uint64_t* count) {
  if (xstep != 1) return kernelsC().scene_sad(pa, sa, pb, sb, width, height, xstep, bits, sum, count);
  using V16 = typename S::V16;
  using V32 = typename S::V32;
  const int shift = bits - 8;
  uint64_t total = 0;
  for (int y = 0; y < height; y += 2) {
    const uint8_t* ra = pa + y * sa;
    const uint8_t* rb = pb + y * sb;
    V32 acc = S::set32(0);
    int x = 0;
    if (bits == 8) {
      const V16 mask = S::set16(0xFF);
      for (; x + S::N8 <= width; x += S::N8) {
        const V16 va = S::and16(S::asU16(S::load8(ra + x)), mask);
        const V16 vb = S::and16(S::asU16(S::load8(rb + x)), mask);
        acc = S::add32(acc, S::sum16to32(absdiff16(va, vb)));
      }
    } else {
      const V32 mask = S::set32(0xFFFF);
      for (; x + S::N16 <= width; x += S::N16) {
        const V32 va = S::and32(S::asU32(S::shr16(S::load16(ra + ptrdiff_t(x) * 2), shift)), mask);
        const V32 vb = S::and32(S::asU32(S::shr16(S::load16(rb + ptrdiff_t(x) * 2), shift)), mask);
        acc = S::add32(acc, S::sub32(S::max32(va, vb), S::min32(va, vb)));
      }
    }
    total += S::hsum32(acc);
    for (; x < width; x += 2) {
      const int va = bits == 8 ? ra[x] : reinterpret_cast<const uint16_t*>(ra)[x] >> shift;
      const int vb = bits == 8 ? rb[x] : reinterpret_cast<const uint16_t*>(rb)[x] >> shift;
      total += uint64_t(va > vb ? va - vb : vb - va);
    }
  }
  *sum = total;
  *count = uint64_t((width + 1) / 2) * uint64_t((height + 1) / 2);
}

static const Kernels& simdKernels() {
  static const Kernels k = {filterSimd, limitMapSimd, sceneSadSimd};
  return k;
}
