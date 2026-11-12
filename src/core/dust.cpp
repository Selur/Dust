// SPDX-License-Identifier: MIT
/*
  Dust - limit-clipping spatiotemporal denoiser in the philosophy of Steady's DustV5 (2003)
  DustPort 1.0.0 (AviSynth+): (c) 2026 Yuri Pronkin
  Dust (AviSynth+ and VapourSynth, exact formulas, scene changes, field mode, SIMD): (c) 2026 Selur
  MIT License, see LICENSE.
*/
#include "dust.hpp"

#include <cstring>
#include <stdexcept>

namespace dust {
namespace {

[[noreturn]] void fail(const std::string& msg) { throw std::runtime_error("Dust: " + msg); }

// One component of a frame: a plane, or one component of the packed YUY2 plane.
struct Comp {
  ptrdiff_t offset = 0;  // bytes from data[plane]
  int plane = 0;
  int width = 0, height = 0;
  int xstep = 1;
};

}  // namespace

void parsePlanes(const char* s, bool planes[3]) {
  bool y = false, u = false, v = false;
  for (const char* c = s ? s : "yuv"; *c; ++c) {
    char ch = *c;
    if (ch >= 'A' && ch <= 'Z') ch = char(ch + 32);
    if (ch == 'y')
      y = true;
    else if (ch == 'u')
      u = true;
    else if (ch == 'v')
      v = true;
    else if (ch != ' ' && ch != '\t')
      fail(std::string("planes string '") + s + "' contains invalid characters (only y, u, v allowed)");
  }
  if (!y && !u && !v) fail(std::string("planes string '") + (s ? s : "") + "' selects no planes");
  planes[0] = y;
  planes[1] = u;
  planes[2] = v;
}

Engine::Engine(const Params& p, const Format& f, int num_frames) : p_(p), fmt_(f), num_frames_(num_frames) {
  if (f.bits < 8 || f.bits > 16) fail("unsupported format: use Y/Gray, planar YUV or YUY2 with 8-16 bit integer samples");
  if (f.layout == Layout::YUY2 && f.bits != 8) fail("unsupported bit depth");
  if (p.mode < 0 || p.mode > 3) fail("mode must be 0..3");
  if (p.limit < 1 || p.limit > 64) fail("limit must be 1..64 (8-bit scale)");
  if (p.tr < 1 || p.tr > 5) fail("tr must be 1..5");
  if (p.temporal != 0 && p.temporal != 1) fail("temporal must be 0 (mean) or 1 (median)");
  if (p.limit_edge < -1 || p.limit_edge > 64) fail("limit_edge must be 0..64, or -1 to disable");
  if (p.limit_edge > p.limit)
    fail("limit_edge (" + std::to_string(p.limit_edge) + ") must be <= limit (" + std::to_string(p.limit) + ")");
  if (p.edge_thresh < 1 || p.edge_thresh > 255) fail("edge_thresh must be 1..255");
  if (p.scthresh < -1 || p.scthresh > 255) fail("scthresh must be -1 (off), 0 (frame properties only) or 1..255");
  if (p.interlaced < -1 || p.interlaced > 1) fail("interlaced must be a boolean");
  if (p.num_prev != 0 && p.num_prev != p.tr)
    fail("prev array size (" + std::to_string(p.num_prev) + ") must match tr (" + std::to_string(p.tr) + ")");
  if (p.num_next != 0 && p.num_next != p.tr)
    fail("next array size (" + std::to_string(p.num_next) + ") must match tr (" + std::to_string(p.tr) + ")");
  if (!f.chroma && f.layout == Layout::Planar && !p.planes[0])
    fail("planes selects no plane of this format (Y/Gray has only y)");
  if (!selectIsa(p.opt, &isa_)) fail("opt=" + std::to_string(p.opt) + ": instruction set not available");
  k_ = &kernelsFor(isa_);
  kc_ = &kernelsC();
  temporal_ = p.mode != 3;
}

void Engine::sourceRange(int n, int* first, int* last) const {
  const bool need = temporal_ && (p_.num_prev == 0 || p_.num_next == 0 || p_.scthresh >= 0);
  const int r = need ? p_.tr : 0;
  *first = n - r < 0 ? 0 : n - r;
  *last = n + r >= num_frames_ ? num_frames_ - 1 : n + r;
}

bool Engine::cutBetween(int f, int first, const std::vector<FrameView>& src) const {
  if (p_.scthresh < 0) return false;
  const FrameView& a = src[size_t(f - first)];
  const FrameView& b = src[size_t(f + 1 - first)];
  if (b.scene_prev == 1 || a.scene_next == 1) return true;
  if (p_.scthresh == 0) return false;
  uint64_t sum = 0, count = 0;
  const int xstep = fmt_.layout == Layout::YUY2 ? 2 : 1;
  const Kernels& k = xstep == 1 ? *k_ : *kc_;
  k.scene_sad(a.data[0], a.stride[0], b.data[0], b.stride[0], fmt_.width, fmt_.height, xstep, fmt_.bits, &sum, &count);
  return count > 0 && sum / count > uint64_t(p_.scthresh);
}

void Engine::process(int n, int first, const std::vector<FrameView>& src, const std::vector<FrameView>& prev,
                     const std::vector<FrameView>& next, const FrameOut& dst) const {
  const FrameView& centre = src[size_t(n - first)];
  const int bps = fmt_.bits == 8 ? 1 : 2;

  // Window: offset of the frame used at each position -tr..tr (0 = centre), see ALGORITHM.md stage 1.
  const int tr = temporal_ ? p_.tr : 0;
  int offs[11];
  offs[tr] = 0;
  for (int side = -1; side <= 1; side += 2) {
    int valid = 0;
    bool blocked = false;
    for (int k = 1; k <= tr; k++) {
      const int f = n + side * k;
      if (!blocked) {
        if (f < 0 || f >= num_frames_)
          blocked = true;
        else if (cutBetween(side < 0 ? f : f - 1, first, src))
          blocked = true;
        else
          valid = k;
      }
      offs[tr + side * k] = side * valid;
    }
  }
  const int count = 2 * tr + 1;
  const FrameView* win[11];
  for (int i = 0; i < count; i++) {
    const int o = offs[i];
    if (o == 0)
      win[i] = &centre;
    else if (o < 0)
      win[i] = p_.num_prev ? &prev[size_t(-o - 1)] : &src[size_t(n + o - first)];
    else
      win[i] = p_.num_next ? &next[size_t(o - 1)] : &src[size_t(n + o - first)];
  }

  const bool fields = p_.interlaced == 1 || (p_.interlaced == -1 && centre.field_based > 0);

  // Components.
  Comp comps[3];
  int ncomp = 0;
  if (fmt_.layout == Layout::YUY2) {
    comps[0] = {0, 0, fmt_.width, fmt_.height, 2};
    comps[1] = {1, 0, fmt_.width / 2, fmt_.height, 4};
    comps[2] = {3, 0, fmt_.width / 2, fmt_.height, 4};
    ncomp = 3;
    if (!(p_.planes[0] && p_.planes[1] && p_.planes[2]))
      for (int y = 0; y < fmt_.height; y++)
        std::memcpy(dst.data[0] + y * dst.stride[0], centre.data[0] + y * centre.stride[0], size_t(fmt_.width) * 2);
  } else {
    comps[0] = {0, 0, fmt_.width, fmt_.height, 1};
    ncomp = 1;
    if (fmt_.chroma) {
      comps[1] = {0, 1, fmt_.width >> fmt_.ssw, fmt_.height >> fmt_.ssh, 1};
      comps[2] = {0, 2, fmt_.width >> fmt_.ssw, fmt_.height >> fmt_.ssh, 1};
      ncomp = 3;
    }
  }
  const Kernels& k = fmt_.layout == Layout::YUY2 ? *kc_ : *k_;

  std::vector<uint8_t> map;
  for (int c = 0; c < ncomp; c++) {
    const Comp& cp = comps[c];
    if (!p_.planes[c]) {
      if (fmt_.layout == Layout::Planar && dst.data[cp.plane])  // null: the host shares the source plane
        for (int y = 0; y < cp.height; y++)
          std::memcpy(dst.data[cp.plane] + y * dst.stride[cp.plane], centre.data[cp.plane] + y * centre.stride[cp.plane],
                      size_t(cp.width) * bps);
      continue;
    }
    for (int field = 0; field < (fields ? 2 : 1); field++) {
      const int mul = fields ? 2 : 1;
      const int h = fields ? (cp.height - field + 1) / 2 : cp.height;
      if (h <= 0 || cp.width <= 0) continue;

      const uint8_t* wp[11];
      ptrdiff_t ws[11];
      for (int i = 0; i < count; i++) {
        wp[i] = win[i]->data[cp.plane] + cp.offset + field * win[i]->stride[cp.plane];
        ws[i] = win[i]->stride[cp.plane] * mul;
      }

      FilterArgs a;
      a.win = wp;
      a.win_stride = ws;
      a.count = count;
      a.centre = tr;
      a.dst = dst.data[cp.plane] + cp.offset + field * dst.stride[cp.plane];
      a.dst_stride = dst.stride[cp.plane] * mul;
      a.width = cp.width;
      a.height = h;
      a.xstep = cp.xstep;
      a.bits = fmt_.bits;
      a.mode = p_.mode;
      a.temporal = p_.temporal;
      a.limit = p_.limit;

      if (p_.limit_edge >= 0) {
        map.resize(size_t(cp.width) * size_t(h));
        LimitMapArgs m;
        m.src = wp[tr];
        m.stride = ws[tr];
        m.width = cp.width;
        m.height = h;
        m.xstep = cp.xstep;
        m.bits = fmt_.bits;
        m.limit = p_.limit;
        m.limit_edge = p_.limit_edge;
        m.edge_thresh = p_.edge_thresh;
        m.map = map.data();
        k.limit_map(m);
        a.map = map.data();
      }
      k.filter(a);
    }
  }
}

}  // namespace dust
