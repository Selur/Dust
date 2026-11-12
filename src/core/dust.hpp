// SPDX-License-Identifier: MIT
// Host-independent Dust engine: parameter checks, window assembly with clip ends and scene changes, field mode,
// per-plane dispatch to the kernels. The hosts (AviSynth+, VapourSynth) only fetch frames and read properties.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "kernels.hpp"

namespace dust {

enum class Layout { Planar, YUY2 };

struct Format {
  Layout layout = Layout::Planar;
  int width = 0, height = 0;  // luma samples
  int bits = 8;               // 8..16
  bool chroma = false;        // planar U and V present (YUV); false for Gray / Y. YUY2 always has chroma.
  int ssw = 0, ssh = 0;       // chroma subsampling (log2), planar only
};

struct Params {
  int mode = 1;
  int limit = 5;
  int tr = 1;
  int temporal = 0;
  int limit_edge = -1;
  int edge_thresh = 32;
  int scthresh = 20;
  int interlaced = -1;  // -1 = from _FieldBased, 0 = progressive, 1 = field mode
  int opt = -1;
  bool planes[3] = {true, true, true};
  int num_prev = 0, num_next = 0;  // motion compensated clips (0 or tr each)
};

// Read-only view of one frame. Planar: Y, U, V (alpha is the host's job). YUY2: the packed plane in [0].
// Properties: -1 = not present.
struct FrameView {
  const uint8_t* data[3] = {nullptr, nullptr, nullptr};
  ptrdiff_t stride[3] = {0, 0, 0};
  int scene_prev = -1;
  int scene_next = -1;
  int field_based = -1;
};

struct FrameOut {
  uint8_t* data[3] = {nullptr, nullptr, nullptr};
  ptrdiff_t stride[3] = {0, 0, 0};
};

// Parses an AviSynth style planes string ("y", "uv", "yuv", ...). Throws std::runtime_error.
void parsePlanes(const char* s, bool planes[3]);

class Engine {
 public:
  Engine(const Params& p, const Format& f, int num_frames);  // throws std::runtime_error with "Dust: ..." texts

  // Source frames (of the input clip) that process(n) needs: first..last. Mode 3 needs only n.
  void sourceRange(int n, int* first, int* last) const;
  // Whether process(n) needs the prev/next frames at n (motion compensated clips given and a temporal mode).
  bool usesMc() const { return temporal_ && (p_.num_prev > 0 || p_.num_next > 0); }

  // src: frames first..last from sourceRange(n); prev/next: prev[k-1] and next[k-1] at frame n (only with usesMc()).
  void process(int n, int first, const std::vector<FrameView>& src, const std::vector<FrameView>& prev,
               const std::vector<FrameView>& next, const FrameOut& dst) const;

  Isa isa() const { return isa_; }

 private:
  bool cutBetween(int f, int first, const std::vector<FrameView>& src) const;  // scene change between f and f+1

  Params p_;
  Format fmt_;
  int num_frames_;
  bool temporal_;
  Isa isa_;
  const Kernels* k_;
  const Kernels* kc_;
};

}  // namespace dust
