// SPDX-License-Identifier: MIT
// AviSynth+ interface: FaeryDust, PixieDust, GoldDust, SpaceDust. Y, planar YUV(A) 8-16 bit, YUY2.
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "avisynth.h"
#include "core/dust.hpp"
#include "version.hpp"

using namespace dust;

namespace {

int propInt(IScriptEnvironment* env, const AVSMap* props, const char* key) {
  int err = 0;
  const int64_t v = env->propGetInt(props, key, 0, &err);
  return err ? -1 : int(v);
}

class DustAvs : public GenericVideoFilter {
 public:
  DustAvs(PClip child, const Params& p, const Format& fmt, std::vector<PClip> prev, std::vector<PClip> next,
          IScriptEnvironment* env)
      : GenericVideoFilter(child), fmt_(fmt), prev_(std::move(prev)), next_(std::move(next)) {
    try {
      env->CheckVersion(8);
      has_v8_ = true;
    } catch (const AvisynthError&) {
      has_v8_ = false;
    }
    try {
      engine_ = std::make_unique<Engine>(p, fmt_, vi.num_frames);
    } catch (const std::exception& e) {
      env->ThrowError("%s", e.what());
    }
    auto check = [&](const PClip& c, const char* tag, int i) {
      const VideoInfo& m = c->GetVideoInfo();
      if (m.width != vi.width || m.height != vi.height) env->ThrowError("Dust: %s[%d] size mismatch", tag, i);
      if (m.pixel_type != vi.pixel_type) env->ThrowError("Dust: %s[%d] pixel_type mismatch", tag, i);
      if (m.num_frames < vi.num_frames) env->ThrowError("Dust: %s[%d] shorter than the main clip", tag, i);
    };
    for (int i = 0; i < int(prev_.size()); i++) check(prev_[i], "prev", i);
    for (int i = 0; i < int(next_.size()); i++) check(next_[i], "next", i);
  }

  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) override {
    n = n < 0 ? 0 : (n >= vi.num_frames ? vi.num_frames - 1 : n);
    int first = n, last = n;
    engine_->sourceRange(n, &first, &last);

    std::vector<PVideoFrame> hold;
    hold.reserve(size_t(last - first + 1) + prev_.size() + next_.size());
    std::vector<FrameView> src, prev, next;
    PVideoFrame centre;
    for (int f = first; f <= last; f++) {
      hold.push_back(child->GetFrame(f, env));
      if (f == n) centre = hold.back();
      src.push_back(view(hold.back(), env));
    }
    if (engine_->usesMc()) {
      for (auto& c : prev_) {
        hold.push_back(c->GetFrame(n, env));
        prev.push_back(view(hold.back(), env));
      }
      for (auto& c : next_) {
        hold.push_back(c->GetFrame(n, env));
        next.push_back(view(hold.back(), env));
      }
    }

    PVideoFrame dst = has_v8_ ? env->NewVideoFrameP(vi, &centre) : env->NewVideoFrame(vi);
    FrameOut out;
    if (fmt_.layout == Layout::YUY2) {
      out.data[0] = dst->GetWritePtr();
      out.stride[0] = dst->GetPitch();
    } else {
      static const int planes[] = {PLANAR_Y, PLANAR_U, PLANAR_V};
      for (int i = 0; i < (fmt_.chroma ? 3 : 1); i++) {
        out.data[i] = dst->GetWritePtr(planes[i]);
        out.stride[i] = dst->GetPitch(planes[i]);
      }
      if (vi.IsYUVA())
        env->BitBlt(dst->GetWritePtr(PLANAR_A), dst->GetPitch(PLANAR_A), centre->GetReadPtr(PLANAR_A),
                    centre->GetPitch(PLANAR_A), centre->GetRowSize(PLANAR_A), centre->GetHeight(PLANAR_A));
    }
    try {
      engine_->process(n, first, src, prev, next, out);
    } catch (const std::exception& e) {
      env->ThrowError("%s", e.what());
    }
    return dst;
  }

  int __stdcall SetCacheHints(int cachehints, int frame_range) override {
    (void)frame_range;
    return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
  }

 private:
  FrameView view(const PVideoFrame& f, IScriptEnvironment* env) const {
    FrameView v;
    if (fmt_.layout == Layout::YUY2) {
      v.data[0] = f->GetReadPtr();
      v.stride[0] = f->GetPitch();
    } else {
      static const int planes[] = {PLANAR_Y, PLANAR_U, PLANAR_V};
      for (int i = 0; i < (fmt_.chroma ? 3 : 1); i++) {
        v.data[i] = f->GetReadPtr(planes[i]);
        v.stride[i] = f->GetPitch(planes[i]);
      }
    }
    if (has_v8_) {
      const AVSMap* props = env->getFramePropsRO(f);
      v.scene_prev = propInt(env, props, "_SceneChangePrev");
      v.scene_next = propInt(env, props, "_SceneChangeNext");
      v.field_based = propInt(env, props, "_FieldBased");
    }
    return v;
  }

  Format fmt_;
  std::vector<PClip> prev_, next_;
  bool has_v8_ = false;
  std::unique_ptr<Engine> engine_;
};

Format formatOf(const VideoInfo& vi, IScriptEnvironment* env) {
  Format f;
  f.width = vi.width;
  f.height = vi.height;
  f.bits = vi.BitsPerComponent();
  if (vi.IsYUY2()) {
    f.layout = Layout::YUY2;
  } else if (vi.IsY()) {
    f.layout = Layout::Planar;
  } else if (vi.IsPlanar() && (vi.IsYUV() || vi.IsYUVA())) {
    f.layout = Layout::Planar;
    f.chroma = true;
    f.ssw = vi.GetPlaneWidthSubsampling(PLANAR_U);
    f.ssh = vi.GetPlaneHeightSubsampling(PLANAR_U);
  } else {
    env->ThrowError("Dust: unsupported format: use Y, planar YUV(A) or YUY2 with 8-16 bit integer samples");
  }
  if (f.bits == 32) env->ThrowError("Dust: 32 bit float is not supported");
  return f;
}

std::vector<PClip> clipArray(const AVSValue& v) {
  std::vector<PClip> out;
  if (!v.Defined()) return out;
  if (v.IsArray()) {
    for (int i = 0; i < v.ArraySize(); i++) out.push_back(v[i].AsClip());
  } else if (v.IsClip()) {
    out.push_back(v.AsClip());
  }
  return out;
}

int optBool(const AVSValue& v) { return v.Defined() ? (v.AsBool() ? 1 : 0) : -1; }

struct Defaults {
  int mode, limit;
};
const Defaults kFaery = {0, 2}, kPixie = {1, 5}, kGold = {2, 8}, kSpace = {3, 6};

// c[limit]i[mode]i[prev]c+[next]c+[tr]i[temporal]i[limit_edge]i[edge_thresh]i[planes]s[scthresh]i[interlaced]b[opt]i
AVSValue __cdecl CreateTemporal(AVSValue a, void* user, IScriptEnvironment* env) {
  const Defaults& d = *static_cast<const Defaults*>(user);
  PClip child = a[0].AsClip();
  Params p;
  p.limit = a[1].AsInt(d.limit);
  p.mode = a[2].AsInt(d.mode);
  std::vector<PClip> prev = clipArray(a[3]), next = clipArray(a[4]);
  p.tr = a[5].AsInt(1);
  p.temporal = a[6].AsInt(0);
  p.limit_edge = a[7].AsInt(-1);
  p.edge_thresh = a[8].AsInt(32);
  try {
    parsePlanes(a[9].AsString("yuv"), p.planes);
  } catch (const std::exception& e) {
    env->ThrowError("%s", e.what());
  }
  p.scthresh = a[10].AsInt(20);
  p.interlaced = optBool(a[11]);
  p.opt = a[12].AsInt(-1);
  p.num_prev = int(prev.size());
  p.num_next = int(next.size());
  const Format fmt = formatOf(child->GetVideoInfo(), env);
  return new DustAvs(child, p, fmt, std::move(prev), std::move(next), env);
}

// c[limit]i[mode]i[limit_edge]i[edge_thresh]i[planes]s[scthresh]i[interlaced]b[opt]i
AVSValue __cdecl CreateSpace(AVSValue a, void*, IScriptEnvironment* env) {
  PClip child = a[0].AsClip();
  Params p;
  p.limit = a[1].AsInt(kSpace.limit);
  p.mode = a[2].AsInt(kSpace.mode);
  p.limit_edge = a[3].AsInt(-1);
  p.edge_thresh = a[4].AsInt(32);
  try {
    parsePlanes(a[5].AsString("yuv"), p.planes);
  } catch (const std::exception& e) {
    env->ThrowError("%s", e.what());
  }
  p.scthresh = a[6].AsInt(20);
  p.interlaced = optBool(a[7]);
  p.opt = a[8].AsInt(-1);
  const Format fmt = formatOf(child->GetVideoInfo(), env);
  return new DustAvs(child, p, fmt, {}, {}, env);
}

}  // namespace

#ifdef _WIN32
#define DUST_AVS_EXPORT extern "C" __declspec(dllexport)
#else
#define DUST_AVS_EXPORT extern "C" __attribute__((visibility("default")))
#endif

const AVS_Linkage* AVS_linkage = nullptr;

DUST_AVS_EXPORT const char* __stdcall AvisynthPluginInit3(IScriptEnvironment* env, const AVS_Linkage* const vectors) {
  AVS_linkage = vectors;
  const char* full =
      "c[limit]i[mode]i[prev]c+[next]c+[tr]i[temporal]i[limit_edge]i[edge_thresh]i[planes]s[scthresh]i[interlaced]b"
      "[opt]i";
  const char* space = "c[limit]i[mode]i[limit_edge]i[edge_thresh]i[planes]s[scthresh]i[interlaced]b[opt]i";
  env->AddFunction("FaeryDust", full, CreateTemporal, const_cast<Defaults*>(&kFaery));
  env->AddFunction("PixieDust", full, CreateTemporal, const_cast<Defaults*>(&kPixie));
  env->AddFunction("GoldDust", full, CreateTemporal, const_cast<Defaults*>(&kGold));
  env->AddFunction("SpaceDust", space, CreateSpace, nullptr);
  return "Dust " DUST_VERSION;
}
