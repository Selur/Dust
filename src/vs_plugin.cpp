// SPDX-License-Identifier: MIT
// VapourSynth (API 4) interface: dust.FaeryDust, PixieDust, GoldDust, SpaceDust. Gray and YUV, 8-16 bit integer.
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "VSHelper4.h"
#include "VapourSynth4.h"
#include "core/dust.hpp"
#include "version.hpp"

using namespace dust;

namespace {

struct Defaults {
  const char* name;
  int mode, limit;
  bool temporal_args;
};
const Defaults kFaery = {"FaeryDust", 0, 2, true}, kPixie = {"PixieDust", 1, 5, true},
               kGold = {"GoldDust", 2, 8, true}, kSpace = {"SpaceDust", 3, 6, false};

struct FilterData {
  VSNode* node = nullptr;
  std::vector<VSNode*> prev, next;
  Format fmt;
  bool planes[3] = {true, true, true};
  std::unique_ptr<Engine> engine;
};

int propInt(const VSAPI* vsapi, const VSMap* props, const char* key) {
  int err = 0;
  const int64_t v = vsapi->mapGetInt(props, key, 0, &err);
  return err ? -1 : int(v);
}

FrameView view(const VSFrame* f, const FilterData* d, const VSAPI* vsapi) {
  FrameView v;
  for (int i = 0; i < (d->fmt.chroma ? 3 : 1); i++) {
    v.data[i] = vsapi->getReadPtr(f, i);
    v.stride[i] = vsapi->getStride(f, i);
  }
  const VSMap* props = vsapi->getFramePropertiesRO(f);
  v.scene_prev = propInt(vsapi, props, "_SceneChangePrev");
  v.scene_next = propInt(vsapi, props, "_SceneChangeNext");
  v.field_based = propInt(vsapi, props, "_FieldBased");
  return v;
}

const VSFrame* VS_CC getFrame(int n, int reason, void* instance, void**, VSFrameContext* ctx, VSCore* core,
                              const VSAPI* vsapi) {
  auto* d = static_cast<FilterData*>(instance);
  int first = n, last = n;
  d->engine->sourceRange(n, &first, &last);
  if (reason == arInitial) {
    for (int k = first; k <= last; k++) vsapi->requestFrameFilter(k, d->node, ctx);
    if (d->engine->usesMc()) {
      for (VSNode* c : d->prev) vsapi->requestFrameFilter(n, c, ctx);
      for (VSNode* c : d->next) vsapi->requestFrameFilter(n, c, ctx);
    }
    return nullptr;
  }
  if (reason != arAllFramesReady) return nullptr;

  std::vector<const VSFrame*> hold;
  std::vector<FrameView> src, prev, next;
  const VSFrame* centre = nullptr;
  for (int k = first; k <= last; k++) {
    hold.push_back(vsapi->getFrameFilter(k, d->node, ctx));
    if (k == n) centre = hold.back();
    src.push_back(view(hold.back(), d, vsapi));
  }
  if (d->engine->usesMc()) {
    for (VSNode* c : d->prev) {
      hold.push_back(vsapi->getFrameFilter(n, c, ctx));
      prev.push_back(view(hold.back(), d, vsapi));
    }
    for (VSNode* c : d->next) {
      hold.push_back(vsapi->getFrameFilter(n, c, ctx));
      next.push_back(view(hold.back(), d, vsapi));
    }
  }

  // Planes that are not processed are shared with the source frame.
  const int np = d->fmt.chroma ? 3 : 1;
  const VSFrame* plane_src[3] = {nullptr, nullptr, nullptr};
  const int plane_idx[3] = {0, 1, 2};
  for (int i = 0; i < np; i++)
    if (!d->planes[i]) plane_src[i] = centre;
  VSFrame* dst = vsapi->newVideoFrame2(vsapi->getVideoFrameFormat(centre), vsapi->getFrameWidth(centre, 0),
                                       vsapi->getFrameHeight(centre, 0), plane_src, plane_idx, centre, core);
  FrameOut out;
  for (int i = 0; i < np; i++) {
    if (!d->planes[i]) continue;
    out.data[i] = vsapi->getWritePtr(dst, i);
    out.stride[i] = vsapi->getStride(dst, i);
  }
  const VSFrame* result = dst;
  try {
    d->engine->process(n, first, src, prev, next, out);
  } catch (const std::exception& e) {
    vsapi->setFilterError(e.what(), ctx);
    vsapi->freeFrame(dst);
    result = nullptr;
  }
  for (const VSFrame* f : hold) vsapi->freeFrame(f);
  return result;
}

void freeNodes(FilterData* d, const VSAPI* vsapi) {
  vsapi->freeNode(d->node);
  for (VSNode* c : d->prev) vsapi->freeNode(c);
  for (VSNode* c : d->next) vsapi->freeNode(c);
}

void VS_CC freeFilter(void* instance, VSCore*, const VSAPI* vsapi) {
  auto* d = static_cast<FilterData*>(instance);
  freeNodes(d, vsapi);
  delete d;
}

void VS_CC create(const VSMap* in, VSMap* out, void* user, VSCore* core, const VSAPI* vsapi) {
  const Defaults& def = *static_cast<const Defaults*>(user);
  auto d = std::make_unique<FilterData>();
  d->node = vsapi->mapGetNode(in, "clip", 0, nullptr);
  auto fail = [&](const std::string& msg) {
    vsapi->mapSetError(out, msg.c_str());
    freeNodes(d.get(), vsapi);
  };

  int e = 0;
  Params p;
  p.mode = def.mode;
  p.limit = def.limit;
  auto i = [&](const char* key, int& v) {
    const int64_t x = vsapi->mapGetInt(in, key, 0, &e);
    if (!e) v = int(x);
  };
  i("limit", p.limit);
  i("mode", p.mode);
  i("limit_edge", p.limit_edge);
  i("edge_thresh", p.edge_thresh);
  i("scthresh", p.scthresh);
  i("opt", p.opt);
  {
    const int64_t x = vsapi->mapGetInt(in, "interlaced", 0, &e);
    p.interlaced = e ? -1 : (x != 0 ? 1 : 0);
  }
  if (def.temporal_args) {
    i("tr", p.tr);
    i("temporal", p.temporal);
    for (int k = 0; k < vsapi->mapNumElements(in, "prev"); k++) d->prev.push_back(vsapi->mapGetNode(in, "prev", k, nullptr));
    for (int k = 0; k < vsapi->mapNumElements(in, "next"); k++) d->next.push_back(vsapi->mapGetNode(in, "next", k, nullptr));
  }
  p.num_prev = int(d->prev.size());
  p.num_next = int(d->next.size());

  const VSVideoInfo* vi = vsapi->getVideoInfo(d->node);
  if (!vsh::isConstantVideoFormat(vi)) return fail("Dust: only constant format input is supported");
  const VSVideoFormat& vf = vi->format;
  Format& fmt = d->fmt;
  fmt.width = vi->width;
  fmt.height = vi->height;
  fmt.bits = vf.bitsPerSample;
  if (vf.sampleType != stInteger || vf.bitsPerSample > 16)
    return fail(vf.sampleType == stFloat ? "Dust: float is not supported"
                                         : "Dust: unsupported format: use Gray or YUV with 8-16 bit integer samples");
  if (vf.colorFamily == cfYUV) {
    fmt.chroma = true;
    fmt.ssw = vf.subSamplingW;
    fmt.ssh = vf.subSamplingH;
  } else if (vf.colorFamily != cfGray) {
    return fail("Dust: unsupported format: use Gray or YUV with 8-16 bit integer samples");
  }

  const int np = vf.numPlanes;
  const int nplanes = vsapi->mapNumElements(in, "planes");
  if (nplanes > 0) {
    for (int k = 0; k < 3; k++) p.planes[k] = false;
    for (int k = 0; k < nplanes; k++) {
      const int64_t pl = vsapi->mapGetInt(in, "planes", k, nullptr);
      if (pl < 0 || pl >= np) return fail("Dust: plane index " + std::to_string(pl) + " out of range");
      if (p.planes[pl]) return fail("Dust: plane " + std::to_string(pl) + " specified twice");
      p.planes[pl] = true;
    }
  }
  for (int k = 0; k < 3; k++) d->planes[k] = p.planes[k];

  auto check = [&](VSNode* c, const char* tag, int k) -> bool {
    const VSVideoInfo* m = vsapi->getVideoInfo(c);
    std::string where = std::string("Dust: ") + tag + "[" + std::to_string(k) + "] ";
    if (m->width != vi->width || m->height != vi->height) return fail(where + "size mismatch"), false;
    if (!vsh::isSameVideoFormat(&m->format, &vf)) return fail(where + "format mismatch"), false;
    if (m->numFrames < vi->numFrames) return fail(where + "shorter than the main clip"), false;
    return true;
  };
  for (int k = 0; k < int(d->prev.size()); k++)
    if (!check(d->prev[k], "prev", k)) return;
  for (int k = 0; k < int(d->next.size()); k++)
    if (!check(d->next[k], "next", k)) return;

  try {
    d->engine = std::make_unique<Engine>(p, fmt, vi->numFrames);
  } catch (const std::exception& ex) {
    return fail(ex.what());
  }

  std::vector<VSFilterDependency> deps = {{d->node, rpGeneral}};
  for (VSNode* c : d->prev) deps.push_back({c, rpGeneral});
  for (VSNode* c : d->next) deps.push_back({c, rpGeneral});
  vsapi->createVideoFilter(out, def.name, vi, getFrame, freeFilter, fmParallel, deps.data(), int(deps.size()),
                           d.release(), core);
}

}  // namespace

VS_EXTERNAL_API(void) VapourSynthPluginInit2(VSPlugin* plugin, const VSPLUGINAPI* vspapi) {
  vspapi->configPlugin("io.github.selur.dust", "dust", "Dust " DUST_VERSION,
                       VS_MAKE_VERSION(DUST_VERSION_MAJOR, DUST_VERSION_MINOR), VAPOURSYNTH_API_VERSION, 0, plugin);
  const char* common = "limit:int:opt;mode:int:opt;limit_edge:int:opt;edge_thresh:int:opt;planes:int[]:opt;"
                       "scthresh:int:opt;interlaced:int:opt;opt:int:opt;";
  const std::string temporal = std::string("clip:vnode;") + common +
                               "tr:int:opt;temporal:int:opt;prev:vnode[]:opt;next:vnode[]:opt;";
  const std::string space = std::string("clip:vnode;") + common;
  for (const Defaults* f : {&kFaery, &kPixie, &kGold})
    vspapi->registerFunction(f->name, temporal.c_str(), "clip:vnode;", create, const_cast<Defaults*>(f), plugin);
  vspapi->registerFunction(kSpace.name, space.c_str(), "clip:vnode;", create, const_cast<Defaults*>(&kSpace), plugin);
}
