// SPDX-License-Identifier: MIT
// NEON kernels for arm64 (16 pixels per step at 8 bit, 8 at 9-16 bit); widening and packing keep pixel order.
#include <arm_neon.h>

#include "kernels.hpp"
#include "median_nets.hpp"
#include "scalar.hpp"

namespace dust {
namespace {

struct S {
  using V8 = uint8x16_t;
  using V16 = uint16x8_t;
  using V32 = uint32x4_t;
  static constexpr int N8 = 16, N16 = 8;

  static V8 load8(const uint8_t* p) { return vld1q_u8(p); }
  static void store8(uint8_t* p, V8 v) { vst1q_u8(p, v); }
  static V16 load16(const uint8_t* p) { return vld1q_u16(reinterpret_cast<const uint16_t*>(p)); }
  static void store16(uint8_t* p, V16 v) { vst1q_u16(reinterpret_cast<uint16_t*>(p), v); }
  static V16 lo8(V8 v) { return vmovl_u8(vget_low_u8(v)); }
  static V16 hi8(V8 v) { return vmovl_u8(vget_high_u8(v)); }
  static V8 pack16(V16 a, V16 b) { return vcombine_u8(vqmovn_u16(a), vqmovn_u16(b)); }
  static V32 lo16(V16 v) { return vmovl_u16(vget_low_u16(v)); }
  static V32 hi16(V16 v) { return vmovl_u16(vget_high_u16(v)); }
  static V16 pack32(V32 a, V32 b) { return vcombine_u16(vqmovn_u32(a), vqmovn_u32(b)); }
  static V16 loadHalf8(const uint8_t* p) { return vmovl_u8(vld1_u8(p)); }
  static void storeHalf8(uint8_t* p, V16 v) { vst1_u8(p, vqmovn_u16(v)); }

  static V8 min8(V8 a, V8 b) { return vminq_u8(a, b); }
  static V8 max8(V8 a, V8 b) { return vmaxq_u8(a, b); }
  static V16 min16(V16 a, V16 b) { return vminq_u16(a, b); }
  static V16 max16(V16 a, V16 b) { return vmaxq_u16(a, b); }
  static V16 add16(V16 a, V16 b) { return vaddq_u16(a, b); }
  static V16 sub16(V16 a, V16 b) { return vsubq_u16(a, b); }
  static V16 adds16(V16 a, V16 b) { return vqaddq_u16(a, b); }
  static V16 subs16(V16 a, V16 b) { return vqsubq_u16(a, b); }
  static V16 mullo16(V16 a, V16 b) { return vmulq_u16(a, b); }
  static V16 set16(int v) { return vdupq_n_u16(uint16_t(v)); }
  static V16 shl16(V16 v, int s) { return vshlq_u16(v, vdupq_n_s16(int16_t(s))); }
  static V16 shr16(V16 v, int s) { return vshlq_u16(v, vdupq_n_s16(int16_t(-s))); }
  static V16 mulhi16(V16 v, uint16_t m) {
    const uint16x4_t mm = vdup_n_u16(m);
    return vcombine_u16(vshrn_n_u32(vmull_u16(vget_low_u16(v), mm), 16),
                        vshrn_n_u32(vmull_u16(vget_high_u16(v), mm), 16));
  }
  static V32 add32(V32 a, V32 b) { return vaddq_u32(a, b); }
  static V32 sub32(V32 a, V32 b) { return vsubq_u32(a, b); }
  static V32 set32(int v) { return vdupq_n_u32(uint32_t(v)); }
  static V32 shl32(V32 v, int s) { return vshlq_u32(v, vdupq_n_s32(s)); }
  static V32 shr32(V32 v, int s) { return vshlq_u32(v, vdupq_n_s32(-s)); }
  static V32 mulshr32(V32 v, const Magic32& m) {
    const uint32x2_t mm = vdup_n_u32(m.m);
    const int64x2_t cnt = vdupq_n_s64(-(32 + m.s));
    const uint64x2_t lo = vshlq_u64(vmull_u32(vget_low_u32(v), mm), cnt);
    const uint64x2_t hi = vshlq_u64(vmull_u32(vget_high_u32(v), mm), cnt);
    return vcombine_u32(vmovn_u64(lo), vmovn_u64(hi));
  }
  static V16 and16(V16 a, V16 b) { return vandq_u16(a, b); }
  static V32 and32(V32 a, V32 b) { return vandq_u32(a, b); }
  static V16 asU16(V8 v) { return vreinterpretq_u16_u8(v); }
  static V32 asU32(V16 v) { return vreinterpretq_u32_u16(v); }
  static V32 sum16to32(V16 v) { return vpaddlq_u16(v); }
  static V32 min32(V32 a, V32 b) { return vminq_u32(a, b); }
  static V32 max32(V32 a, V32 b) { return vmaxq_u32(a, b); }
  static uint64_t hsum32(V32 v) { return vaddlvq_u32(v); }
};

#include "kernels_simd.hpp"

}  // namespace

const Kernels& kernelsNEON() { return simdKernels(); }

}  // namespace dust
