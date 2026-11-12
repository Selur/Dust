// SPDX-License-Identifier: MIT
// SSE4.1 kernels (16 pixels per step at 8 bit, 8 at 9-16 bit). The same file is compiled a second time with AVX
// enabled and DUST_AVX defined: identical intrinsics, VEX encoded (three-operand forms, no 256-bit integer math).
#include <smmintrin.h>

#include "kernels.hpp"
#include "median_nets.hpp"
#include "scalar.hpp"

namespace dust {
namespace {

struct S {
  using V8 = __m128i;
  using V16 = __m128i;
  using V32 = __m128i;
  static constexpr int N8 = 16, N16 = 8;

  static V8 load8(const uint8_t* p) { return _mm_loadu_si128(reinterpret_cast<const __m128i*>(p)); }
  static void store8(uint8_t* p, V8 v) { _mm_storeu_si128(reinterpret_cast<__m128i*>(p), v); }
  static V16 load16(const uint8_t* p) { return load8(p); }
  static void store16(uint8_t* p, V16 v) { store8(p, v); }
  static V16 lo8(V8 v) { return _mm_unpacklo_epi8(v, _mm_setzero_si128()); }
  static V16 hi8(V8 v) { return _mm_unpackhi_epi8(v, _mm_setzero_si128()); }
  static V8 pack16(V16 a, V16 b) { return _mm_packus_epi16(a, b); }
  static V32 lo16(V16 v) { return _mm_unpacklo_epi16(v, _mm_setzero_si128()); }
  static V32 hi16(V16 v) { return _mm_unpackhi_epi16(v, _mm_setzero_si128()); }
  static V16 pack32(V32 a, V32 b) { return _mm_packus_epi32(a, b); }
  static V16 loadHalf8(const uint8_t* p) {
    return _mm_cvtepu8_epi16(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p)));
  }
  static void storeHalf8(uint8_t* p, V16 v) {
    _mm_storel_epi64(reinterpret_cast<__m128i*>(p), _mm_packus_epi16(v, v));
  }

  static V8 min8(V8 a, V8 b) { return _mm_min_epu8(a, b); }
  static V8 max8(V8 a, V8 b) { return _mm_max_epu8(a, b); }
  static V16 min16(V16 a, V16 b) { return _mm_min_epu16(a, b); }
  static V16 max16(V16 a, V16 b) { return _mm_max_epu16(a, b); }
  static V16 add16(V16 a, V16 b) { return _mm_add_epi16(a, b); }
  static V16 sub16(V16 a, V16 b) { return _mm_sub_epi16(a, b); }
  static V16 adds16(V16 a, V16 b) { return _mm_adds_epu16(a, b); }
  static V16 subs16(V16 a, V16 b) { return _mm_subs_epu16(a, b); }
  static V16 mullo16(V16 a, V16 b) { return _mm_mullo_epi16(a, b); }
  static V16 set16(int v) { return _mm_set1_epi16(short(v)); }
  static V16 shl16(V16 v, int s) { return _mm_sll_epi16(v, _mm_cvtsi32_si128(s)); }
  static V16 shr16(V16 v, int s) { return _mm_srl_epi16(v, _mm_cvtsi32_si128(s)); }
  static V16 mulhi16(V16 v, uint16_t m) { return _mm_mulhi_epu16(v, _mm_set1_epi16(short(m))); }
  static V32 add32(V32 a, V32 b) { return _mm_add_epi32(a, b); }
  static V32 sub32(V32 a, V32 b) { return _mm_sub_epi32(a, b); }
  static V32 set32(int v) { return _mm_set1_epi32(v); }
  static V32 shl32(V32 v, int s) { return _mm_sll_epi32(v, _mm_cvtsi32_si128(s)); }
  static V32 shr32(V32 v, int s) { return _mm_srl_epi32(v, _mm_cvtsi32_si128(s)); }
  static V32 mulshr32(V32 v, const Magic32& m) {
    const __m128i mm = _mm_set1_epi32(int(m.m));
    const __m128i cnt = _mm_cvtsi32_si128(32 + m.s);
    const __m128i even = _mm_srl_epi64(_mm_mul_epu32(v, mm), cnt);
    const __m128i odd = _mm_srl_epi64(_mm_mul_epu32(_mm_srli_epi64(v, 32), mm), cnt);
    return _mm_or_si128(even, _mm_slli_epi64(odd, 32));
  }
  static V16 and16(V16 a, V16 b) { return _mm_and_si128(a, b); }
  static V32 and32(V32 a, V32 b) { return _mm_and_si128(a, b); }
  static V16 asU16(V8 v) { return v; }
  static V32 asU32(V16 v) { return v; }
  static V32 sum16to32(V16 v) { return _mm_madd_epi16(v, _mm_set1_epi16(1)); }
  static V32 min32(V32 a, V32 b) { return _mm_min_epu32(a, b); }
  static V32 max32(V32 a, V32 b) { return _mm_max_epu32(a, b); }
  static uint64_t hsum32(V32 v) {
    alignas(16) uint32_t t[4];
    _mm_store_si128(reinterpret_cast<__m128i*>(t), v);
    return uint64_t(t[0]) + t[1] + t[2] + t[3];
  }
};

#include "kernels_simd.hpp"

}  // namespace

#if defined(DUST_AVX)
const Kernels& kernelsAVX() { return simdKernels(); }
#else
const Kernels& kernelsSSE41() { return simdKernels(); }
#endif

}  // namespace dust
