// SPDX-License-Identifier: MIT
// AVX2 kernels (32 pixels per step at 8 bit, 16 at 9-16 bit). Widening and packing work per 128-bit lane, which
// keeps pack(lo(v), hi(v)) == v; loadHalf8/storeHalf8 convert in pixel order.
#include <immintrin.h>

#include "kernels.hpp"
#include "median_nets.hpp"
#include "scalar.hpp"

namespace dust {
namespace {

struct S {
  using V8 = __m256i;
  using V16 = __m256i;
  using V32 = __m256i;
  static constexpr int N8 = 32, N16 = 16;

  static V8 load8(const uint8_t* p) { return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p)); }
  static void store8(uint8_t* p, V8 v) { _mm256_storeu_si256(reinterpret_cast<__m256i*>(p), v); }
  static V16 load16(const uint8_t* p) { return load8(p); }
  static void store16(uint8_t* p, V16 v) { store8(p, v); }
  static V16 lo8(V8 v) { return _mm256_unpacklo_epi8(v, _mm256_setzero_si256()); }
  static V16 hi8(V8 v) { return _mm256_unpackhi_epi8(v, _mm256_setzero_si256()); }
  static V8 pack16(V16 a, V16 b) { return _mm256_packus_epi16(a, b); }
  static V32 lo16(V16 v) { return _mm256_unpacklo_epi16(v, _mm256_setzero_si256()); }
  static V32 hi16(V16 v) { return _mm256_unpackhi_epi16(v, _mm256_setzero_si256()); }
  static V16 pack32(V32 a, V32 b) { return _mm256_packus_epi32(a, b); }
  static V16 loadHalf8(const uint8_t* p) {
    return _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p)));
  }
  static void storeHalf8(uint8_t* p, V16 v) {
    _mm_storeu_si128(reinterpret_cast<__m128i*>(p),
                     _mm_packus_epi16(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1)));
  }

  static V8 min8(V8 a, V8 b) { return _mm256_min_epu8(a, b); }
  static V8 max8(V8 a, V8 b) { return _mm256_max_epu8(a, b); }
  static V16 min16(V16 a, V16 b) { return _mm256_min_epu16(a, b); }
  static V16 max16(V16 a, V16 b) { return _mm256_max_epu16(a, b); }
  static V16 add16(V16 a, V16 b) { return _mm256_add_epi16(a, b); }
  static V16 sub16(V16 a, V16 b) { return _mm256_sub_epi16(a, b); }
  static V16 adds16(V16 a, V16 b) { return _mm256_adds_epu16(a, b); }
  static V16 subs16(V16 a, V16 b) { return _mm256_subs_epu16(a, b); }
  static V16 mullo16(V16 a, V16 b) { return _mm256_mullo_epi16(a, b); }
  static V16 set16(int v) { return _mm256_set1_epi16(short(v)); }
  static V16 shl16(V16 v, int s) { return _mm256_sll_epi16(v, _mm_cvtsi32_si128(s)); }
  static V16 shr16(V16 v, int s) { return _mm256_srl_epi16(v, _mm_cvtsi32_si128(s)); }
  static V16 mulhi16(V16 v, uint16_t m) { return _mm256_mulhi_epu16(v, _mm256_set1_epi16(short(m))); }
  static V32 add32(V32 a, V32 b) { return _mm256_add_epi32(a, b); }
  static V32 sub32(V32 a, V32 b) { return _mm256_sub_epi32(a, b); }
  static V32 set32(int v) { return _mm256_set1_epi32(v); }
  static V32 shl32(V32 v, int s) { return _mm256_sll_epi32(v, _mm_cvtsi32_si128(s)); }
  static V32 shr32(V32 v, int s) { return _mm256_srl_epi32(v, _mm_cvtsi32_si128(s)); }
  static V32 mulshr32(V32 v, const Magic32& m) {
    const __m256i mm = _mm256_set1_epi32(int(m.m));
    const __m128i cnt = _mm_cvtsi32_si128(32 + m.s);
    const __m256i even = _mm256_srl_epi64(_mm256_mul_epu32(v, mm), cnt);
    const __m256i odd = _mm256_srl_epi64(_mm256_mul_epu32(_mm256_srli_epi64(v, 32), mm), cnt);
    return _mm256_or_si256(even, _mm256_slli_epi64(odd, 32));
  }
  static V16 and16(V16 a, V16 b) { return _mm256_and_si256(a, b); }
  static V32 and32(V32 a, V32 b) { return _mm256_and_si256(a, b); }
  static V16 asU16(V8 v) { return v; }
  static V32 asU32(V16 v) { return v; }
  static V32 sum16to32(V16 v) { return _mm256_madd_epi16(v, _mm256_set1_epi16(1)); }
  static V32 min32(V32 a, V32 b) { return _mm256_min_epu32(a, b); }
  static V32 max32(V32 a, V32 b) { return _mm256_max_epu32(a, b); }
  static uint64_t hsum32(V32 v) {
    alignas(32) uint32_t t[8];
    _mm256_store_si256(reinterpret_cast<__m256i*>(t), v);
    uint64_t s = 0;
    for (uint32_t x : t) s += x;
    return s;
  }
};

#include "kernels_simd.hpp"

}  // namespace

const Kernels& kernelsAVX2() { return simdKernels(); }

}  // namespace dust
