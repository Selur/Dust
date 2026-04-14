// DustPort.cpp — AviSynth+ plugin
// =============================================================================
// DustPort — limit-clipping denoiser in the philosophy of Steady's DustV5 (2003)
//
// Version:  1.0.0
// Author:   Yuri Pronkin (doom9: shurik_pronkin, GitHub: schpuppa-art)
// License:  MIT (see LICENSE)
// GitHub:   https://github.com/schpuppa-art/DustPort
// doom9:    <TBD — thread URL goes here after first post>
//
// Overview
// --------
// DustPort is a Level 1 reference-port of Steady's DustV5 (2003) — a limit-clipping
// spatiotemporal denoiser. The guarantee of the architecture is that every output
// pixel stays within a bounded distance from the source: out = clamp(filter, src ± limit).
// This makes the plugin well-suited for film restoration where drift and
// hallucinated detail must be avoided.
//
// DustPort is NOT bit-exact with the original DustV5 DLL. It reconstructs the
// four classic modes (Faery/Pixie/Gold/Space) from reverse-engineering of the
// original 2003 binary combined with the published ReadMe, and adds modern
// extensions: larger temporal radius, median option, external motion compensation,
// edge-aware limit map, 10–16 bit planar YUV support, and per-plane processing.
//
// Original DustV5 by Steady (2003-01-24). Source code was never published.
// This port is an independent implementation inspired by the same philosophy.
//
// Pixel formats supported
// -----------------------
//   YUY2 (8-bit packed)                               — no edge-aware
//   YV12 / YV16 / YV24 (8-bit planar)                 — full features
//   YUV420P10..P16 / YUV422P10..P16 / YUV444P10..P16  — full features
//
// Build
// -----
//   cl /LD /EHsc /O2 /arch:AVX2 /I"path_to_FilterSDK_include" DustPort.cpp
//   /arch:AVX2 implies SSE4.1 which is required for 16-bit min/max intrinsics.
// =============================================================================

#define NOMINMAX
#include <windows.h>
#include <emmintrin.h>    // SSE2
#include <smmintrin.h>    // SSE4.1 (for _mm_min_epu16, _mm_max_epu16, _mm_min_epu32 etc.)
#include "avisynth.h"
#include <cstdint>
#include <algorithm>
#include <cstring>
#include <vector>
#include <cmath>

// ---------------------------------------------------------------------------
// Scalar helpers
// ---------------------------------------------------------------------------

static inline uint8_t limit_clip8(int src, int filt, int limit) {
    int lo = src - limit; if (lo < 0)   lo = 0;
    int hi = src + limit; if (hi > 255) hi = 255;
    if (filt < lo) return (uint8_t)lo;
    if (filt > hi) return (uint8_t)hi;
    return (uint8_t)filt;
}

static inline uint16_t limit_clip16(int src, int filt, int limit, int max_val) {
    int lo = src - limit; if (lo < 0) lo = 0;
    int hi = src + limit; if (hi > max_val) hi = max_val;
    if (filt < lo) return (uint16_t)lo;
    if (filt > hi) return (uint16_t)hi;
    return (uint16_t)filt;
}

static inline uint8_t median_scalar8(const uint8_t* vals, int n) {
    uint8_t tmp[11];
    for (int i = 0; i < n; ++i) tmp[i] = vals[i];
    for (int i = 1; i < n; ++i) {
        uint8_t x = tmp[i];
        int j = i - 1;
        while (j >= 0 && tmp[j] > x) { tmp[j+1] = tmp[j]; --j; }
        tmp[j+1] = x;
    }
    return tmp[n/2];
}

static inline uint16_t median_scalar16(const uint16_t* vals, int n) {
    uint16_t tmp[11];
    for (int i = 0; i < n; ++i) tmp[i] = vals[i];
    for (int i = 1; i < n; ++i) {
        uint16_t x = tmp[i];
        int j = i - 1;
        while (j >= 0 && tmp[j] > x) { tmp[j+1] = tmp[j]; --j; }
        tmp[j+1] = x;
    }
    return tmp[n/2];
}

// ---------------------------------------------------------------------------
// SSE primitives — 8-bit
// ---------------------------------------------------------------------------

static inline __m128i limit_clip_sse2_8(__m128i src, __m128i filt, __m128i vlimit) {
    __m128i lo = _mm_subs_epu8(src, vlimit);
    __m128i hi = _mm_adds_epu8(src, vlimit);
    __m128i o  = _mm_max_epu8(filt, lo);
    o          = _mm_min_epu8(o,    hi);
    return o;
}

static inline __m128i spatial_blend_sse2_8(
    __m128i up, __m128i dn, __m128i left, __m128i right, __m128i centre)
{
    __m128i vh = _mm_avg_epu8(up, dn);
    __m128i hh = _mm_avg_epu8(left, right);
    __m128i n4 = _mm_avg_epu8(vh, hh);
    return _mm_avg_epu8(centre, n4);
}

static const uint16_t kRecipTable[12] = {
    0, 0, 0, 21845, 0, 13107, 0, 9362, 0, 7282, 0, 5958
};

static inline __m128i temporal_mean_sse2_8(const __m128i* vals, int count) {
    __m128i zero   = _mm_setzero_si128();
    __m128i sum_lo = zero;
    __m128i sum_hi = zero;
    for (int i = 0; i < count; ++i) {
        __m128i v = vals[i];
        sum_lo = _mm_add_epi16(sum_lo, _mm_unpacklo_epi8(v, zero));
        sum_hi = _mm_add_epi16(sum_hi, _mm_unpackhi_epi8(v, zero));
    }
    __m128i recip = _mm_set1_epi16((short)kRecipTable[count]);
    __m128i r_lo  = _mm_mulhi_epu16(sum_lo, recip);
    __m128i r_hi  = _mm_mulhi_epu16(sum_hi, recip);
    return _mm_packus_epi16(r_lo, r_hi);
}

// ---------------------------------------------------------------------------
// SSE primitives — 16-bit (SSE4.1 required for _mm_min_epu16 / _mm_max_epu16)
// ---------------------------------------------------------------------------

static inline __m128i limit_clip_sse_16(__m128i src, __m128i filt, __m128i vlimit, __m128i vmax) {
    // lo = max(src - limit, 0) using unsigned saturated subtract
    __m128i lo = _mm_subs_epu16(src, vlimit);
    // hi = min(src + limit, vmax) using unsigned saturated add + clamp
    __m128i hi = _mm_adds_epu16(src, vlimit);
    hi         = _mm_min_epu16(hi, vmax);
    __m128i o  = _mm_max_epu16(filt, lo);
    o          = _mm_min_epu16(o,    hi);
    return o;
}

// 16-bit weighted spatial blend (centre 1/2, each neighbour 1/8).
// No _mm_avg_epu16 in SSE, but we can build it: avg_u16(a,b) = (a+b+1) >> 1
// which we express as (a>>1) + (b>>1) + ((a|b)&1). For our purposes simpler
// implementation via 32-bit intermediates is fine:
static inline __m128i avg_epu16(__m128i a, __m128i b) {
    // Rounding average of unsigned 16-bit. Overflow-safe:
    // ((a & b) + ((a ^ b) >> 1)) handles without overflow, but we need rounding.
    // Standard: avg = (a + b + 1) >> 1 = ((a | b) - ((a ^ b) >> 1))  — no overflow.
    __m128i or_  = _mm_or_si128(a, b);
    __m128i xor_ = _mm_xor_si128(a, b);
    __m128i half = _mm_srli_epi16(xor_, 1);
    return _mm_sub_epi16(or_, half);
}

static inline __m128i spatial_blend_sse_16(
    __m128i up, __m128i dn, __m128i left, __m128i right, __m128i centre)
{
    __m128i vh = avg_epu16(up, dn);
    __m128i hh = avg_epu16(left, right);
    __m128i n4 = avg_epu16(vh, hh);
    return avg_epu16(centre, n4);
}

// 16-bit mean: sum up to 11 u16 values into u32, divide by count.
// For a sum up to 11*65535 = 720885, fits in u32 comfortably.
// We unpack u16 -> u32 twice per register (lo/hi halves of 8 u16s -> 4 u32s each).
// Divide via multiply-by-reciprocal: recip_u32 = (1ULL<<32) / count (rounded).
static const uint32_t kRecipTable32[12] = {
    0, 0, 0,
    1431655765u,   // /3   ≈ 4294967296/3
    0,
    858993459u,    // /5
    0,
    613566757u,    // /7
    0,
    477218588u,    // /9
    0,
    390451572u     // /11
};

static inline __m128i temporal_mean_sse_16(const __m128i* vals, int count) {
    __m128i zero = _mm_setzero_si128();
    __m128i sum_ll = zero;  // pixels 0..3 as u32
    __m128i sum_lh = zero;  // pixels 4..7 as u32
    for (int i = 0; i < count; ++i) {
        __m128i v = vals[i];
        sum_ll = _mm_add_epi32(sum_ll, _mm_unpacklo_epi16(v, zero));
        sum_lh = _mm_add_epi32(sum_lh, _mm_unpackhi_epi16(v, zero));
    }
    // Divide via multiply-hi-32: result = (sum * recip) >> 32.
    // No single-instruction mulhi_epu32 in SSE2/SSE4.1; we use _mm_mul_epu32 on
    // pairs (produces u64 from two u32s, giving us the high 32 bits for free).
    __m128i recip = _mm_set1_epi32((int)kRecipTable32[count]);

    // For each of sum_ll, sum_lh: split into odd/even 32-bit lanes,
    // mul_epu32 each, shuffle back.
    auto mulhi32 = [&](__m128i s, __m128i r) -> __m128i {
        __m128i even   = _mm_mul_epu32(s, r);                              // lanes 0,2 -> u64
        __m128i s_odd  = _mm_srli_si128(s, 4);                             // shift down so lanes 1,3 are at 0,2
        __m128i r_odd  = _mm_srli_si128(r, 4);
        __m128i odd    = _mm_mul_epu32(s_odd, r_odd);                      // lanes 1,3 -> u64
        // Extract high 32 bits of each u64
        __m128i eh = _mm_srli_epi64(even, 32);  // lanes 0,2 hi32 -> at u32 lanes 0,2
        __m128i oh = _mm_srli_epi64(odd,  32);  // lanes 1,3 hi32 -> at u32 lanes 0,2
        // Interleave: eh in lanes 0,2; we need oh shifted to lanes 1,3 and OR'd
        oh = _mm_slli_si128(oh, 4);             // move oh to lanes 1,3
        return _mm_or_si128(eh, oh);
    };
    __m128i r_ll = mulhi32(sum_ll, recip);   // 4 u32 results for low 4 pixels
    __m128i r_lh = mulhi32(sum_lh, recip);   // 4 u32 results for high 4 pixels

    // Pack u32 -> u16 (saturating — our results are already <= 65535)
    return _mm_packus_epi32(r_ll, r_lh);
}

// ---------------------------------------------------------------------------
// Median sorting networks — 8-bit
// ---------------------------------------------------------------------------

#define CSWAP8(a, b) do { \
    __m128i _t = _mm_min_epu8((a), (b)); \
    (b) = _mm_max_epu8((a), (b)); \
    (a) = _t; \
} while (0)

static inline __m128i median3_sse2_8(__m128i a, __m128i b, __m128i c) {
    CSWAP8(a, b); CSWAP8(b, c); CSWAP8(a, b);
    return b;
}
static inline __m128i median5_sse2_8(__m128i a, __m128i b, __m128i c, __m128i d, __m128i e) {
    CSWAP8(a, b); CSWAP8(d, e);
    CSWAP8(c, e); CSWAP8(c, d); CSWAP8(a, d);
    CSWAP8(a, c); CSWAP8(b, e); CSWAP8(b, d);
    CSWAP8(b, c);
    return c;
}
static inline __m128i median7_sse2_8(
    __m128i a, __m128i b, __m128i c, __m128i d, __m128i e, __m128i f, __m128i g)
{
    CSWAP8(b, c); CSWAP8(d, e); CSWAP8(f, g);
    CSWAP8(a, c); CSWAP8(d, f); CSWAP8(e, g);
    CSWAP8(a, b); CSWAP8(e, f); CSWAP8(c, g);
    CSWAP8(a, e); CSWAP8(b, f);
    CSWAP8(a, d); CSWAP8(c, f);
    CSWAP8(b, d); CSWAP8(c, e);
    CSWAP8(c, d);
    return d;
}
static inline __m128i median9_sse2_8(
    __m128i p0, __m128i p1, __m128i p2, __m128i p3, __m128i p4,
    __m128i p5, __m128i p6, __m128i p7, __m128i p8)
{
    CSWAP8(p0,p1); CSWAP8(p3,p4); CSWAP8(p6,p7);
    CSWAP8(p1,p2); CSWAP8(p4,p5); CSWAP8(p7,p8);
    CSWAP8(p0,p1); CSWAP8(p3,p4); CSWAP8(p6,p7);
    CSWAP8(p0,p3); CSWAP8(p3,p6);
    CSWAP8(p1,p4); CSWAP8(p4,p7);
    CSWAP8(p2,p5); CSWAP8(p5,p8);
    CSWAP8(p1,p3); CSWAP8(p5,p7);
    CSWAP8(p2,p6); CSWAP8(p4,p6);
    CSWAP8(p2,p4); CSWAP8(p2,p3); CSWAP8(p5,p6);
    CSWAP8(p3,p4);
    return p4;
}
static inline __m128i median11_sse2_8(
    __m128i p0, __m128i p1, __m128i p2, __m128i p3, __m128i p4, __m128i p5,
    __m128i p6, __m128i p7, __m128i p8, __m128i p9, __m128i pa)
{
    CSWAP8(p0,p1); CSWAP8(p2,p3); CSWAP8(p4,p5); CSWAP8(p6,p7); CSWAP8(p8,p9);
    CSWAP8(p1,p3); CSWAP8(p5,p7); CSWAP8(p0,p2); CSWAP8(p4,p6); CSWAP8(p8,pa);
    CSWAP8(p1,p2); CSWAP8(p5,p6); CSWAP8(p9,pa); CSWAP8(p0,p4); CSWAP8(p3,p7);
    CSWAP8(p1,p5); CSWAP8(p6,pa); CSWAP8(p3,p6); CSWAP8(p7,pa); CSWAP8(p1,p4);
    CSWAP8(p2,p5); CSWAP8(p3,p4); CSWAP8(p5,p6); CSWAP8(p7,p8); CSWAP8(p9,pa);
    CSWAP8(p5,p7); CSWAP8(p3,p5); CSWAP8(p4,p6); CSWAP8(p6,p8); CSWAP8(p4,p5);
    CSWAP8(p6,p7); CSWAP8(p5,p6); CSWAP8(p2,p3); CSWAP8(p8,p9); CSWAP8(p5,p6);
    return p5;
}

static inline __m128i temporal_median_sse2_8(const __m128i* v, int count) {
    switch (count) {
        case 3:  return median3_sse2_8(v[0],v[1],v[2]);
        case 5:  return median5_sse2_8(v[0],v[1],v[2],v[3],v[4]);
        case 7:  return median7_sse2_8(v[0],v[1],v[2],v[3],v[4],v[5],v[6]);
        case 9:  return median9_sse2_8(v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],v[8]);
        case 11: return median11_sse2_8(v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],v[8],v[9],v[10]);
        default: return v[count/2];
    }
}

#undef CSWAP8

// ---------------------------------------------------------------------------
// Median sorting networks — 16-bit (SSE4.1)
// ---------------------------------------------------------------------------

#define CSWAP16(a, b) do { \
    __m128i _t = _mm_min_epu16((a), (b)); \
    (b) = _mm_max_epu16((a), (b)); \
    (a) = _t; \
} while (0)

static inline __m128i median3_sse_16(__m128i a, __m128i b, __m128i c) {
    CSWAP16(a, b); CSWAP16(b, c); CSWAP16(a, b);
    return b;
}
static inline __m128i median5_sse_16(__m128i a, __m128i b, __m128i c, __m128i d, __m128i e) {
    CSWAP16(a, b); CSWAP16(d, e);
    CSWAP16(c, e); CSWAP16(c, d); CSWAP16(a, d);
    CSWAP16(a, c); CSWAP16(b, e); CSWAP16(b, d);
    CSWAP16(b, c);
    return c;
}
static inline __m128i median7_sse_16(
    __m128i a, __m128i b, __m128i c, __m128i d, __m128i e, __m128i f, __m128i g)
{
    CSWAP16(b, c); CSWAP16(d, e); CSWAP16(f, g);
    CSWAP16(a, c); CSWAP16(d, f); CSWAP16(e, g);
    CSWAP16(a, b); CSWAP16(e, f); CSWAP16(c, g);
    CSWAP16(a, e); CSWAP16(b, f);
    CSWAP16(a, d); CSWAP16(c, f);
    CSWAP16(b, d); CSWAP16(c, e);
    CSWAP16(c, d);
    return d;
}
static inline __m128i median9_sse_16(
    __m128i p0, __m128i p1, __m128i p2, __m128i p3, __m128i p4,
    __m128i p5, __m128i p6, __m128i p7, __m128i p8)
{
    CSWAP16(p0,p1); CSWAP16(p3,p4); CSWAP16(p6,p7);
    CSWAP16(p1,p2); CSWAP16(p4,p5); CSWAP16(p7,p8);
    CSWAP16(p0,p1); CSWAP16(p3,p4); CSWAP16(p6,p7);
    CSWAP16(p0,p3); CSWAP16(p3,p6);
    CSWAP16(p1,p4); CSWAP16(p4,p7);
    CSWAP16(p2,p5); CSWAP16(p5,p8);
    CSWAP16(p1,p3); CSWAP16(p5,p7);
    CSWAP16(p2,p6); CSWAP16(p4,p6);
    CSWAP16(p2,p4); CSWAP16(p2,p3); CSWAP16(p5,p6);
    CSWAP16(p3,p4);
    return p4;
}
static inline __m128i median11_sse_16(
    __m128i p0, __m128i p1, __m128i p2, __m128i p3, __m128i p4, __m128i p5,
    __m128i p6, __m128i p7, __m128i p8, __m128i p9, __m128i pa)
{
    CSWAP16(p0,p1); CSWAP16(p2,p3); CSWAP16(p4,p5); CSWAP16(p6,p7); CSWAP16(p8,p9);
    CSWAP16(p1,p3); CSWAP16(p5,p7); CSWAP16(p0,p2); CSWAP16(p4,p6); CSWAP16(p8,pa);
    CSWAP16(p1,p2); CSWAP16(p5,p6); CSWAP16(p9,pa); CSWAP16(p0,p4); CSWAP16(p3,p7);
    CSWAP16(p1,p5); CSWAP16(p6,pa); CSWAP16(p3,p6); CSWAP16(p7,pa); CSWAP16(p1,p4);
    CSWAP16(p2,p5); CSWAP16(p3,p4); CSWAP16(p5,p6); CSWAP16(p7,p8); CSWAP16(p9,pa);
    CSWAP16(p5,p7); CSWAP16(p3,p5); CSWAP16(p4,p6); CSWAP16(p6,p8); CSWAP16(p4,p5);
    CSWAP16(p6,p7); CSWAP16(p5,p6); CSWAP16(p2,p3); CSWAP16(p8,p9); CSWAP16(p5,p6);
    return p5;
}

static inline __m128i temporal_median_sse_16(const __m128i* v, int count) {
    switch (count) {
        case 3:  return median3_sse_16(v[0],v[1],v[2]);
        case 5:  return median5_sse_16(v[0],v[1],v[2],v[3],v[4]);
        case 7:  return median7_sse_16(v[0],v[1],v[2],v[3],v[4],v[5],v[6]);
        case 9:  return median9_sse_16(v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],v[8]);
        case 11: return median11_sse_16(v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],v[8],v[9],v[10]);
        default: return v[count/2];
    }
}

#undef CSWAP16

// ---------------------------------------------------------------------------
// Edge-aware limit map — 8-bit
// Output format: u8 per-pixel limit (in 8-bit scale, always).
// For 16-bit mode, the kernel multiplies this by (1<<(bitdepth-8)) when using.
// ---------------------------------------------------------------------------

static void compute_limit_map_8(
    const uint8_t* src, int sp,
    uint8_t* dst, int dp,
    int W, int H,
    int limit_flat, int limit_edge, int edge_thresh)
{
    const int diff = limit_flat - limit_edge;
    const int inv_thresh = (edge_thresh > 0) ? (int)((65535 + edge_thresh / 2) / edge_thresh) : 0;

    for (int y = 0; y < H; ++y)
        memset(dst + (size_t)y * dp, (uint8_t)limit_flat, W);

    if (W < 3 || H < 3) return;

    __m128i vzero       = _mm_setzero_si128();
    __m128i vthresh     = _mm_set1_epi16((short)edge_thresh);
    __m128i vdiff       = _mm_set1_epi16((short)diff);
    __m128i vlimit_flat = _mm_set1_epi16((short)limit_flat);
    __m128i vinv        = _mm_set1_epi16((short)inv_thresh);
    __m128i vtwo        = _mm_set1_epi16(2);

    for (int y = 1; y < H - 1; ++y) {
        const uint8_t* s0 = src + (size_t)(y - 1) * sp;
        const uint8_t* s1 = src + (size_t) y      * sp;
        const uint8_t* s2 = src + (size_t)(y + 1) * sp;
        uint8_t*       d  = dst + (size_t) y      * dp;

        int x = 1;
        for (; x + 17 <= W - 1; x += 16) {
            __m128i a0 = _mm_loadu_si128((const __m128i*)(s0 + x - 1));
            __m128i b0 = _mm_loadu_si128((const __m128i*)(s0 + x    ));
            __m128i c0 = _mm_loadu_si128((const __m128i*)(s0 + x + 1));
            __m128i d0 = _mm_loadu_si128((const __m128i*)(s1 + x - 1));
            __m128i f0 = _mm_loadu_si128((const __m128i*)(s1 + x + 1));
            __m128i g0 = _mm_loadu_si128((const __m128i*)(s2 + x - 1));
            __m128i h0 = _mm_loadu_si128((const __m128i*)(s2 + x    ));
            __m128i i0 = _mm_loadu_si128((const __m128i*)(s2 + x + 1));

            for (int half = 0; half < 2; ++half) {
                __m128i a = half ? _mm_unpackhi_epi8(a0, vzero) : _mm_unpacklo_epi8(a0, vzero);
                __m128i b = half ? _mm_unpackhi_epi8(b0, vzero) : _mm_unpacklo_epi8(b0, vzero);
                __m128i c = half ? _mm_unpackhi_epi8(c0, vzero) : _mm_unpacklo_epi8(c0, vzero);
                __m128i dv= half ? _mm_unpackhi_epi8(d0, vzero) : _mm_unpacklo_epi8(d0, vzero);
                __m128i f = half ? _mm_unpackhi_epi8(f0, vzero) : _mm_unpacklo_epi8(f0, vzero);
                __m128i g = half ? _mm_unpackhi_epi8(g0, vzero) : _mm_unpacklo_epi8(g0, vzero);
                __m128i h = half ? _mm_unpackhi_epi8(h0, vzero) : _mm_unpacklo_epi8(h0, vzero);
                __m128i i = half ? _mm_unpackhi_epi8(i0, vzero) : _mm_unpacklo_epi8(i0, vzero);

                __m128i Sx_pos = _mm_add_epi16(c, _mm_add_epi16(_mm_mullo_epi16(f,  vtwo), i));
                __m128i Sx_neg = _mm_add_epi16(a, _mm_add_epi16(_mm_mullo_epi16(dv, vtwo), g));
                __m128i Sx_abs = _mm_sub_epi16(
                    _mm_max_epi16(Sx_pos, Sx_neg),
                    _mm_min_epi16(Sx_pos, Sx_neg));

                __m128i Sy_pos = _mm_add_epi16(g, _mm_add_epi16(_mm_mullo_epi16(h, vtwo), i));
                __m128i Sy_neg = _mm_add_epi16(a, _mm_add_epi16(_mm_mullo_epi16(b, vtwo), c));
                __m128i Sy_abs = _mm_sub_epi16(
                    _mm_max_epi16(Sy_pos, Sy_neg),
                    _mm_min_epi16(Sy_pos, Sy_neg));

                __m128i grad = _mm_max_epi16(Sx_abs, Sy_abs);
                grad = _mm_min_epi16(grad, vthresh);

                __m128i grad_diff = _mm_mullo_epi16(grad, vdiff);
                __m128i reduction = _mm_mulhi_epu16(grad_diff, vinv);

                __m128i lim = _mm_sub_epi16(vlimit_flat, reduction);
                lim = _mm_max_epi16(lim, _mm_set1_epi16((short)limit_edge));
                lim = _mm_min_epi16(lim, vlimit_flat);

                __m128i lim_u8 = _mm_packus_epi16(lim, lim);
                if (!half) _mm_storel_epi64((__m128i*)(d + x),     lim_u8);
                else       _mm_storel_epi64((__m128i*)(d + x + 8), lim_u8);
            }
        }
        for (; x < W - 1; ++x) {
            int a = s0[x-1], b = s0[x], c = s0[x+1];
            int dvs = s1[x-1],           fv = s1[x+1];
            int g = s2[x-1], h = s2[x], i = s2[x+1];
            int Sx = (c + 2*fv + i) - (a + 2*dvs + g);
            int Sy = (g + 2*h  + i) - (a + 2*b   + c);
            int grad = std::max(std::abs(Sx), std::abs(Sy));
            if (grad > edge_thresh) grad = edge_thresh;
            int reduction = (grad * diff + edge_thresh / 2) / edge_thresh;
            int lim = limit_flat - reduction;
            if (lim < limit_edge) lim = limit_edge;
            if (lim > limit_flat) lim = limit_flat;
            d[x] = (uint8_t)lim;
        }
    }
}

// ---------------------------------------------------------------------------
// Edge-aware limit map — 16-bit source, still outputs 8-bit limit map.
// Sobel is computed on 16-bit values, but gradient is normalized by scaling
// down to 8-bit equivalent before thresholding — because edge_thresh is in
// 8-bit scale. Effectively: bit-shift source 16-bit values to 8-bit scale,
// then run the same Sobel. This is cheap (one shift) and keeps threshold
// semantics the same.
// ---------------------------------------------------------------------------

static void compute_limit_map_16(
    const uint16_t* src, int sp_bytes,
    uint8_t* dst, int dp,
    int W, int H,
    int limit_flat, int limit_edge, int edge_thresh,
    int bitdepth)
{
    const int sp = sp_bytes / 2;  // pitch in samples
    const int shift = bitdepth - 8;  // down-shift to 8-bit for Sobel

    const int diff = limit_flat - limit_edge;
    const int inv_thresh = (edge_thresh > 0) ? (int)((65535 + edge_thresh / 2) / edge_thresh) : 0;

    for (int y = 0; y < H; ++y)
        memset(dst + (size_t)y * dp, (uint8_t)limit_flat, W);

    if (W < 3 || H < 3) return;

    __m128i vthresh     = _mm_set1_epi16((short)edge_thresh);
    __m128i vdiff       = _mm_set1_epi16((short)diff);
    __m128i vlimit_flat = _mm_set1_epi16((short)limit_flat);
    __m128i vinv        = _mm_set1_epi16((short)inv_thresh);
    __m128i vtwo        = _mm_set1_epi16(2);

    // Process 8 pixels per SSE iteration (vs 16 for 8-bit).
    for (int y = 1; y < H - 1; ++y) {
        const uint16_t* s0 = src + (size_t)(y - 1) * sp;
        const uint16_t* s1 = src + (size_t) y      * sp;
        const uint16_t* s2 = src + (size_t)(y + 1) * sp;
        uint8_t*        d  = dst + (size_t) y      * dp;

        int x = 1;
        for (; x + 9 <= W - 1; x += 8) {
            // Load 8 u16 at each position, shift down to 8-bit scale.
            auto load_shr = [shift](const uint16_t* p) -> __m128i {
                __m128i v = _mm_loadu_si128((const __m128i*)p);
                return _mm_srli_epi16(v, shift);
            };
            __m128i a = load_shr(s0 + x - 1);
            __m128i b = load_shr(s0 + x    );
            __m128i c = load_shr(s0 + x + 1);
            __m128i dv= load_shr(s1 + x - 1);
            __m128i f = load_shr(s1 + x + 1);
            __m128i g = load_shr(s2 + x - 1);
            __m128i h = load_shr(s2 + x    );
            __m128i i = load_shr(s2 + x + 1);

            __m128i Sx_pos = _mm_add_epi16(c, _mm_add_epi16(_mm_mullo_epi16(f,  vtwo), i));
            __m128i Sx_neg = _mm_add_epi16(a, _mm_add_epi16(_mm_mullo_epi16(dv, vtwo), g));
            __m128i Sx_abs = _mm_sub_epi16(
                _mm_max_epi16(Sx_pos, Sx_neg),
                _mm_min_epi16(Sx_pos, Sx_neg));

            __m128i Sy_pos = _mm_add_epi16(g, _mm_add_epi16(_mm_mullo_epi16(h, vtwo), i));
            __m128i Sy_neg = _mm_add_epi16(a, _mm_add_epi16(_mm_mullo_epi16(b, vtwo), c));
            __m128i Sy_abs = _mm_sub_epi16(
                _mm_max_epi16(Sy_pos, Sy_neg),
                _mm_min_epi16(Sy_pos, Sy_neg));

            __m128i grad = _mm_max_epi16(Sx_abs, Sy_abs);
            grad = _mm_min_epi16(grad, vthresh);

            __m128i grad_diff = _mm_mullo_epi16(grad, vdiff);
            __m128i reduction = _mm_mulhi_epu16(grad_diff, vinv);

            __m128i lim = _mm_sub_epi16(vlimit_flat, reduction);
            lim = _mm_max_epi16(lim, _mm_set1_epi16((short)limit_edge));
            lim = _mm_min_epi16(lim, vlimit_flat);

            __m128i lim_u8 = _mm_packus_epi16(lim, lim);
            _mm_storel_epi64((__m128i*)(d + x), lim_u8);
        }
        // Scalar tail — compute Sobel on shifted 8-bit values
        for (; x < W - 1; ++x) {
            int a = s0[x-1] >> shift, b = s0[x] >> shift, c = s0[x+1] >> shift;
            int dvs = s1[x-1] >> shift,                    fv = s1[x+1] >> shift;
            int g = s2[x-1] >> shift, h = s2[x] >> shift, i = s2[x+1] >> shift;
            int Sx = (c + 2*fv + i) - (a + 2*dvs + g);
            int Sy = (g + 2*h  + i) - (a + 2*b   + c);
            int grad = std::max(std::abs(Sx), std::abs(Sy));
            if (grad > edge_thresh) grad = edge_thresh;
            int reduction = (grad * diff + edge_thresh / 2) / edge_thresh;
            int lim = limit_flat - reduction;
            if (lim < limit_edge) lim = limit_edge;
            if (lim > limit_flat) lim = limit_flat;
            d[x] = (uint8_t)lim;
        }
    }
}

// ---------------------------------------------------------------------------
// DustPort filter class
// ---------------------------------------------------------------------------

class DustPort : public GenericVideoFilter {
public:
    DustPort(PClip _child,
             int _limit, int _mode,
             int _tr, int _temporal,
             std::vector<PClip> _prev_mc, std::vector<PClip> _next_mc,
             int _limit_edge, int _edge_thresh,
             const char* _planes_str,
             IScriptEnvironment* env)
        : GenericVideoFilter(_child),
          limit(_limit), mode(_mode),
          tr(_tr), temporal(_temporal),
          prev_mc(std::move(_prev_mc)), next_mc(std::move(_next_mc)),
          limit_edge(_limit_edge), edge_thresh(_edge_thresh),
          edge_aware(_limit_edge >= 0)
    {
        // Format detection
        is_yuy2 = vi.IsYUY2();
        is_planar_8 = (vi.IsYV12() || vi.IsYV16() || vi.IsYV24()) && vi.BitsPerComponent() == 8;
        is_planar_16 = !is_yuy2 && !is_planar_8 && vi.IsPlanar() && vi.IsYUV()
                       && vi.BitsPerComponent() >= 10 && vi.BitsPerComponent() <= 16;
        bitdepth = vi.BitsPerComponent();
        max_val = (bitdepth == 8) ? 255 : ((1 << bitdepth) - 1);

        if (!(is_yuy2 || is_planar_8 || is_planar_16))
            env->ThrowError("DustPort: unsupported format. Use YUY2, or planar YUV 8/10/12/14/16.");
        if (mode < 0 || mode > 3)
            env->ThrowError("DustPort: mode must be 0..3.");
        if (limit < 1 || limit > 64)
            env->ThrowError("DustPort: limit must be in 1..64 (8-bit scale).");
        if (tr < 1 || tr > 5)
            env->ThrowError("DustPort: tr must be in 1..5.");
        if (temporal != 0 && temporal != 1)
            env->ThrowError("DustPort: temporal must be 0 (mean) or 1 (median).");

        if (edge_aware) {
            if (is_yuy2)
                env->ThrowError("DustPort: edge-aware requires planar (not YUY2).");
            if (limit_edge < 0 || limit_edge > 64)
                env->ThrowError("DustPort: limit_edge must be 0..64 or -1 to disable.");
            if (limit_edge > limit)
                env->ThrowError("DustPort: limit_edge (%d) must be <= limit (%d).", limit_edge, limit);
            if (edge_thresh < 1 || edge_thresh > 255)
                env->ThrowError("DustPort: edge_thresh must be 1..255 (8-bit scale).");
        }

        // Parse 'planes' string: any combination of 'y','u','v' (case-insensitive).
        // Default "yuv" = process all planes (iter3 backwards-compatible behaviour).
        {
            const char* s = _planes_str ? _planes_str : "yuv";
            bool py=false, pu=false, pv=false;
            bool any_invalid=false;
            for (const char* p = s; *p; ++p) {
                char c = *p;
                if (c >= 'A' && c <= 'Z') c = (char)(c + 32);  // tolower
                if      (c == 'y') py = true;
                else if (c == 'u') pu = true;
                else if (c == 'v') pv = true;
                else if (c != ' ' && c != '\t') any_invalid = true;
            }
            if (any_invalid)
                env->ThrowError("DustPort: planes string '%s' contains invalid characters (only y/u/v allowed).", s);
            if (!py && !pu && !pv)
                env->ThrowError("DustPort: planes string '%s' selects no planes.", s);

            bool non_default = !(py && pu && pv);
            if (is_yuy2 && non_default)
                env->ThrowError(
                    "DustPort: 'planes' parameter is only supported for planar formats. "
                    "Either remove planes= from your call, or convert clip to YV16/YV24 first.");

            process_y = py;
            process_u = pu;
            process_v = pv;
        }

        if (!prev_mc.empty() && (int)prev_mc.size() != tr)
            env->ThrowError("DustPort: prev array size (%d) must match tr (%d).",
                            (int)prev_mc.size(), tr);
        if (!next_mc.empty() && (int)next_mc.size() != tr)
            env->ThrowError("DustPort: next array size (%d) must match tr (%d).",
                            (int)next_mc.size(), tr);

        auto check_mc = [&](const PClip& c, const char* tag, int offset) {
            const VideoInfo& mvi = c->GetVideoInfo();
            if (mvi.width != vi.width || mvi.height != vi.height)
                env->ThrowError("DustPort: %s[%d] size mismatch.", tag, offset);
            if (mvi.pixel_type != vi.pixel_type)
                env->ThrowError("DustPort: %s[%d] pixel_type mismatch.", tag, offset);
            if (mvi.num_frames < vi.num_frames)
                env->ThrowError("DustPort: %s[%d] shorter than main clip.", tag, offset);
        };
        for (int i = 0; i < (int)prev_mc.size(); ++i) check_mc(prev_mc[i], "prev", i);
        for (int i = 0; i < (int)next_mc.size(); ++i) check_mc(next_mc[i], "next", i);
    }

    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) override;

    int __stdcall SetCacheHints(int cachehints, int) override {
        return cachehints == CACHE_GET_MTMODE ? MT_SERIALIZED : 0;
    }

private:
    int limit;
    int mode;
    int tr;
    int temporal;
    std::vector<PClip> prev_mc;
    std::vector<PClip> next_mc;
    int limit_edge;
    int edge_thresh;
    bool edge_aware;

    bool process_y = true;
    bool process_u = true;
    bool process_v = true;

    bool is_yuy2 = false;
    bool is_planar_8 = false;
    bool is_planar_16 = false;
    int  bitdepth = 8;
    int  max_val = 255;

    std::vector<uint8_t> limit_map_buf[3];

    uint8_t* get_limit_map(int pi, int W, int H) {
        size_t need = (size_t)W * (size_t)H;
        if (limit_map_buf[pi].size() < need) limit_map_buf[pi].resize(need);
        return limit_map_buf[pi].data();
    }

    // YUY2 (plain C, unchanged)
    void spatial_yuy2(
        const uint8_t* src, int sp,
        uint8_t* dst, int dp,
        int W, int H);
    void spatiotemporal_yuy2(
        const uint8_t* const* frames, const int* pitches, int fcount, int centre_idx,
        uint8_t* dst, int dp,
        int W, int H);

    // Planar 8-bit
    void spatial_plane_8(
        const uint8_t* src, int sp,
        uint8_t* dst, int dp,
        int W, int H,
        const uint8_t* limit_map);
    void spatiotemporal_plane_8(
        const uint8_t* const* frames, const int* pitches, int fcount, int centre_idx,
        uint8_t* dst, int dp,
        int W, int H,
        const uint8_t* limit_map);

    // Planar 16-bit (10/12/14/16 stored in u16)
    void spatial_plane_16(
        const uint16_t* src, int sp_bytes,
        uint16_t* dst, int dp_bytes,
        int W, int H,
        const uint8_t* limit_map);
    void spatiotemporal_plane_16(
        const uint8_t* const* frames, const int* pitches, int fcount, int centre_idx,
        uint16_t* dst, int dp_bytes,
        int W, int H,
        const uint8_t* limit_map);
};

// ---------------------------------------------------------------------------
// GetFrame
// ---------------------------------------------------------------------------

PVideoFrame __stdcall DustPort::GetFrame(int n, IScriptEnvironment* env) {
    PVideoFrame src_frame = child->GetFrame(n, env);
    PVideoFrame dst_frame = env->NewVideoFrame(vi);

    const bool spatial_only = (mode == 3);
    std::vector<PVideoFrame> window;
    int centre_idx = 0;

    if (!spatial_only) {
        window.reserve(2 * tr + 1);
        for (int k = tr; k >= 1; --k) {
            if (!prev_mc.empty()) window.push_back(prev_mc[k-1]->GetFrame(n, env));
            else {
                int np = n - k; if (np < 0) np = 0;
                window.push_back(child->GetFrame(np, env));
            }
        }
        centre_idx = (int)window.size();
        window.push_back(src_frame);
        for (int k = 1; k <= tr; ++k) {
            if (!next_mc.empty()) window.push_back(next_mc[k-1]->GetFrame(n, env));
            else {
                int nx = n + k; if (nx >= vi.num_frames) nx = vi.num_frames - 1;
                window.push_back(child->GetFrame(nx, env));
            }
        }
    }

    if (is_yuy2) {
        if (spatial_only) {
            spatial_yuy2(
                src_frame->GetReadPtr(), src_frame->GetPitch(),
                dst_frame->GetWritePtr(), dst_frame->GetPitch(),
                vi.width, vi.height);
        } else {
            std::vector<const uint8_t*> ptrs(window.size());
            std::vector<int> pitches(window.size());
            for (size_t i = 0; i < window.size(); ++i) {
                ptrs[i] = window[i]->GetReadPtr();
                pitches[i] = window[i]->GetPitch();
            }
            spatiotemporal_yuy2(
                ptrs.data(), pitches.data(), (int)window.size(), centre_idx,
                dst_frame->GetWritePtr(), dst_frame->GetPitch(),
                vi.width, vi.height);
        }
        return dst_frame;
    }

    // Planar (8-bit or 16-bit)
    const int planes[3] = { PLANAR_Y, PLANAR_U, PLANAR_V };
    const bool do_plane[3] = { process_y, process_u, process_v };
    for (int pi = 0; pi < 3; ++pi) {
        int p = planes[pi];
        const int pw_samples = src_frame->GetRowSize(p) / (is_planar_16 ? 2 : 1);
        const int ph = src_frame->GetHeight(p);
        const int sp_pitch_bytes = src_frame->GetPitch(p);
        const int dp_pitch_bytes = dst_frame->GetPitch(p);
        const uint8_t* sp_ptr = src_frame->GetReadPtr(p);
        uint8_t*       dp_ptr = dst_frame->GetWritePtr(p);

        // Skip-and-copy path — plane not selected for processing
        if (!do_plane[pi]) {
            const int row_bytes = src_frame->GetRowSize(p);
            env->BitBlt(dp_ptr, dp_pitch_bytes, sp_ptr, sp_pitch_bytes, row_bytes, ph);
            continue;
        }

        const uint8_t* lmap_ptr = nullptr;
        if (edge_aware) {
            uint8_t* buf = get_limit_map(pi, pw_samples, ph);
            if (is_planar_8) {
                compute_limit_map_8(sp_ptr, sp_pitch_bytes, buf, pw_samples, pw_samples, ph,
                                    limit, limit_edge, edge_thresh);
            } else {
                compute_limit_map_16((const uint16_t*)sp_ptr, sp_pitch_bytes,
                                     buf, pw_samples, pw_samples, ph,
                                     limit, limit_edge, edge_thresh, bitdepth);
            }
            lmap_ptr = buf;
        }

        if (is_planar_8) {
            if (spatial_only) {
                spatial_plane_8(sp_ptr, sp_pitch_bytes, dp_ptr, dp_pitch_bytes,
                                pw_samples, ph, lmap_ptr);
            } else {
                std::vector<const uint8_t*> ptrs(window.size());
                std::vector<int> pitches(window.size());
                for (size_t i = 0; i < window.size(); ++i) {
                    ptrs[i]    = window[i]->GetReadPtr(p);
                    pitches[i] = window[i]->GetPitch(p);
                }
                spatiotemporal_plane_8(
                    ptrs.data(), pitches.data(), (int)ptrs.size(), centre_idx,
                    dp_ptr, dp_pitch_bytes,
                    pw_samples, ph,
                    lmap_ptr);
            }
        } else {
            // 16-bit planar
            if (spatial_only) {
                spatial_plane_16((const uint16_t*)sp_ptr, sp_pitch_bytes,
                                 (uint16_t*)dp_ptr, dp_pitch_bytes,
                                 pw_samples, ph, lmap_ptr);
            } else {
                std::vector<const uint8_t*> ptrs(window.size());
                std::vector<int> pitches(window.size());
                for (size_t i = 0; i < window.size(); ++i) {
                    ptrs[i]    = window[i]->GetReadPtr(p);
                    pitches[i] = window[i]->GetPitch(p);
                }
                spatiotemporal_plane_16(
                    ptrs.data(), pitches.data(), (int)ptrs.size(), centre_idx,
                    (uint16_t*)dp_ptr, dp_pitch_bytes,
                    pw_samples, ph,
                    lmap_ptr);
            }
        }
    }
    return dst_frame;
}

// ===========================================================================
// YUY2 kernels (unchanged, 8-bit only)
// ===========================================================================

void DustPort::spatial_yuy2(
    const uint8_t* src, int sp,
    uint8_t* dst, int dp,
    int W, int H)
{
    const int Wb = W * 2;
    memcpy(dst,                       src,                       Wb);
    memcpy(dst + (size_t)(H-1) * dp,  src + (size_t)(H-1) * sp,  Wb);

    for (int y = 1; y < H - 1; ++y) {
        const uint8_t* s0 = src + (size_t)(y - 1) * sp;
        const uint8_t* s1 = src + (size_t) y      * sp;
        const uint8_t* s2 = src + (size_t)(y + 1) * sp;
        uint8_t*       d  = dst + (size_t) y      * dp;

        d[0] = s1[0]; d[1] = s1[1]; d[Wb - 2] = s1[Wb - 2]; d[Wb - 1] = s1[Wb - 1];

        for (int x = 1; x < W - 1; ++x) {
            int bx = x * 2;
            int c  = s1[bx];
            int n4 = s0[bx] + s2[bx] + s1[bx - 2] + s1[bx + 2];
            int f  = (c * 2 + n4 + 3) / 6;
            d[bx]  = limit_clip8(c, f, limit);
        }
        for (int off = 1; off <= 3; off += 2) {
            for (int bx = off; bx < Wb - 1; bx += 4) {
                int c = s1[bx];
                int left  = (bx >= 4)     ? s1[bx - 4] : c;
                int right = (bx + 4 < Wb) ? s1[bx + 4] : c;
                int up = s0[bx], dn = s2[bx];
                int f = (c * 2 + left + right + up + dn + 3) / 6;
                d[bx] = limit_clip8(c, f, limit);
            }
        }
    }
}

void DustPort::spatiotemporal_yuy2(
    const uint8_t* const* frames, const int* pitches, int fcount, int centre_idx,
    uint8_t* dst, int dp,
    int W, int H)
{
    const int Wb = W * 2;
    int sw = (mode == 1) ? 1 : (mode == 2) ? 2 : 0;

    auto temporal_at = [&](int y, int bx) -> int {
        uint8_t vals[11];
        for (int i = 0; i < fcount; ++i)
            vals[i] = frames[i][(size_t)y * pitches[i] + bx];
        if (temporal == 1) return median_scalar8(vals, fcount);
        int sum = 0; for (int i = 0; i < fcount; ++i) sum += vals[i];
        return (sum + fcount / 2) / fcount;
    };

    for (int y = 0; y < H; ++y) {
        const uint8_t* c1 = frames[centre_idx] + (size_t)y * pitches[centre_idx];
        const bool edge_row = (y == 0 || y == H - 1);
        const uint8_t* c0 = edge_row ? c1 : frames[centre_idx] + (size_t)(y - 1) * pitches[centre_idx];
        const uint8_t* c2 = edge_row ? c1 : frames[centre_idx] + (size_t)(y + 1) * pitches[centre_idx];
        uint8_t*       d  = dst + (size_t)y * dp;

        for (int bx = 0; bx < Wb; ++bx) {
            int src_val = c1[bx];
            int tm = temporal_at(y, bx);
            int out;
            bool edge_col = (bx < 2 || bx >= Wb - 2);
            if (sw == 0 || edge_row || edge_col) {
                out = tm;
            } else {
                bool is_luma = ((bx & 1) == 0);
                int left, right;
                if (is_luma) { left = c1[bx - 2]; right = c1[bx + 2]; }
                else {
                    left  = (bx >= 4)     ? c1[bx - 4] : src_val;
                    right = (bx + 4 < Wb) ? c1[bx + 4] : src_val;
                }
                int n4 = c0[bx] + c2[bx] + left + right;
                int sp = (src_val * 2 + n4 + 3) / 6;
                out = (tm * (3 - sw) + sp * sw + 1) / 3;
            }
            d[bx] = limit_clip8(src_val, out, limit);
        }
    }
}

// ===========================================================================
// Planar 8-bit kernels (from commit 2a, same logic)
// ===========================================================================

static inline __m128i load_vlimit_8(
    const uint8_t* limit_map, int map_row_off, int x, __m128i vlimit_scalar)
{
    if (limit_map) return _mm_loadu_si128((const __m128i*)(limit_map + map_row_off + x));
    return vlimit_scalar;
}
static inline int load_slimit_8(
    const uint8_t* limit_map, int map_row_off, int x, int limit_scalar)
{
    if (limit_map) return limit_map[map_row_off + x];
    return limit_scalar;
}

void DustPort::spatial_plane_8(
    const uint8_t* src, int sp,
    uint8_t* dst, int dp,
    int W, int H,
    const uint8_t* limit_map)
{
    if (W < 3 || H < 3) {
        for (int y = 0; y < H; ++y) memcpy(dst + (size_t)y * dp, src + (size_t)y * sp, W);
        return;
    }
    __m128i vlimit_s = _mm_set1_epi8((char)limit);
    const int map_pitch = W;

    memcpy(dst,                       src,                       W);
    memcpy(dst + (size_t)(H-1) * dp,  src + (size_t)(H-1) * sp,  W);

    for (int y = 1; y < H - 1; ++y) {
        const uint8_t* s0 = src + (size_t)(y - 1) * sp;
        const uint8_t* s1 = src + (size_t) y      * sp;
        const uint8_t* s2 = src + (size_t)(y + 1) * sp;
        uint8_t*       d  = dst + (size_t) y      * dp;
        const int mapoff  = limit_map ? y * map_pitch : 0;

        d[0] = s1[0]; d[W - 1] = s1[W - 1];

        int x = 1;
        for (; x + 16 <= W - 1; x += 16) {
            __m128i up     = _mm_loadu_si128((const __m128i*)(s0 + x));
            __m128i dn     = _mm_loadu_si128((const __m128i*)(s2 + x));
            __m128i left   = _mm_loadu_si128((const __m128i*)(s1 + x - 1));
            __m128i right  = _mm_loadu_si128((const __m128i*)(s1 + x + 1));
            __m128i centre = _mm_loadu_si128((const __m128i*)(s1 + x));
            __m128i filt   = spatial_blend_sse2_8(up, dn, left, right, centre);
            __m128i vlim   = load_vlimit_8(limit_map, mapoff, x, vlimit_s);
            __m128i out    = limit_clip_sse2_8(centre, filt, vlim);
            _mm_storeu_si128((__m128i*)(d + x), out);
        }
        for (; x < W - 1; ++x) {
            int c  = s1[x];
            int n4 = s0[x] + s2[x] + s1[x-1] + s1[x+1];
            int f  = (c * 2 + n4 + 3) / 6;
            d[x]   = limit_clip8(c, f, load_slimit_8(limit_map, mapoff, x, limit));
        }
    }
}

void DustPort::spatiotemporal_plane_8(
    const uint8_t* const* frames, const int* pitches, int fcount, int centre_idx,
    uint8_t* dst, int dp,
    int W, int H,
    const uint8_t* limit_map)
{
    __m128i vlimit_s = _mm_set1_epi8((char)limit);
    int sw = (mode == 1) ? 1 : (mode == 2) ? 2 : 0;
    const int map_pitch = W;

    auto do_row = [&](int y, bool edge_row) {
        const uint8_t* c1  = frames[centre_idx] + (size_t)y * pitches[centre_idx];
        const uint8_t* c0r = edge_row ? c1 : frames[centre_idx] + (size_t)(y - 1) * pitches[centre_idx];
        const uint8_t* c2r = edge_row ? c1 : frames[centre_idx] + (size_t)(y + 1) * pitches[centre_idx];
        uint8_t*       d   = dst + (size_t)y * dp;
        const int      mapoff = limit_map ? y * map_pitch : 0;

        auto scalar_one = [&](int x) {
            uint8_t vals[11];
            for (int i = 0; i < fcount; ++i)
                vals[i] = frames[i][(size_t)y * pitches[i] + x];
            int src_val = c1[x];
            int tm;
            if (temporal == 1) tm = median_scalar8(vals, fcount);
            else {
                int sum = 0; for (int i = 0; i < fcount; ++i) sum += vals[i];
                tm = (sum + fcount / 2) / fcount;
            }
            int out_val;
            if (sw == 0 || edge_row || x == 0 || x == W - 1) {
                out_val = tm;
            } else {
                int n4 = c0r[x] + c2r[x] + c1[x-1] + c1[x+1];
                int sp = (src_val * 2 + n4 + 3) / 6;
                out_val = (tm * (3 - sw) + sp * sw + 1) / 3;
            }
            d[x] = limit_clip8(src_val, out_val, load_slimit_8(limit_map, mapoff, x, limit));
        };

        if (edge_row) {
            int col = 0;
            __m128i buf[11];
            for (; col + 16 <= W; col += 16) {
                for (int i = 0; i < fcount; ++i) {
                    buf[i] = _mm_loadu_si128((const __m128i*)(
                        frames[i] + (size_t)y * pitches[i] + col));
                }
                __m128i vc = buf[centre_idx];
                __m128i vtm = (temporal == 1)
                    ? temporal_median_sse2_8(buf, fcount)
                    : temporal_mean_sse2_8(buf, fcount);
                __m128i vlim = load_vlimit_8(limit_map, mapoff, col, vlimit_s);
                __m128i out  = limit_clip_sse2_8(vc, vtm, vlim);
                _mm_storeu_si128((__m128i*)(d + col), out);
            }
            for (; col < W; ++col) scalar_one(col);
            return;
        }

        scalar_one(0);
        scalar_one(W - 1);

        int x = 1;
        __m128i buf[11];
        for (; x + 16 <= W - 1; x += 16) {
            for (int i = 0; i < fcount; ++i) {
                buf[i] = _mm_loadu_si128((const __m128i*)(
                    frames[i] + (size_t)y * pitches[i] + x));
            }
            __m128i vc = buf[centre_idx];
            __m128i vtm = (temporal == 1)
                ? temporal_median_sse2_8(buf, fcount)
                : temporal_mean_sse2_8(buf, fcount);

            __m128i filt;
            if (sw == 0) filt = vtm;
            else {
                __m128i up    = _mm_loadu_si128((const __m128i*)(c0r + x));
                __m128i dn    = _mm_loadu_si128((const __m128i*)(c2r + x));
                __m128i left  = _mm_loadu_si128((const __m128i*)(c1  + x - 1));
                __m128i right = _mm_loadu_si128((const __m128i*)(c1  + x + 1));
                __m128i vsp   = spatial_blend_sse2_8(up, dn, left, right, vc);
                if (sw == 1) filt = _mm_avg_epu8(vtm, vsp);
                else {
                    __m128i b = _mm_avg_epu8(vtm, vsp);
                    filt      = _mm_avg_epu8(b,   vsp);
                }
            }
            __m128i vlim = load_vlimit_8(limit_map, mapoff, x, vlimit_s);
            __m128i out  = limit_clip_sse2_8(vc, filt, vlim);
            _mm_storeu_si128((__m128i*)(d + x), out);
        }
        for (; x < W - 1; ++x) scalar_one(x);
    };

    do_row(0, true);
    for (int y = 1; y < H - 1; ++y) do_row(y, false);
    if (H > 1) do_row(H - 1, true);
}

// ===========================================================================
// Planar 16-bit kernels (SSE4.1)
// ===========================================================================
//
// Storage: u16 per sample. Values occupy range 0..max_val (max_val = (1<<bitdepth)-1).
// 'limit' parameter is in 8-bit scale; we scale it up by (1 << (bitdepth-8)) for
// 16-bit operations. limit_map (if present) contains u8 values (8-bit scale),
// which we unpack to u16 and scale up per row.
//
// Process 8 u16 pixels per SSE iteration.
// ===========================================================================

// Load 8 u16 limit values from the 8-bit limit map and scale up by (1<<shift).
static inline __m128i load_vlimit_16(
    const uint8_t* limit_map, int map_row_off, int x,
    __m128i vlimit_scalar_16, int shift)
{
    if (!limit_map) return vlimit_scalar_16;
    __m128i lo8 = _mm_loadl_epi64((const __m128i*)(limit_map + map_row_off + x));
    __m128i lo16 = _mm_unpacklo_epi8(lo8, _mm_setzero_si128());
    return _mm_slli_epi16(lo16, shift);
}
static inline int load_slimit_16(
    const uint8_t* limit_map, int map_row_off, int x, int limit_scalar_native)
{
    if (!limit_map) return limit_scalar_native;
    // NOTE: caller must already handle scaling of map value by 1<<shift
    return limit_map[map_row_off + x];
}

void DustPort::spatial_plane_16(
    const uint16_t* src, int sp_bytes,
    uint16_t* dst, int dp_bytes,
    int W, int H,
    const uint8_t* limit_map)
{
    const int sp = sp_bytes / 2;
    const int dp = dp_bytes / 2;
    const int shift = bitdepth - 8;
    const int limit_native = limit << shift;
    __m128i vlimit_s16 = _mm_set1_epi16((short)limit_native);
    __m128i vmax       = _mm_set1_epi16((short)max_val);
    const int map_pitch = W;

    if (W < 3 || H < 3) {
        for (int y = 0; y < H; ++y)
            memcpy(dst + (size_t)y * dp, src + (size_t)y * sp, (size_t)W * 2);
        return;
    }
    // Row 0 and H-1 — copy
    memcpy(dst,                       src,                       (size_t)W * 2);
    memcpy(dst + (size_t)(H-1) * dp,  src + (size_t)(H-1) * sp,  (size_t)W * 2);

    for (int y = 1; y < H - 1; ++y) {
        const uint16_t* s0 = src + (size_t)(y - 1) * sp;
        const uint16_t* s1 = src + (size_t) y      * sp;
        const uint16_t* s2 = src + (size_t)(y + 1) * sp;
        uint16_t*       d  = dst + (size_t) y      * dp;
        const int mapoff   = limit_map ? y * map_pitch : 0;

        d[0] = s1[0]; d[W - 1] = s1[W - 1];

        int x = 1;
        for (; x + 8 <= W - 1; x += 8) {
            __m128i up     = _mm_loadu_si128((const __m128i*)(s0 + x));
            __m128i dn     = _mm_loadu_si128((const __m128i*)(s2 + x));
            __m128i left   = _mm_loadu_si128((const __m128i*)(s1 + x - 1));
            __m128i right  = _mm_loadu_si128((const __m128i*)(s1 + x + 1));
            __m128i centre = _mm_loadu_si128((const __m128i*)(s1 + x));
            __m128i filt   = spatial_blend_sse_16(up, dn, left, right, centre);
            __m128i vlim   = load_vlimit_16(limit_map, mapoff, x, vlimit_s16, shift);
            __m128i out    = limit_clip_sse_16(centre, filt, vlim, vmax);
            _mm_storeu_si128((__m128i*)(d + x), out);
        }
        for (; x < W - 1; ++x) {
            int c  = s1[x];
            int n4 = s0[x] + s2[x] + s1[x-1] + s1[x+1];
            int f  = (c * 2 + n4 + 3) / 6;
            int lim_here = limit_map
                ? (limit_map[mapoff + x] << shift)
                : limit_native;
            d[x] = limit_clip16(c, f, lim_here, max_val);
        }
    }
}

void DustPort::spatiotemporal_plane_16(
    const uint8_t* const* frames, const int* pitches, int fcount, int centre_idx,
    uint16_t* dst, int dp_bytes,
    int W, int H,
    const uint8_t* limit_map)
{
    const int dp = dp_bytes / 2;
    const int shift = bitdepth - 8;
    const int limit_native = limit << shift;
    __m128i vlimit_s16 = _mm_set1_epi16((short)limit_native);
    __m128i vmax       = _mm_set1_epi16((short)max_val);
    int sw = (mode == 1) ? 1 : (mode == 2) ? 2 : 0;
    const int map_pitch = W;

    // Samples-pitch arrays — pitches[i] is bytes, convert to u16 samples
    auto row16 = [&](int frame_i, int y) -> const uint16_t* {
        return (const uint16_t*)(frames[frame_i] + (size_t)y * pitches[frame_i]);
    };

    auto do_row = [&](int y, bool edge_row) {
        const uint16_t* c1  = row16(centre_idx, y);
        const uint16_t* c0r = edge_row ? c1 : row16(centre_idx, y - 1);
        const uint16_t* c2r = edge_row ? c1 : row16(centre_idx, y + 1);
        uint16_t*       d   = dst + (size_t)y * dp;
        const int mapoff    = limit_map ? y * map_pitch : 0;

        auto scalar_one = [&](int x) {
            uint16_t vals[11];
            for (int i = 0; i < fcount; ++i) vals[i] = row16(i, y)[x];
            int src_val = c1[x];
            int tm;
            if (temporal == 1) tm = median_scalar16(vals, fcount);
            else {
                int sum = 0; for (int i = 0; i < fcount; ++i) sum += vals[i];
                tm = (sum + fcount / 2) / fcount;
            }
            int out_val;
            if (sw == 0 || edge_row || x == 0 || x == W - 1) {
                out_val = tm;
            } else {
                int n4 = c0r[x] + c2r[x] + c1[x-1] + c1[x+1];
                int sp_v = (src_val * 2 + n4 + 3) / 6;
                out_val = (tm * (3 - sw) + sp_v * sw + 1) / 3;
            }
            int lim_here = limit_map ? (limit_map[mapoff + x] << shift) : limit_native;
            d[x] = limit_clip16(src_val, out_val, lim_here, max_val);
        };

        if (edge_row) {
            int col = 0;
            __m128i buf[11];
            for (; col + 8 <= W; col += 8) {
                for (int i = 0; i < fcount; ++i) {
                    buf[i] = _mm_loadu_si128((const __m128i*)(row16(i, y) + col));
                }
                __m128i vc = buf[centre_idx];
                __m128i vtm = (temporal == 1)
                    ? temporal_median_sse_16(buf, fcount)
                    : temporal_mean_sse_16(buf, fcount);
                __m128i vlim = load_vlimit_16(limit_map, mapoff, col, vlimit_s16, shift);
                __m128i out  = limit_clip_sse_16(vc, vtm, vlim, vmax);
                _mm_storeu_si128((__m128i*)(d + col), out);
            }
            for (; col < W; ++col) scalar_one(col);
            return;
        }

        scalar_one(0);
        scalar_one(W - 1);

        int x = 1;
        __m128i buf[11];
        for (; x + 8 <= W - 1; x += 8) {
            for (int i = 0; i < fcount; ++i) {
                buf[i] = _mm_loadu_si128((const __m128i*)(row16(i, y) + x));
            }
            __m128i vc = buf[centre_idx];
            __m128i vtm = (temporal == 1)
                ? temporal_median_sse_16(buf, fcount)
                : temporal_mean_sse_16(buf, fcount);

            __m128i filt;
            if (sw == 0) filt = vtm;
            else {
                __m128i up    = _mm_loadu_si128((const __m128i*)(c0r + x));
                __m128i dn    = _mm_loadu_si128((const __m128i*)(c2r + x));
                __m128i left  = _mm_loadu_si128((const __m128i*)(c1  + x - 1));
                __m128i right = _mm_loadu_si128((const __m128i*)(c1  + x + 1));
                __m128i vsp   = spatial_blend_sse_16(up, dn, left, right, vc);
                if (sw == 1) filt = avg_epu16(vtm, vsp);
                else {
                    __m128i b = avg_epu16(vtm, vsp);
                    filt      = avg_epu16(b,   vsp);
                }
            }
            __m128i vlim = load_vlimit_16(limit_map, mapoff, x, vlimit_s16, shift);
            __m128i out  = limit_clip_sse_16(vc, filt, vlim, vmax);
            _mm_storeu_si128((__m128i*)(d + x), out);
        }
        for (; x < W - 1; ++x) scalar_one(x);
    };

    do_row(0, true);
    for (int y = 1; y < H - 1; ++y) do_row(y, false);
    if (H > 1) do_row(H - 1, true);
}

// ---------------------------------------------------------------------------
// Arg parsing + callbacks
// ---------------------------------------------------------------------------

static std::vector<PClip> extract_clip_array(const AVSValue& v) {
    std::vector<PClip> out;
    if (!v.Defined()) return out;
    if (v.IsArray()) {
        int n = v.ArraySize();
        out.reserve(n);
        for (int i = 0; i < n; ++i) out.push_back(v[i].AsClip());
    } else if (v.IsClip()) {
        out.push_back(v.AsClip());
    }
    return out;
}

static AVSValue __cdecl Create_FaeryDust(AVSValue a, void*, IScriptEnvironment* env) {
    return new DustPort(a[0].AsClip(),
                        a[1].AsInt(2), a[2].AsInt(0),
                        a[5].AsInt(1), a[6].AsInt(0),
                        extract_clip_array(a[3]), extract_clip_array(a[4]),
                        a[7].AsInt(-1), a[8].AsInt(32),
                        a[9].AsString("yuv"), env);
}
static AVSValue __cdecl Create_PixieDust(AVSValue a, void*, IScriptEnvironment* env) {
    return new DustPort(a[0].AsClip(),
                        a[1].AsInt(5), a[2].AsInt(1),
                        a[5].AsInt(1), a[6].AsInt(0),
                        extract_clip_array(a[3]), extract_clip_array(a[4]),
                        a[7].AsInt(-1), a[8].AsInt(32),
                        a[9].AsString("yuv"), env);
}
static AVSValue __cdecl Create_GoldDust(AVSValue a, void*, IScriptEnvironment* env) {
    return new DustPort(a[0].AsClip(),
                        a[1].AsInt(8), a[2].AsInt(2),
                        a[5].AsInt(1), a[6].AsInt(0),
                        extract_clip_array(a[3]), extract_clip_array(a[4]),
                        a[7].AsInt(-1), a[8].AsInt(32),
                        a[9].AsString("yuv"), env);
}
static AVSValue __cdecl Create_SpaceDust(AVSValue a, void*, IScriptEnvironment* env) {
    return new DustPort(a[0].AsClip(),
                        a[1].AsInt(6), a[2].AsInt(3),
                        1, 0,
                        std::vector<PClip>(), std::vector<PClip>(),
                        a[3].AsInt(-1), a[4].AsInt(32),
                        a[5].AsString("yuv"), env);
}

const AVS_Linkage* AVS_linkage = nullptr;

extern "C" __declspec(dllexport)
const char* __stdcall AvisynthPluginInit3(
    IScriptEnvironment* env, const AVS_Linkage* const vectors)
{
    AVS_linkage = vectors;
    const char* sig_full  = "c[limit]i[mode]i[prev]c+[next]c+[tr]i[temporal]i[limit_edge]i[edge_thresh]i[planes]s";
    const char* sig_space = "c[limit]i[mode]i[limit_edge]i[edge_thresh]i[planes]s";
    env->AddFunction("FaeryDustPort", sig_full,  Create_FaeryDust, 0);
    env->AddFunction("PixieDustPort", sig_full,  Create_PixieDust, 0);
    env->AddFunction("GoldDustPort",  sig_full,  Create_GoldDust,  0);
    env->AddFunction("SpaceDustPort", sig_space, Create_SpaceDust, 0);
    return "DustPort iter3/commit2c — 8/10/12/14/16-bit planar + YUY2, tr/median/edge-aware/planes (SSE2+SSE4.1).";
}
