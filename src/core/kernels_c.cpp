// SPDX-License-Identifier: MIT
// C reference kernels: the formulas of ALGORITHM.md (scalar.hpp), applied to every sample.
#include "kernels.hpp"
#include "scalar.hpp"

namespace dust {
namespace {

template <typename T>
void filterT(const FilterArgs& a) {
  for (int y = 0; y < a.height; y++) {
    const Rows r = rowsAt(a, y);
    for (int x = 0; x < a.width; x++) filterPixel<T>(a, r, x);
  }
}

void filterC(const FilterArgs& a) {
  if (a.bits == 8)
    filterT<uint8_t>(a);
  else
    filterT<uint16_t>(a);
}

template <typename T>
void limitMapT(const LimitMapArgs& a) {
  for (int y = 0; y < a.height; y++) {
    const uint8_t* r0 = a.src + mirror(y - 1, a.height) * a.stride;
    const uint8_t* r1 = a.src + y * a.stride;
    const uint8_t* r2 = a.src + mirror(y + 1, a.height) * a.stride;
    uint8_t* m = a.map + ptrdiff_t(y) * a.width;
    for (int x = 0; x < a.width; x++) m[x] = limitPixel<T>(a, r0, r1, r2, x);
  }
}

void limitMapC(const LimitMapArgs& a) {
  if (a.bits == 8)
    limitMapT<uint8_t>(a);
  else
    limitMapT<uint16_t>(a);
}

template <typename T>
void sceneSadT(const uint8_t* a, ptrdiff_t sa, const uint8_t* b, ptrdiff_t sb, int width, int height, int xstep,
               int bits, uint64_t* sum, uint64_t* count) {
  const int shift = bits - 8;
  uint64_t s = 0, n = 0;
  for (int y = 0; y < height; y += 2) {
    const uint8_t* ra = a + y * sa;
    const uint8_t* rb = b + y * sb;
    for (int x = 0; x < width; x += 2) {
      s += uint64_t(std::abs((loadSample<T>(ra, x, xstep) >> shift) - (loadSample<T>(rb, x, xstep) >> shift)));
      n++;
    }
  }
  *sum = s;
  *count = n;
}

void sceneSadC(const uint8_t* a, ptrdiff_t sa, const uint8_t* b, ptrdiff_t sb, int width, int height, int xstep,
               int bits, uint64_t* sum, uint64_t* count) {
  if (bits == 8)
    sceneSadT<uint8_t>(a, sa, b, sb, width, height, xstep, bits, sum, count);
  else
    sceneSadT<uint16_t>(a, sa, b, sb, width, height, xstep, bits, sum, count);
}

}  // namespace

const Kernels& kernelsC() {
  static const Kernels k = {filterC, limitMapC, sceneSadC};
  return k;
}

}  // namespace dust
