// SPDX-License-Identifier: MIT
// Median selection networks for 3, 5, 7, 9 and 11 values: Batcher's odd-even merge sort, pruned to the comparators
// the middle output depends on. Each comparator (i, j) puts min into i and max into j. All verified exhaustively
// with the 0-1 principle; test/gen_median_nets.py generates and verifies the tables.
#pragma once

#include <cstdint>

namespace dust {

inline constexpr int8_t kMedian3[][2] = {{0, 1}, {0, 2}, {1, 2}};
inline constexpr int8_t kMedian5[][2] = {{0, 1}, {2, 3}, {0, 2}, {1, 3}, {1, 2}, {0, 4}, {2, 4}, {1, 2}};
inline constexpr int8_t kMedian7[][2] = {{0, 1}, {2, 3}, {0, 2}, {1, 3}, {1, 2}, {4, 5}, {4, 6}, {5, 6}, {0, 4}, {2, 6}, {2, 4}, {1, 5}, {3, 5}, {3, 4}};
inline constexpr int8_t kMedian9[][2] = {{0, 1}, {2, 3}, {0, 2}, {1, 3}, {1, 2}, {4, 5}, {6, 7}, {4, 6}, {5, 7}, {5, 6}, {0, 4}, {2, 6}, {2, 4}, {1, 5}, {3, 7}, {3, 5}, {1, 2}, {3, 4}, {5, 6}, {0, 8}, {4, 8}, {2, 4}, {3, 5}, {3, 4}};
inline constexpr int8_t kMedian11[][2] = {{0, 1}, {2, 3}, {0, 2}, {1, 3}, {1, 2}, {4, 5}, {6, 7}, {4, 6}, {5, 7}, {5, 6}, {0, 4}, {2, 6}, {2, 4}, {1, 5}, {3, 7}, {3, 5}, {1, 2}, {3, 4}, {5, 6}, {8, 9}, {8, 10}, {9, 10}, {9, 10}, {0, 8}, {4, 8}, {2, 10}, {6, 10}, {6, 8}, {1, 9}, {5, 9}, {3, 5}, {5, 6}};

struct MedianNet {
  const int8_t (*pairs)[2];
  int count;
};

static inline MedianNet medianNet(int n) {
  switch (n) {
    case 3: return {kMedian3, int(sizeof(kMedian3) / sizeof(kMedian3[0]))};
    case 5: return {kMedian5, int(sizeof(kMedian5) / sizeof(kMedian5[0]))};
    case 7: return {kMedian7, int(sizeof(kMedian7) / sizeof(kMedian7[0]))};
    case 9: return {kMedian9, int(sizeof(kMedian9) / sizeof(kMedian9[0]))};
    default: return {kMedian11, int(sizeof(kMedian11) / sizeof(kMedian11[0]))};
  }
}

}  // namespace dust
