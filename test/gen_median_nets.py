#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Generates the median selection networks of src/core/median_nets.hpp and verifies them with the 0-1 principle.

Batcher's odd-even merge sort for the next power of two, comparators touching padding wires dropped, then pruned to
the comparators the middle output depends on. Prints the table lines of the header.
"""
import itertools


def batcher(n):
    p = 1
    while p < n:
        p *= 2
    out = []

    def merge(lo, hi, r):
        step = r * 2
        if step < hi - lo:
            merge(lo, hi, step)
            merge(lo + r, hi, step)
            for i in range(lo + r, hi - r, step):
                out.append((i, i + r))
        else:
            out.append((lo, lo + r))

    def sort(lo, hi):
        if hi - lo >= 1:
            mid = lo + (hi - lo) // 2
            sort(lo, mid)
            sort(mid + 1, hi)
            merge(lo, hi, 1)

    sort(0, p - 1)
    return [(i, j) for i, j in out if j < n]


def prune(net, n):
    need = {n // 2}
    keep = []
    for i, j in reversed(net):
        if i in need or j in need:
            keep.append((i, j))
            need |= {i, j}
    return keep[::-1]


def is_median(net, n):
    for bits in itertools.product((0, 1), repeat=n):
        v = list(bits)
        for i, j in net:
            if v[i] > v[j]:
                v[i], v[j] = v[j], v[i]
        if v[n // 2] != sorted(bits)[n // 2]:
            return False
    return True


def main():
    for n in (3, 5, 7, 9, 11):
        net = prune(batcher(n), n)
        assert is_median(net, n), n
        pairs = ", ".join(f"{{{i}, {j}}}" for i, j in net)
        print(f"inline constexpr int8_t kMedian{n}[][2] = {{{pairs}}};")


if __name__ == "__main__":
    main()
