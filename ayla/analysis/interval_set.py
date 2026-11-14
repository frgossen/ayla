"""Finite unions of half-open integer intervals used by tile dependencies.

Shapes are specialized before analysis. Symbolic extents must first pass through
Ayla's shape specialization; no ordering of unknown expressions is assumed.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable


@dataclass(frozen=True)
class IntervalSet:
    intervals: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        previous: int | None = None
        for lo, hi in self.intervals:
            if (
                type(lo) is not int
                or type(hi) is not int
                or lo >= hi
                or previous is not None
                and lo <= previous
            ):
                raise ValueError(
                    "interval sets must contain normalized nonempty integer intervals"
                )
            previous = hi

    @classmethod
    def build(cls, intervals: Iterable[tuple[int, int]]) -> IntervalSet:
        merged: list[tuple[int, int]] = []
        intervals = list(intervals)
        for lo, hi in intervals:
            if type(lo) is not int or type(hi) is not int or lo > hi:
                raise ValueError("interval bounds must be ordered integers")
        for lo, hi in sorted(intervals):
            if lo == hi:
                continue
            if merged and lo <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(hi, merged[-1][1]))
            else:
                merged.append((lo, hi))
        return cls(tuple(merged))

    def union(self, other: IntervalSet) -> IntervalSet:
        return self.build((*self.intervals, *other.intervals))

    def intersection(self, other: IntervalSet) -> IntervalSet:
        result: list[tuple[int, int]] = []
        i = j = 0
        while i < len(self.intervals) and j < len(other.intervals):
            a, b = self.intervals[i], other.intervals[j]
            lo, hi = max(a[0], b[0]), min(a[1], b[1])
            if lo < hi:
                result.append((lo, hi))
            if a[1] <= b[1]:
                i += 1
            else:
                j += 1
        return self.build(result)

    def difference(self, other: IntervalSet) -> IntervalSet:
        result: list[tuple[int, int]] = []
        j = 0
        for lo, hi in self.intervals:
            while j < len(other.intervals) and other.intervals[j][1] <= lo:
                j += 1
            k = j
            while k < len(other.intervals) and other.intervals[k][0] < hi:
                start, stop = other.intervals[k]
                if start > lo:
                    result.append((lo, min(start, hi)))
                lo = max(lo, stop)
                if lo >= hi:
                    break
                k += 1
            if lo < hi:
                result.append((lo, hi))
        return self.build(result)

    def overlaps(self, other: IntervalSet) -> bool:
        if not self.intervals or not other.intervals:
            return False
        if (
            self.intervals[-1][1] <= other.intervals[0][0]
            or other.intervals[-1][1] <= self.intervals[0][0]
        ):
            return False
        i = j = 0
        while i < len(self.intervals) and j < len(other.intervals):
            a, b = self.intervals[i], other.intervals[j]
            if max(a[0], b[0]) < min(a[1], b[1]):
                return True
            if a[1] <= b[1]:
                i += 1
            else:
                j += 1
        return False

    def shifted(self, offset: int) -> IntervalSet:
        return self.build((lo + offset, hi + offset) for lo, hi in self.intervals)
