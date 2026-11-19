"""Admission metrics for concrete GEMM schedules, separate from tile padding."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass


class GemmScheduleRejected(ValueError):
    """A schedule has poor output tails or compiled local-memory allocation."""


@dataclass(frozen=True)
class GemmScheduleMetrics:
    logical_tiles: int
    output_slots: int
    grid_ctas: int
    fixed_grid: bool

    @property
    def output_tail_efficiency(self) -> float:
        return self.logical_tiles / self.output_slots

    def validate(self) -> None:
        if 100 * self.logical_tiles < 98 * self.output_slots:
            raise GemmScheduleRejected(
                f"output tile slots: {self.logical_tiles}/{self.output_slots} "
                f"({self.output_tail_efficiency:.2%} efficiency, below 98%)"
            )


def gemm_schedule_metrics(config: dict[str, object], m: int, n: int,
                          batch: int = 1) -> GemmScheduleMetrics:
    """Use realized strided group bounds, including shortened output loops.

    Fixed grids override grouping. Batch multiplies launch supply and slots;
    it does not change the per-batch loop trip count. Edge padding is separate.
    """
    if min(m, n, batch) <= 0:
        raise ValueError("GEMM dimensions and batch must be positive")
    if config.get("transpose_output"):
        m, n = n, m

    def positive(key: str, default: int = 1) -> int:
        value = config.get(key, default)
        if type(value) is not int or value <= 0:
            raise ValueError(f"{key} must be a positive integer")
        return value

    tm, tn = positive("output_m_tile"), positive("output_n_tile")
    mt, nt = math.ceil(m / tm), math.ceil(n / tn)
    fixed_count = config.get("output_cta_count")
    if fixed_count is None:
        fixed_count = config.get("output_grid_size")
    fixed = fixed_count is not None
    if fixed:
        if type(fixed_count) is not int or fixed_count <= 0:
            raise ValueError("output grid size must be a positive integer")
        grid = fixed_count
        iterations = math.ceil(mt * nt / grid)
    else:
        mg = positive("output_m_tiles_per_cta", positive("output_tiles_per_cta"))
        ng = positive("output_n_tiles_per_cta")
        cm, cn = math.ceil(mt / mg), math.ceil(nt / ng)
        grid = cm * cn
        iterations = math.ceil(m / (cm * tm)) * math.ceil(n / (cn * tn))
    return GemmScheduleMetrics(batch * mt * nt, batch * grid * iterations,
                               batch * grid, fixed)


def validate_compiled_gemm(resources: dict[str, int]) -> None:
    """Check the actual loaded kernel before caching or executing it.

    Generated GEMMs do not declare local arrays or use a call stack. A local
    allocation therefore indicates compiler spill backing; reject it instead
    of inferring spills from the register count. Residency comes from CUDA,
    including cluster occupancy, rather than a shared-memory size estimate.
    """
    local = resources["local_bytes_per_thread"]
    if local:
        raise GemmScheduleRejected(f"compiled GEMM uses {local} local bytes/thread (spill backing)")
    capacity = resources["resident_ctas_on_device"]
    if capacity <= 0:
        raise GemmScheduleRejected("compiled GEMM has no resident CTA capacity")
    grid = resources["grid_ctas"]
    wave = grid / (math.ceil(grid / capacity) * capacity)
    if wave < .70:
        warnings.warn(
            f"GEMM wave utilization {wave:.1%} is below 70% "
            f"({grid} CTAs, {capacity:g} resident CTA slots); "
            "retain only after measuring complete GEMM latency",
            RuntimeWarning,
            stacklevel=2,
        )
