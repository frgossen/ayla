"""Blackwell GEMMs: the configurable GEMM schedule on a tcgen05 target."""

from copy import deepcopy
from typing import Any

from ..gpu_dialect import CollectiveAxis, ModuleOp, gpu_dialect
from ..pipelines.tile_matmul_pipeline import _schedule_matmul


def blackwell_tcgen05_gemm_module(
    *,
    problem_size: int = 4096,
    m: int | None = None,
    n: int | None = None,
    k: int | None = None,
    output_n_tile: int = 256,
    operand_k_tile: int = 64,
    operand_stages: int = 3,
    cyclic_operand_pipeline: bool = False,
    dtype: str = "bf16",
    output_grid_size: int | None = None,
    output_grid_fast_axis: str = "m",
    output_store: str = "global",
    output_slices: int = 1,
    output_store_buffers: int = 1,
    output_layout: str | None = None,
    operand_move_attrs: dict[str, dict[str, object]] | None = None,
) -> ModuleOp:
    """Single-CTA tcgen05 GEMM ``a @ bt.T``, with four warps sharing one pipeline.

    The ``sm_100a`` target makes ``schedule_matmul`` accumulate each M128 tile
    in tensor memory; warp 0 supplies the TMA, MMA, and allocation issuers.
    ``output_store="stmatrix"`` stages the output for TMA stores.
    ``cyclic_operand_pipeline=True`` queues MMA batches while retaining
    completion waits before operand-bank reuse and accumulator writeback.
    """
    m, n, k = (problem_size if dim is None else dim for dim in (m, n, k))
    if min(m, n, k) <= 0 or m % 128 or n % max(output_n_tile, 1):
        raise ValueError(
            "GEMM dimensions must be positive and divide the M128 and N tiles"
        )
    if operand_k_tile <= 0 or k % operand_k_tile:
        raise ValueError("the K tile must divide the reduction")
    if not 1 <= operand_stages <= k // operand_k_tile:
        raise ValueError("stages must be positive and fit the reduction")
    # B200 permits 227 KiB per CTA. Leave 1 KiB for alignment and barriers.
    operand_bytes = (128 + output_n_tile) * operand_k_tile * 2 * operand_stages
    if operand_bytes > 226 * 1024:
        raise ValueError("operand stages exceed Blackwell shared-memory capacity")
    if dtype not in ("bf16", "f16"):
        raise ValueError("dtype must be bf16 or f16")
    if output_grid_fast_axis not in ("m", "n"):
        raise ValueError("output_grid_fast_axis must be m or n")
    name = f"blackwell_m{m}_n{n}_k{k}_{dtype}_n{output_n_tile}_k{operand_k_tile}_s{operand_stages}"
    module = gpu_dialect.parse_module(f"""
        module attrs={{target: "sm_100a"}} {{
          kernel @{name}(%a: gmem<{m}x{k}x{dtype}>, %b: gmem<{n}x{k}x{dtype}>, %c: gmem<{m}x{n}x{dtype}>) {{
            %zero: rmem<f32> = constant 0.0
            matmul(%a: gmem<{m}x{k}x{dtype}>, %b: gmem<{n}x{k}x{dtype}>, %c: gmem<{m}x{n}x{dtype}>, %zero: rmem<f32>) collective=xyz.grid attrs={{label: "mm"}}
          }}
        }}
    """)
    # A launch-bound grid traverses M fastest on X, so N-fast swaps the axes.
    axes = (
        (CollectiveAxis.Y, CollectiveAxis.X)
        if output_grid_size is None and output_grid_fast_axis == "n"
        else (CollectiveAxis.X, CollectiveAxis.Y)
    )
    return _schedule_matmul(
        module,
        output_grid_size=output_grid_size,
        output_grid_axes=axes,
        output_grid_fast_axis=output_grid_fast_axis,
        output_m_tile=128,
        output_n_tile=output_n_tile,
        operand_stages=operand_stages,
        operand_k_tile=operand_k_tile,
        cyclic_operand_pipeline=cyclic_operand_pipeline,
        output_slices=output_slices,
        output_store_buffers=output_store_buffers,
        output_store=output_store,
        output_layout=output_layout,
        operand_move_attrs=operand_move_attrs,
    )


# B200 BF16, one graph-captured launch after a 256 MiB L2 flush.
# Sweep and joint finalist measurements: B200_SWEEP.md and B200_PIPELINE.md.
_B200_SQUARE_SCHEDULES: dict[int, dict[str, Any]] = {
    512: {"output_n_tile": 16, "operand_k_tile": 256, "operand_stages": 1},
    1024: {"output_n_tile": 64, "operand_k_tile": 256, "operand_stages": 2},
    2048: {
        "output_n_tile": 256,
        "operand_k_tile": 64,
        "operand_stages": 4,
        "cyclic_operand_pipeline": True,
    },
    4096: {
        "output_n_tile": 256,
        "operand_k_tile": 64,
        "operand_stages": 4,
        "cyclic_operand_pipeline": True,
        "output_grid_size": 148,
        "output_grid_fast_axis": "n",
    },
    8192: {
        "output_n_tile": 256,
        "operand_k_tile": 64,
        "operand_stages": 4,
        "cyclic_operand_pipeline": True,
        "output_grid_size": 1184,
        "output_grid_fast_axis": "n",
    },
    16384: {
        "output_n_tile": 256,
        "operand_k_tile": 64,
        "operand_stages": 4,
        "cyclic_operand_pipeline": True,
        "output_grid_fast_axis": "m",
        "operand_move_attrs": {"rhs": {"l2_cache_hint": "evict_last"}},
    },
}


def select_blackwell_square_schedule(problem_size: int) -> dict[str, Any]:
    """Return the measured B200 BF16 schedule for a supported square size.

    Measurements cover 512, 1024, 2048, 4096, 8192, and 16384. The copy allows
    callers to refine parameters without mutating the measured table.
    """
    if problem_size not in _B200_SQUARE_SCHEDULES:
        raise ValueError(f"no measured B200 BF16 square schedule for {problem_size}")
    return deepcopy(_B200_SQUARE_SCHEDULES[problem_size])


def blackwell_square_gemm_module(*, problem_size: int = 4096) -> ModuleOp:
    """Build a BF16 square GEMM using its measured B200 schedule."""
    return blackwell_tcgen05_gemm_module(
        problem_size=problem_size, **select_blackwell_square_schedule(problem_size)
    )
