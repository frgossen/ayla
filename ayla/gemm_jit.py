"""JIT-compile mm, addmm, and bmm from the measured schedules in a lookup table.

A shape that won a sweep compiles its measured schedule. Any other shape takes
the nearest measured shape's first schedule that is legal for it, with distance
measured in log2 of (batch, m, n, k)::

    op = gemm_op("mm", "bfloat16")      # op(a, bt, out): out = a @ bt.T
    op(a, bt, out)
"""

from __future__ import annotations

from typing import cast

import math
from functools import cache
from dataclasses import dataclass

import torch

from .gemm_schedule_table import CONFIGS, SCHEDULES
from .gpu_dialect import GmemTensorType, KernelOp, ModuleOp, gpu_dialect
from .jit_compiler import ShapeSpecializingJITCompiler
from .torch_custom_call import DEFAULT_NAMESPACE, OperandKind

_IR_DTYPES = {"float16": "f16", "bfloat16": "bf16"}
# Table entries omit default fields; these are the ones _adapt_config indexes.
_DEFAULTS: dict[str, object] = {
    "output_n_tiles_per_cta": 1,
    "ptx_optimization_level": None,
    "output_cta_count": None,
    "unroll_output_tiles": True,
}


@dataclass(frozen=True)
class GemmScheduleSeed:
    captured_config: dict[str, object]
    builder_config: dict[str, object] | None
    adaptations: tuple[str, ...]
    source_shape: tuple[int, int, int] | None
    source_percentage_of_torch: float | None = None
    post_builder_passes: tuple[str, ...] = ()


def _adapt_config(
    captured: dict[str, object],
) -> dict[str, object] | None:
    # Keep feature-only measurements in the table, but do not execute them on main.
    for key, default in (
        ("cyclic_operand_pipeline", False),
        ("multicast", False),
        ("cluster_m_ctas", 1),
        ("cluster_n_ctas", 1),
        ("pipeline_output_waves", False),
        ("pipeline_output_writeback", False),
        ("swap_output_grid_axes", False),
        ("transpose_output", False),
        ("output_tile_reverse", None),
        ("warpgroup_m_tile", None),
        ("output_cta_count", None),
        ("unsafe_reuse_smem", False),
    ):
        if captured.get(key, default) not in (default, None, False):
            return None
    if captured.get("rhs_tma_load_slices") not in (None, 1):
        return None
    if captured.get("warpgroup_fast_axis", "n") != "n":
        return None
    if captured.get("output_store") != "stmatrix":
        return None
    builder = {
        key: captured[key]
        for key in (
            "output_m_tile",
            "output_n_tile",
            "warpgroup_n_tile",
            "operand_stages",
            "operand_k_tile",
            "output_slices",
            "output_store_buffers",
            "output_tma_inner_size",
            "output_store",
            "ptx_optimization_level",
        )
    }
    builder["output_m_tiles_per_cta"] = captured.get(
        "output_m_tiles_per_cta",
        captured.get("output_tiles_per_cta", 1),
    )
    builder["output_n_tiles_per_cta"] = captured.get("output_n_tiles_per_cta", 1)
    if any(
        builder[key] not in (1, 2, 3, 4, 6, 8, 16)
        for key in (
            "output_m_tiles_per_cta",
            "output_n_tiles_per_cta",
        )
    ):
        return None
    attrs = {}
    for operand in ("lhs", "rhs", "dst"):
        attrs[operand] = {"l2_cache_hint": captured.get(f"{operand}_l2_cache_hint")}
        if promotion := captured.get(f"{operand}_l2_promotion"):
            attrs[operand]["l2_promotion"] = promotion
    builder["operand_move_attrs"] = attrs
    return builder


def _as_int(value: object) -> int:
    if not isinstance(value, int):
        raise TypeError(f"expected an integer GEMM configuration value, got {value!r}")
    return value


def _legal(
    config: dict[str, object],
    m: int,
    n: int,
    k: int,
    *,
    allow_partial_output_tiles: bool = False,
) -> bool:
    m_tile = _as_int(config["output_m_tile"])
    n_tile = _as_int(config["output_n_tile"])
    k_tile = _as_int(config["operand_k_tile"])
    m_group = _as_int(config["output_m_tiles_per_cta"])
    n_group = _as_int(config["output_n_tiles_per_cta"])
    blocked_output = (
        config["output_store"] != "scalar"
        and config["output_tma_inner_size"] is not None
    )
    return (
        (allow_partial_output_tiles or (m >= m_tile and n >= n_tile))
        and k % k_tile == 0
        and (not blocked_output or (n % 64 == 0 and n_tile % 64 == 0))
        and math.ceil(m / m_tile) >= m_group
        and math.ceil(n / n_tile) >= n_group
    )


def _seed(
    captured: dict[str, object],
    operation: str,
    m: int,
    n: int,
    k: int,
    source: tuple[int, int, int, int],
    exact: bool,
) -> GemmScheduleSeed | None:
    if n % 8 or min(m, n, k) <= 0:
        return None
    builder = _adapt_config(captured)
    if builder is None or not _legal(builder, m, n, k, allow_partial_output_tiles=True):
        return None
    return GemmScheduleSeed(
        captured,
        builder,
        () if exact else (f"nearest measured shape {source}",),
        source[1:],
        post_builder_passes=(
            ("schedule_async", "canonicalize")
            if captured.get("reschedule_async")
            else ()
        ),
    )


def measured_gemm_configs(
    operation: str,
    dtype: str,
    m: int,
    n: int,
    k: int,
    batch: int = 1,
) -> tuple[dict[str, object], ...]:
    """Return all exact-shape measured configs, winner first, as fresh copies.

    These records include features available only on feature/consolidated.
    Use select_gemm_schedule to choose a schedule executable on this branch.
    """
    indices = SCHEDULES.get((operation, dtype), {}).get((batch, m, n, k), ())
    return tuple({**_DEFAULTS, **CONFIGS[index]} for index in indices)


def select_gemm_schedule(
    operation: str,
    dtype: str,
    m: int,
    n: int,
    k: int,
    batch: int = 1,
) -> GemmScheduleSeed | None:
    """Return a main-compatible measured schedule, or a conservative fallback.

    A transferred schedule names its source shape in ``adaptations``.
    """
    if min(batch, m, n, k) <= 0 or operation not in ("mm", "addmm", "bmm"):
        return None
    table = SCHEDULES.get((operation, dtype), {})
    if not table:
        return None
    query = (batch, m, n, k)

    def distance(shape: tuple[int, int, int, int]) -> float:
        return sum(
            abs(math.log2(a) - math.log2(b)) for a, b in zip(shape, query, strict=True)
        )

    for shape in sorted(table, key=distance):
        for index in table[shape]:
            seed = _seed(
                {**_DEFAULTS, **CONFIGS[index]},
                operation,
                m,
                n,
                k,
                shape,
                exact=shape == query,
            )
            if seed is not None:
                return seed
    return _fallback_schedule(operation, dtype, m, n, k, batch)


def _fallback_schedule(
    operation: str,
    dtype: str,
    m: int,
    n: int,
    k: int,
    batch: int = 1,
) -> GemmScheduleSeed | None:
    """Conservative unmeasured schedule using only main compiler features."""
    if k % 64 or n % 8 or min(batch, m, n, k) <= 0:
        return None
    captured: dict[str, object] = {
        **_DEFAULTS,
        "output_m_tile": 64,
        "output_n_tile": 64 if n % 64 == 0 else 48,
        "warpgroup_n_tile": 64 if n % 64 == 0 else 48,
        "operand_stages": 2,
        "operand_k_tile": 64,
        "output_tiles_per_cta": 1,
        "output_tile_fast_axis": "m",
        "output_slices": 1,
        "output_store_buffers": 1,
        "output_tma_inner_size": 128 if n % 64 == 0 else None,
        "output_store": "stmatrix",
        "lhs_l2_cache_hint": None,
        "rhs_l2_cache_hint": None,
        "dst_l2_cache_hint": None,
    }
    builder = _adapt_config(captured)
    if builder is None:
        return None
    return GemmScheduleSeed(
        captured,
        builder,
        (
            "unmeasured main-compatible fallback; use feature/consolidated for full search",
        ),
        None,
        None,
    )


def _module(
    operation: str,
    dtype: str,
    batch: int,
    m: int,
    n: int,
    k: int,
) -> ModuleOp:
    d = _IR_DTYPES[dtype]
    a, bt, c = f"gmem<{m}x{k}x{d}>", f"gmem<{n}x{k}x{d}>", f"gmem<{m}x{n}x{d}>"
    signature = f"%a: {a}, %bt: {bt}, %c: {c}"
    body = f'%zero: rmem<f32> = constant 0.0\nmatmul(%a: {a}, %bt: {bt}, %c: {c}, %zero: rmem<f32>) collective=xyz.grid attrs={{label: "mm"}}'
    grid = ""
    if operation == "addmm":
        bias = f"gmem<{m}x{n}x{d} strides=[0, 1]>"
        signature = f"%bias: {bias}, {signature}"
        body = f'matmul(%a: {a}, %bt: {bt}, %c: {c}, %bias: {bias}) collective=xyz.grid attrs={{label: "mm"}}'
    elif operation == "bmm":
        a3, bt3, c3 = (
            f"gmem<{batch}x{m}x{k}x{d}>",
            f"gmem<{batch}x{n}x{k}x{d}>",
            f"gmem<{batch}x{m}x{n}x{d}>",
        )
        a1, bt1, c1 = (
            f"gmem<1x{m}x{k}x{d}>",
            f"gmem<1x{n}x{k}x{d}>",
            f"gmem<1x{m}x{n}x{d}>",
        )
        signature = f"%a3: {a3}, %bt3: {bt3}, %c3: {c3}"
        grid = f" grid_dim=[None, None, {batch}]"
        body = f"""%bz: index = block_id.z
          %a1: {a1} = tile(%a3: {a3}, %bz: index, 0, 0)
          %a: {a} = reshape(%a1: {a1})
          %bt1: {bt1} = tile(%bt3: {bt3}, %bz: index, 0, 0)
          %bt: {bt} = reshape(%bt1: {bt1})
          %c1: {c1} = tile(%c3: {c3}, %bz: index, 0, 0)
          %c: {c} = reshape(%c1: {c1})
          {body}"""
    return gpu_dialect.parse_module(
        f"module {{ kernel @{operation}({signature}){grid} {{ {body} }} }}"
    )


def build_gemm_module(
    operation: str,
    dtype: str,
    batch: int,
    m: int,
    n: int,
    k: int,
    seed: GemmScheduleSeed,
) -> ModuleOp:
    """Build and schedule one GEMM with the selected main-compatible seed."""
    from .gemm_collection.gemm import schedule_matmul

    assert seed.builder_config is not None
    kwargs = dict(seed.builder_config)
    ptx_optimization_level = kwargs.pop("ptx_optimization_level")
    module = _module(operation, dtype, batch, m, n, k)
    [kernel] = module.body.ops
    assert isinstance(kernel, KernelOp)
    if ptx_optimization_level is not None:
        kernel.attrs["ptx_optimization_level"] = ptx_optimization_level
    module = schedule_matmul(module, **kwargs)  # type: ignore[arg-type]
    if seed.post_builder_passes:
        from .passes.schedule_async import ScheduleAsync
        from .pipelines.canonicalize import Canonicalize

        module = Canonicalize().run(ScheduleAsync().run(module))
    return module


def gemm_jit_compiler(
    operation: str,
    dtype: str,
) -> ShapeSpecializingJITCompiler:
    """Compile each concrete mm, addmm, or bmm shape from the schedule table.

    The dynamic template only gives the launcher its operand signature; each
    specialization is rebuilt with :func:`build_gemm_module`.
    """
    from .ptx_codegen import make_ptx_kernel
    from .ptx_runtime import make_ptx_launcher

    if operation not in ("mm", "addmm", "bmm"):
        raise ValueError(f"unsupported GEMM operation {operation}")
    d = _IR_DTYPES[dtype]
    t = f"gmem<{'?x?x?' if operation == 'bmm' else '?x?'}x{d}>"
    bias = f"%bias: gmem<?x{d}>, " if operation == "addmm" else ""
    template = gpu_dialect.parse(
        f"""\
        module {{
          kernel @jit_{operation}_{d}({bias}%a: {t}, %bt: {t}, %c: {t}) {{
            matmul(%a: {t}, %bt: {t}, %c: {t}) collective=xyz.grid
          }}
        }}"""
    )
    assert isinstance(template, ModuleOp)

    def compile_specialization(module: ModuleOp):
        [kernel] = module.body.ops
        assert isinstance(kernel, KernelOp)
        *_, a, bt, _c = (
            typ.sizes for typ in kernel.signature if isinstance(typ, GmemTensorType)
        )
        assert all(size is not None for size in (*a, *bt))
        a = cast(tuple[int, ...], a)
        bt = cast(tuple[int, ...], bt)
        *batch, m, k = a
        n = bt[-2]
        schedule = select_gemm_schedule(operation, dtype, m, n, k, *batch)
        if schedule is None:
            raise ValueError(
                f"{kernel.name}: no schedule for {operation} {tuple(batch)}x{m}x{n}x{k}"
            )
        scheduled = build_gemm_module(
            operation,
            dtype,
            batch[0] if batch else 1,
            m,
            n,
            k,
            schedule,
        )
        return make_ptx_launcher(make_ptx_kernel(scheduled))

    return ShapeSpecializingJITCompiler(template, compile_specialization)


@cache
def gemm_op(
    operation: str,
    dtype: str,
    namespace: str = DEFAULT_NAMESPACE,
) -> torch._ops.OpOverload:
    """Torch op ``(bias?, a, bt, out) -> ()`` writing ``out = [bias +] a @ bt^T``."""
    from .torch_custom_call import make_torch_custom_call

    compiler = gemm_jit_compiler(operation, dtype)
    inputs = 3 if operation == "addmm" else 2
    return make_torch_custom_call(
        compiler,
        [OperandKind.R] * inputs + [OperandKind.RW],
        namespace=namespace,
    )
