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
import textwrap
from functools import cache

import torch

from .gemm_heuristics import (
    GemmScheduleSeed,
    _adapt_config,
    _legal,
    apply_gemm_schedule_seed_post_build,
)
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
# Unrolling more output waves than this is only trusted on the measured shape.
_MAX_TRANSFERRED_WAVES = 8


def _seed(
    captured: dict[str, object],
    operation: str,
    m: int,
    n: int,
    k: int,
    source: tuple[int, int, int, int],
    exact: bool,
) -> GemmScheduleSeed | None:
    """Adapt a table entry to this GEMM, or None if it cannot transfer."""
    if n % 8 or min(m, n, k) <= 0:
        return None
    transpose = bool(captured.get("transpose_output"))
    fixed_grid = captured["output_cta_count"] is not None
    if not exact and (
        n % 8  # the TMA output row stride must be 16-byte aligned
        or captured.get("cluster_m_ctas", 1) != 1
        or captured.get("cluster_n_ctas", 1) != 1
        or (fixed_grid and operation == "bmm")
    ):
        return None
    builder_m, builder_n = (n, m) if transpose else (m, n)
    builder, _ = _adapt_config(
        captured,
        builder_m,
        builder_n,
        k,
        preserve_output_grid=True,
    )
    if builder is None:
        return None
    unroll = None
    if fixed_grid and captured["unroll_output_tiles"]:
        tiles = math.ceil(
            builder_m / int(cast(int, captured["output_m_tile"]))
        ) * math.ceil(builder_n / int(cast(int, captured["output_n_tile"])))
        unroll = math.ceil(tiles / int(cast(int, captured["output_cta_count"])))
    if (
        not _legal(
            builder,
            builder_m,
            builder_n,
            k,
            allow_partial_output_tiles=True,
        )
        or (unroll or 0) > _MAX_TRANSFERRED_WAVES
    ):
        return None
    return GemmScheduleSeed(
        captured_config=captured,
        builder_config=builder,
        adaptations=() if exact else (f"nearest measured shape {source}",),
        source_shape=source[1:],
        source_percentage_of_torch=None,
        post_builder_passes=(
            ("schedule_async", "canonicalize")
            if captured.get("reschedule_async")
            else ()
        ),
        transpose_output=transpose,
        fixed_output_unroll_factor=unroll,
        pipeline_output_waves=bool(captured.get("pipeline_output_waves")),
        pipeline_output_writeback=bool(captured.get("pipeline_output_writeback")),
    )


def measured_gemm_configs(
    operation: str, dtype: str, m: int, n: int, k: int, batch: int = 1,
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
    operation: str, dtype: str, m: int, n: int, k: int, batch: int = 1,
) -> GemmScheduleSeed | None:
    """Conservative unmeasured schedule using only main compiler features."""
    if k % 64 or n % 8 or min(batch, m, n, k) <= 0:
        return None
    captured: dict[str, object] = {
        **_DEFAULTS,
        "output_m_tile": 64, "output_n_tile": 64 if n % 64 == 0 else 48,
        "warpgroup_n_tile": 64 if n % 64 == 0 else 48,
        "operand_stages": 2, "operand_k_tile": 64,
        "output_tiles_per_cta": 1, "output_tile_fast_axis": "m",
        "output_slices": 1, "output_store_buffers": 1,
        "output_tma_inner_size": 128 if n % 64 == 0 else None,
        "output_store": "stmatrix",
        "lhs_l2_cache_hint": None, "rhs_l2_cache_hint": None,
        "dst_l2_cache_hint": None,
    }
    builder, _ = _adapt_config(captured, m, n, k)
    if builder is None:
        return None
    return GemmScheduleSeed(
        captured, builder,
        ("unmeasured main-compatible fallback; use feature/consolidated for full search",),
        None, None,
    )


def _module(
    operation: str,
    dtype: str,
    batch: int,
    m: int,
    n: int,
    k: int,
    transpose_output: bool,
) -> ModuleOp:
    d = _IR_DTYPES[dtype]
    a, bt = f"gmem<{m}x{k}x{d}>", f"gmem<{n}x{k}x{d}>"
    c = f"gmem<{m}x{n}x{d}>"
    ct = f"gmem<{n}x{m}x{d} strides=[1, {n}]>"
    if operation == "addmm":
        if transpose_output:
            bias = f"gmem<{n}x{m}x{d} strides=[1, 0]>"
            signature, operands = ct, f"%b: {bt}, %a: {a}, %c: {ct}"
        else:
            bias = f"gmem<{m}x{n}x{d} strides=[0, 1]>"
            signature, operands = c, f"%a: {a}, %b: {bt}, %c: {c}"
        text = f"""\
            module {{
              kernel @addmm(%bias: {bias}, %a: {a}, %b: {bt}, %c: {signature}) {{
                matmul({operands}, %bias: {bias}) collective=xyz.grid attrs={{label: "mm"}}
              }}
            }}"""
    elif operation == "bmm":
        a3, bt3 = f"gmem<{batch}x{m}x{k}x{d}>", f"gmem<{batch}x{n}x{k}x{d}>"
        a1, bt1 = f"gmem<1x{m}x{k}x{d}>", f"gmem<1x{n}x{k}x{d}>"
        if transpose_output:
            c3 = f"gmem<{batch}x{n}x{m}x{d} strides=[{m * n}, 1, {n}]>"
            c1 = f"gmem<1x{n}x{m}x{d} strides=[{m * n}, 1, {n}]>"
            c2, operands = ct, f"%bt2: {bt}, %a2: {a}, %c2: {ct}"
        else:
            c3, c1 = f"gmem<{batch}x{m}x{n}x{d}>", f"gmem<1x{m}x{n}x{d}>"
            c2, operands = c, f"%a2: {a}, %bt2: {bt}, %c2: {c}"
        text = f"""\
            module {{
              kernel @bmm(%a: {a3}, %bt: {bt3}, %c: {c3}) grid_dim=[None, None, {batch}] {{
                %bz: index = block_id.z
                %a3: {a1} = tile(%a: {a3}, %bz: index, 0, 0)
                %a2: {a} = reshape(%a3: {a1})
                %bt3: {bt1} = tile(%bt: {bt3}, %bz: index, 0, 0)
                %bt2: {bt} = reshape(%bt3: {bt1})
                %c3: {c1} = tile(%c: {c3}, %bz: index, 0, 0)
                %c2: {c2} = reshape(%c3: {c1})
                %zero: rmem<f32> = constant 0.0
                matmul({operands}, %zero: rmem<f32>) collective=xyz.grid attrs={{label: "mm"}}
              }}
            }}"""
    else:
        if transpose_output:
            signature, operands = ct, f"%b: {bt}, %a: {a}, %c: {ct}"
        else:
            signature, operands = c, f"%a: {a}, %b: {bt}, %c: {c}"
        text = f"""\
            module {{
              kernel @mm(%a: {a}, %b: {bt}, %c: {signature}) {{
                %zero: rmem<f32> = constant 0.0
                matmul({operands}, %zero: rmem<f32>) collective=xyz.grid attrs={{label: "mm"}}
              }}
            }}"""
    module = gpu_dialect.parse(textwrap.dedent(text))
    assert isinstance(module, ModuleOp)
    return module


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
    module = _module(operation, dtype, batch, m, n, k, seed.transpose_output)
    [kernel] = module.body.ops
    assert isinstance(kernel, KernelOp)
    if ptx_optimization_level is not None:
        kernel.attrs["ptx_optimization_level"] = ptx_optimization_level
    module = schedule_matmul(module, **kwargs)  # type: ignore[arg-type]
    return apply_gemm_schedule_seed_post_build(module, seed)


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
        textwrap.dedent(f"""\
        module {{
          kernel @jit_{operation}_{d}({bias}%a: {t}, %bt: {t}, %c: {t}) {{
            matmul(%a: {t}, %bt: {t}, %c: {t}) collective=xyz.grid
          }}
        }}""")
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
