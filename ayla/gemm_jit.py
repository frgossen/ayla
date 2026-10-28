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
import re
from dataclasses import dataclass
from typing import Any
import textwrap
from functools import cache

import torch

from .gemm_schedule_table import CONFIGS, SCHEDULES
from .gpu_dialect import CollectiveAxis, GmemTensorType, KernelOp, ModuleOp, gpu_dialect
from .jit_compiler import ShapeSpecializingJITCompiler
from .torch_custom_call import DEFAULT_NAMESPACE, OperandKind

_IR_DTYPES = {"float16": "f16", "bfloat16": "bf16"}
# Table entries omit default fields; these are the ones _builder_config indexes.
_DEFAULTS: dict[str, object] = {
    "output_n_tiles_per_cta": 1,
    "ptx_optimization_level": None,
    "output_cta_count": None,
    "unroll_output_tiles": True,
}
# Unrolling more output waves than this is only trusted on the measured shape.
_MAX_TRANSFERRED_WAVES = 8


@dataclass(frozen=True)
class GemmScheduleSeed:
    captured_config: dict[str, object]
    builder_config: dict[str, object] | None
    adaptations: tuple[str, ...]
    source_shape: tuple[int, int, int] | None
    source_percentage_of_torch: float | None
    post_builder_passes: tuple[str, ...] = ()
    transpose_output: bool = False
    fixed_output_unroll_factor: int | None = None
    pipeline_output_waves: bool = False
    pipeline_output_writeback: bool = False


def apply_gemm_schedule_seed_post_build(
    module: Any,
    seed: GemmScheduleSeed,
) -> Any:
    """Apply the measured post-builder pass sequence required by ``seed``."""
    if seed.pipeline_output_waves or seed.pipeline_output_writeback:
        if (
            seed.fixed_output_unroll_factor is not None
            or seed.builder_config is None
            or seed.builder_config.get("output_grid_size") is None
        ):
            raise ValueError("wave pipelining requires a rolled fixed grid")
        from .base.ir import find_ancestor_of_type, find_ops_of_type
        from .gpu_dialect import AsyncOp, AwaitOp, ForOp
        from .pipelines.canonicalize import Canonicalize
        from .passes.pipeline_loop import PipelineLoop
        from .passes.schedule_async import ScheduleAsync

        outer_loops = [
            loop
            for loop in find_ops_of_type(module, ForOp)
            if find_ancestor_of_type(loop, ForOp) is None
        ]
        if len(outer_loops) != 1:
            raise ValueError("wave pipelining requires one outer tile loop")
        outer_loops[0].attrs["label"] = "heuristic_output_grid"
        if seed.pipeline_output_waves:
            for body_op in outer_loops[0].body.ops:
                if isinstance(body_op, AwaitOp):
                    break
                if isinstance(body_op, AsyncOp) and re.fullmatch(
                    r"mm_(lhs|rhs)_smem_iter\d+_async",
                    str(body_op.attrs.get("label", "")),
                ):
                    body_op.attrs["label"] = "heuristic_wave_prologue_async"
            module = PipelineLoop(
                label_filter_loop="heuristic_output_grid",
                label_filter_op=r"heuristic_wave_prologue_async",
                pipelining_offset=1,
            ).run(module)
        if seed.pipeline_output_writeback:
            # Issue everything but the output writeback one tile early, so a
            # tile's writeback runs while the next tile's work is in flight.
            module = PipelineLoop(
                label_filter_loop="heuristic_output_grid",
                label_filter_op=r"(?!mm_dst_(smem|tma)).*",
                pipelining_offset=1,
            ).run(module)
        module = ScheduleAsync().run(module)
        module = Canonicalize().run(module)
    output_was_unrolled = (
        seed.fixed_output_unroll_factor is not None
        and seed.fixed_output_unroll_factor > 1
    )
    if output_was_unrolled:
        from .base.ir import find_ancestor_of_type, find_ops_of_type
        from .gpu_dialect import ForOp
        from .pipelines.canonicalize import Canonicalize
        from .passes.licm import LICM
        from .passes.unroll_loop import UnrollLoop

        module = LICM().run(module)
        module = LICM().run(module)
        outer_loops = [
            loop
            for loop in find_ops_of_type(module, ForOp)
            if find_ancestor_of_type(loop, ForOp) is None
        ]
        if len(outer_loops) != 1:
            raise ValueError("fixed output-grid unrolling requires one outer tile loop")
        outer_loops[0].attrs["label"] = "heuristic_output_grid"
        module = UnrollLoop(
            unroll_factor=cast(int, seed.fixed_output_unroll_factor),
            label_filter="heuristic_output_grid",
        ).run(module)
        module = Canonicalize().run(module)
    expected = ("schedule_async", "canonicalize")
    if seed.post_builder_passes and seed.post_builder_passes != expected:
        raise ValueError(
            f"unsupported GEMM post-builder passes: {seed.post_builder_passes}"
        )
    if not output_was_unrolled and not seed.post_builder_passes:
        return module
    # Keep pass imports lazy so selecting a heuristic remains lightweight.
    from .pipelines.canonicalize import Canonicalize
    from .passes.schedule_async import ScheduleAsync

    return Canonicalize().run(ScheduleAsync().run(module))


def _as_int(value: object) -> int:
    if not isinstance(value, int):
        raise TypeError(f"expected an integer GEMM configuration value, got {value!r}")
    return value


def _builder_config(captured: dict[str, object]) -> dict[str, object] | None:
    """Translate the measured search configuration into builder arguments."""
    if captured.get("unsafe_reuse_smem", False) is True:
        return None
    if "output_m_tiles_per_cta" in captured:
        return dict(captured)
    preferred_ctas = captured["output_cta_count"]
    m_group = _as_int(captured.get("output_tiles_per_cta", 1))
    n_group = _as_int(captured["output_n_tiles_per_cta"])
    if preferred_ctas is not None:
        m_group, n_group = 1, 1

    lhs = {"l2_cache_hint": captured["lhs_l2_cache_hint"]}
    if captured.get("lhs_l2_promotion") is not None:
        lhs["l2_promotion"] = captured["lhs_l2_promotion"]
    rhs = {"l2_cache_hint": captured["rhs_l2_cache_hint"]}
    if captured.get("rhs_l2_promotion") is not None:
        rhs["l2_promotion"] = captured["rhs_l2_promotion"]
    if captured.get("multicast", False) is True:
        if _as_int(captured.get("cluster_n_ctas", 1)) > 1:
            lhs["multicast"] = True
        if _as_int(captured.get("cluster_m_ctas", 1)) > 1:
            rhs["multicast"] = True
    builder: dict[str, object] = {
        "output_m_tiles_per_cta": m_group,
        "output_n_tiles_per_cta": n_group,
        "output_m_tile": captured["output_m_tile"],
        "output_n_tile": captured["output_n_tile"],
        "warpgroup_n_tile": captured["warpgroup_n_tile"],
        "operand_stages": captured["operand_stages"],
        "operand_k_tile": captured["operand_k_tile"],
        "cyclic_operand_pipeline": captured.get(
            "cyclic_operand_pipeline",
            False,
        ),
        "output_slices": captured["output_slices"],
        "output_store_buffers": captured["output_store_buffers"],
        "output_tma_inner_size": captured["output_tma_inner_size"],
        "output_store": captured["output_store"],
        "operand_move_attrs": {
            "lhs": lhs,
            "rhs": rhs,
            "dst": {"l2_cache_hint": captured["dst_l2_cache_hint"]},
        },
        "ptx_optimization_level": captured["ptx_optimization_level"],
    }
    if preferred_ctas is not None:
        builder["output_grid_size"] = _as_int(preferred_ctas)
        builder["output_grid_fast_axis"] = captured.get(
            "output_tile_fast_axis",
            "n",
        )
    if captured.get("swap_output_grid_axes") is True:
        builder["output_grid_axes"] = (
            CollectiveAxis.Y,
            CollectiveAxis.X,
        )
    if reverse := str(captured.get("output_tile_reverse") or ""):
        builder["output_reversed_dims"] = tuple(reverse.upper())
    optional = (
        "warpgroup_m_tile",
        "warpgroup_fast_axis",
        "rhs_tma_load_slices",
        "cluster_m_ctas",
        "cluster_n_ctas",
    )
    for key in optional:
        value = captured.get(key)
        if value not in (None, 1, "n"):
            builder[key] = value
    return builder


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
    """Adapt a table entry to this GEMM, or None if it cannot transfer."""
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
    builder = _builder_config(captured)
    assert builder is not None
    unroll = None
    if fixed_grid and captured["unroll_output_tiles"]:
        tiles = math.ceil(
            builder_m / int(cast(int, captured["output_m_tile"]))
        ) * math.ceil(builder_n / int(cast(int, captured["output_n_tile"])))
        unroll = math.ceil(tiles / int(cast(int, captured["output_cta_count"])))
    if not exact and (
        not _legal(
            builder,
            builder_m,
            builder_n,
            k,
            allow_partial_output_tiles=operation == "bmm",
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


def select_gemm_schedule(
    operation: str,
    dtype: str,
    m: int,
    n: int,
    k: int,
    batch: int = 1,
) -> GemmScheduleSeed | None:
    """Return the nearest measured schedule legal for this GEMM, if any.

    A transferred schedule names its source shape in ``adaptations``.
    """
    table = SCHEDULES.get((operation, dtype), {})
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
    return None


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
    """Build and schedule one GEMM exactly as the sweep measured ``seed``."""
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
