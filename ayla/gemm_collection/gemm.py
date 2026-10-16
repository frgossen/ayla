"""Configurable GEMM module builders used by collection entries."""
import textwrap
from typing import cast

from ..base.ir import find_ops_of_type
from ..base.pipeline import op_label_matches
from ..gpu_dialect import (
    KernelOp,
    MatmulOp,
    MemTensorType,
    ModuleOp,
    WgrmemTensorType,
    gpu_dialect,
)
from ..pipelines.tile_matmul_pipeline import TileMatmulPipeline


def schedule_matmul(
    module: ModuleOp,
    *,
    output_m_tiles_per_cta: int = 1,
    output_n_tiles_per_cta: int = 1,
    output_m_tile: int = 128,
    output_n_tile: int = 256,
    warpgroup_n_tile: int | None = None,
    operand_stages: int = 3,
    operand_k_tile: int = 64,
    output_slices: int = 1,
    output_store_buffers: int = 1,
    # Split each 256-column output row into two 128-element TMA dimensions.
    # Shared-only padding controls the physical row stride seen by the TMA
    # store; the best overall stride may trade some STMatrix bank expansion
    # for faster TMA reads of the arena.
    output_tma_inner_size: int | None = 128,
    output_store: str = "stmatrix",
    operand_move_attrs: dict[str, dict[str, object]] | None = None,
    label: str | None = None,
    epilogue_label_filter: str | None = None,
    loop_label_prefix: str | None = None,
    unsafe_reuse_smem: bool = False,
) -> ModuleOp:
    """Apply the configurable WGMMA schedule to one concrete matmul.

    A label selects one matmul when the module contains several. Matmuls with
    a WGRMEM destination already own their output tiling and promotion.
    Keeping scheduling separate from module construction lets the JIT first
    specialize a dynamic template, then reuse the static collection's passes.
    """
    [kernel] = module.body.ops
    if not isinstance(kernel, KernelOp):
        raise ValueError("schedule_matmul requires exactly one kernel")
    matmuls = [
        op for op in find_ops_of_type(module, MatmulOp)
        if op_label_matches(op, label)
    ]
    if len(matmuls) != 1:
        raise ValueError("schedule_matmul requires exactly one matmul")
    matmul = matmuls[0]
    assert isinstance(matmul, MatmulOp)
    if label is None:
        label = cast(str, matmul.attrs.get("label", "mm"))
    register_accumulator = isinstance(matmul.dst.type, WgrmemTensorType)
    assert isinstance(matmul.lhs.type, MemTensorType)
    assert isinstance(matmul.rhs.type, MemTensorType)
    m, k = matmul.lhs.type.sizes
    n, rhs_k = matmul.rhs.type.sizes
    if None in (m, n, k, rhs_k):
        raise ValueError("schedule_matmul requires static dimensions")
    if rhs_k != k:
        raise ValueError("matmul reduction dimensions must match")
    m, n, k = cast(tuple[int, int, int], (m, n, k))

    return TileMatmulPipeline(
        m=m, n=n, k=k,
        register_accumulator=register_accumulator,
        output_m_tiles_per_cta=output_m_tiles_per_cta,
        output_n_tiles_per_cta=output_n_tiles_per_cta,
        output_m_tile=output_m_tile,
        output_n_tile=output_n_tile,
        warpgroup_n_tile=warpgroup_n_tile,
        operand_stages=operand_stages,
        operand_k_tile=operand_k_tile,
        output_slices=output_slices,
        output_store_buffers=output_store_buffers,
        output_tma_inner_size=output_tma_inner_size,
        output_store=output_store,
        operand_move_attrs=operand_move_attrs,
        label=label,
        epilogue_label_filter=epilogue_label_filter,
        loop_label_prefix=loop_label_prefix,
        unsafe_reuse_smem=unsafe_reuse_smem,
    ).run(module)


def auto_matmul_4096_wgmma256_wg_tma_store_module(
    *, problem_size: int = 4096,
    m: int | None = None,
    n: int | None = None,
    k: int | None = None,
    output_m_tiles_per_cta: int = 1,
    output_n_tiles_per_cta: int = 1,
    output_m_tile: int = 128,
    output_n_tile: int = 256,
    warpgroup_n_tile: int | None = None,
    operand_stages: int = 3,
    operand_k_tile: int = 64,
    output_slices: int = 1,
    output_store_buffers: int = 1,
    output_tma_inner_size: int | None = 128,
    output_store: str = "stmatrix",
    operand_move_attrs: dict[str, dict[str, object]] | None = None,
    ptx_optimization_level: int | None = None,
    dtype: str = "bf16",
) -> ModuleOp:
    """Build and schedule a static BF16 or FP16 GEMM module."""
    m = problem_size if m is None else m
    n = problem_size if n is None else n
    k = problem_size if k is None else k
    if min(m, n, k) <= 0:
        raise ValueError("problem dimensions must be positive")
    if dtype not in ("bf16", "f16"):
        raise ValueError("dtype must be 'bf16' or 'f16'")
    problem_name = (
        str(problem_size)
        if m == n == k == problem_size
        else f"m{m}_n{n}_k{k}"
    )
    resolved_warpgroup_n_tile = warpgroup_n_tile
    if resolved_warpgroup_n_tile is None:
        resolved_warpgroup_n_tile = (
            output_n_tile if output_n_tile <= 256 else 128
        )
    name = f"auto_matmul_{problem_name}_wgmma256_wg_tma_store_k{operand_stages}"
    if dtype != "bf16":
        name += f"_{dtype}"
    if operand_k_tile != 64:
        name += f"_ktile{operand_k_tile}"
    if output_m_tile != 128:
        name += f"_m{output_m_tile}"
    if output_n_tile != 256:
        name += f"_n{output_n_tile}"
    if resolved_warpgroup_n_tile not in (128, 256):
        name += f"_wg{resolved_warpgroup_n_tile}"
    if output_m_tiles_per_cta != 1:
        name += f"_{output_m_tiles_per_cta}tiles"
    if output_n_tiles_per_cta != 1:
        name += f"_n{output_n_tiles_per_cta}tiles"
    if output_slices != 1:
        name += f"_split{output_slices}"
    if output_store_buffers != 1:
        name += f"_dbuf{output_store_buffers}"
    if output_store == "scalar":
        name += "_pre_stsm"

    module = gpu_dialect.parse(textwrap.dedent(f"""\
        module {{
          kernel @{name}(%a: gmem<{m}x{k}x{dtype}>, %b: gmem<{n}x{k}x{dtype}>, %c: gmem<{m}x{n}x{dtype}>) attrs={{ptx_optimization_level: {ptx_optimization_level}}} {{
            %zero: rmem<f32> = constant() attrs={{value: 0.0}}
            matmul(%a: gmem<{m}x{k}x{dtype}>, %b: gmem<{n}x{k}x{dtype}>, %c: gmem<{m}x{n}x{dtype}>, %zero: rmem<f32>) collective=xyz.grid attrs={{label: "mm"}}
          }}
        }}"""))
    assert isinstance(module, ModuleOp)
    return schedule_matmul(
        module,
        output_m_tiles_per_cta=output_m_tiles_per_cta,
        output_n_tiles_per_cta=output_n_tiles_per_cta,
        output_m_tile=output_m_tile,
        output_n_tile=output_n_tile,
        warpgroup_n_tile=resolved_warpgroup_n_tile,
        operand_stages=operand_stages,
        operand_k_tile=operand_k_tile,
        output_slices=output_slices,
        output_store_buffers=output_store_buffers,
        output_tma_inner_size=output_tma_inner_size,
        output_store=output_store,
        operand_move_attrs=operand_move_attrs,
    )
