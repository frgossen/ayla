"""Configurable GEMM module builders used by collection entries."""

import textwrap
from ..base.ir import Value, find_ops_of_type
from ..gpu_dialect import (
    AllocOp,
    CollectiveLevel,
    ElementwiseOp,
    F32Type,
    GmemTensorType,
    KernelOp,
    MoveOp,
    RmemTensorType,
    WgrmemTensorType,
    CollectiveAxis,
    ModuleOp,
    gpu_dialect,
)
from ..pipelines.tile_matmul_pipeline import (
    TileMatmulPipeline,
    _schedule_matmul,
    schedule_register_matmul as schedule_register_matmul,
    schedule_tiled_matmul as schedule_tiled_matmul,
)

schedule_matmul = _schedule_matmul


def auto_matmul_4096_wgmma256_wg_tma_store_module(
    *,
    problem_size: int = 4096,
    m: int | None = None,
    n: int | None = None,
    k: int | None = None,
    output_m_tiles_per_cta: int = 1,
    output_n_tiles_per_cta: int = 1,
    output_grid_size: int | None = None,
    output_grid_fast_axis: str = "n",
    output_grid_axes: tuple[CollectiveAxis, CollectiveAxis] = (
        CollectiveAxis.X,
        CollectiveAxis.Y,
    ),
    output_reversed_dims: tuple[str, ...] = (),
    output_m_tile: int = 128,
    output_n_tile: int = 256,
    warpgroup_m_tile: int | None = None,
    warpgroup_n_tile: int | None = None,
    cluster_m_ctas: int = 1,
    cluster_n_ctas: int = 1,
    operand_stages: int = 3,
    operand_stage_overrides: dict[str, int] | None = None,
    operand_k_tile: int = 64,
    cyclic_operand_pipeline: bool = False,
    output_slices: int = 1,
    output_store_buffers: int = 1,
    output_fragment_cols: int | None = None,
    output_fragment_swizzle: str | None = None,
    warpgroup_ctas: int = 1,
    output_tma_inner_size: int | None = 128,
    output_store: str = "stmatrix",
    unsafe_reuse_smem: bool = False,
    operand_move_attrs: dict[str, dict[str, object]] | None = None,
    ptx_optimization_level: int | None = None,
    dtype: str = "bf16",
    output_dtype: str | None = None,
    rhs_dtype: str | None = None,
    scaled: bool = False,
    split_k_parts: int = 1,
    split_k_atomic: bool = False,
) -> ModuleOp:
    """Build and schedule a static BF16, FP16 or FP8 GEMM module.

    Atomic Split-K uses ``split_k_atomic=True``, ``split_k_parts``, and
    ``output_dtype="f32"``. Initialize C before each overwriting GEMM; this
    single-launch mode adds into its existing values.
    ``operand_stage_overrides`` sets independent ``mm_lhs_smem`` and
    ``mm_rhs_smem`` ring depths using the normal library pipeline.
    ``output_fragment_cols`` converts owned eight-row fragments and queues
    their TMA writeback through ``output_store_buffers`` private warp slices.
    ``output_fragment_swizzle="128B"`` uses padded N64 shared rows for
    16-bit output, including smaller tail fragments.
    ``warpgroup_ctas=2`` executes two single-warpgroup task programs per
    physical CTA, sharing operand stages with completion handoffs. Rings
    rotate across tasks when their depth does not divide the K-step count.
    ``output_grid_size`` counts the virtual 128-thread CTAs and must describe
    complete persistent waves on an even one-dimensional grid.
    """
    m = problem_size if m is None else m
    n = problem_size if n is None else n
    k = problem_size if k is None else k
    if min(m, n, k) <= 0:
        raise ValueError("problem dimensions must be positive")
    if type(warpgroup_ctas) is not int or warpgroup_ctas not in (1, 2):
        raise ValueError("warpgroup_ctas must be one or two")
    if dtype not in ("bf16", "f16", "e4m3", "e5m2"):
        raise ValueError("dtype must be bf16, f16, e4m3, or e5m2")
    rhs_dtype = dtype if rhs_dtype is None else rhs_dtype
    fp8 = dtype in ("e4m3", "e5m2")
    if rhs_dtype != dtype and not (fp8 and rhs_dtype in ("e4m3", "e5m2")):
        raise ValueError("mixed operands require two FP8 dtypes")
    output_dtype = ("bf16" if fp8 else dtype) if output_dtype is None else output_dtype
    if scaled and (not fp8 or split_k_atomic):
        raise ValueError("tensor scaling requires a non-atomic FP8 GEMM")
    if output_dtype not in ("bf16", "f16", "f32"):
        raise ValueError("output_dtype must be 'bf16', 'f16', or 'f32'")
    if output_fragment_cols is not None or fp8 and n % 64:
        output_tma_inner_size = None
    problem_name = (
        str(problem_size) if m == n == k == problem_size else f"m{m}_n{n}_k{k}"
    )
    resolved_warpgroup_n_tile = warpgroup_n_tile
    if resolved_warpgroup_n_tile is None:
        resolved_warpgroup_n_tile = output_n_tile if output_n_tile <= 256 else 128
    name = f"auto_matmul_{problem_name}_wgmma256_wg_tma_store_k{operand_stages}"
    if dtype != "bf16":
        name += f"_{dtype}"
    if rhs_dtype != dtype:
        name += f"_rhs{rhs_dtype}"
    if output_dtype != dtype:
        name += f"_out{output_dtype}"
    if scaled:
        name += "_scaled"
    if split_k_atomic:
        name += f"_splitk{split_k_parts}_atomic"
    if operand_k_tile != 64:
        name += f"_ktile{operand_k_tile}"
    if cyclic_operand_pipeline:
        name += "_kring"
    for operand, stages in sorted((operand_stage_overrides or {}).items()):
        name += f"_{operand}_k{stages}"
    if output_m_tile != 128:
        name += f"_m{output_m_tile}"
    if output_n_tile != 256:
        name += f"_n{output_n_tile}"
    if resolved_warpgroup_n_tile not in (128, 256):
        name += f"_wg{resolved_warpgroup_n_tile}"
    if warpgroup_m_tile is not None:
        name += f"_wgm{warpgroup_m_tile}"
    if cluster_m_ctas > 1 or cluster_n_ctas > 1:
        name += f"_cluster{cluster_m_ctas}x{cluster_n_ctas}"
    if output_m_tiles_per_cta != 1:
        name += f"_{output_m_tiles_per_cta}tiles"
    if output_n_tiles_per_cta != 1:
        name += f"_n{output_n_tiles_per_cta}tiles"
    if output_grid_size is not None:
        name += f"_grid{output_grid_size}"
    if output_grid_axes != (CollectiveAxis.X, CollectiveAxis.Y):
        axes = "".join(axis.name.lower() for axis in output_grid_axes)
        name += f"_grid{axes}"
    if output_slices != 1:
        name += f"_split{output_slices}"
    if output_fragment_cols is not None:
        name += f"_fragment{output_fragment_cols}"
    if output_fragment_swizzle is not None:
        name += f"_swizzle{output_fragment_swizzle}"
    if output_store_buffers != 1:
        name += f"_dbuf{output_store_buffers}"
    if output_store == "scalar":
        name += "_pre_stsm"
    elif output_store == "global":
        name += "_global_store"
    if output_reversed_dims:
        name += "_rev" + "".join(output_reversed_dims).lower()
    if warpgroup_ctas != 1:
        name += f"_wgctas{warpgroup_ctas}"

    module = gpu_dialect.parse(
        textwrap.dedent(f"""\
        module {{
          kernel @{name}(%a: gmem<{m}x{k}x{dtype}>, %b: gmem<{n}x{k}x{rhs_dtype}>, %c: gmem<{m}x{n}x{output_dtype}>) attrs={{ptx_optimization_level: {ptx_optimization_level}}} {{
            %zero: rmem<f32> = constant 0.0
            matmul(%a: gmem<{m}x{k}x{dtype}>, %b: gmem<{n}x{k}x{rhs_dtype}>, %c: gmem<{m}x{n}x{output_dtype}>, %zero: rmem<f32>) collective=xyz.grid attrs={{label: "mm"}}
          }}
        }}""")
    )
    assert isinstance(module, ModuleOp)
    module = TileMatmulPipeline(
        m=m,
        n=n,
        k=k,
        output_m_tiles_per_cta=output_m_tiles_per_cta,
        output_n_tiles_per_cta=output_n_tiles_per_cta,
        output_grid_size=output_grid_size,
        output_grid_fast_axis=output_grid_fast_axis,
        output_grid_axes=output_grid_axes,
        output_reversed_dims=output_reversed_dims,
        output_m_tile=output_m_tile,
        output_n_tile=output_n_tile,
        warpgroup_m_tile=warpgroup_m_tile,
        warpgroup_n_tile=resolved_warpgroup_n_tile,
        cluster_m_ctas=cluster_m_ctas,
        cluster_n_ctas=cluster_n_ctas,
        operand_stages=operand_stages,
        operand_stage_overrides=operand_stage_overrides,
        operand_k_tile=operand_k_tile,
        cyclic_operand_pipeline=cyclic_operand_pipeline,
        warpgroup_ctas=warpgroup_ctas,
        output_slices=output_slices,
        output_store_buffers=output_store_buffers,
        output_fragment_cols=output_fragment_cols,
        output_fragment_swizzle=output_fragment_swizzle,
        output_tma_inner_size=output_tma_inner_size,
        output_store=output_store,
        unsafe_reuse_smem=unsafe_reuse_smem,
        operand_move_attrs=operand_move_attrs,
        split_k_parts=split_k_parts,
        split_k_atomic=split_k_atomic,
    ).run(module)

    if scaled:
        _add_tensor_scales(module)
    return module


def _add_tensor_scales(module: ModuleOp) -> None:
    """Scale FP32 accumulators before output conversion or writeback.

    Scalar device loads stay outside the tile/K loops. Ordinary IR arithmetic
    then follows each accumulator's existing ownership and synchronization.
    """
    [kernel] = module.body.ops
    assert isinstance(kernel, KernelOp)
    loads = []
    scales = []
    for name in ("scale_a", "scale_b"):
        arg = Value(name=name, type=GmemTensorType.build([], F32Type()))
        kernel.body.append_argument(arg)
        reg = AllocOp.build_with_fresh_id(RmemTensorType.build([], F32Type()))
        loads.extend(
            [
                reg,
                MoveOp.build(
                    reg.result(),
                    arg,
                    collective={
                        CollectiveAxis.X: CollectiveLevel.THREAD,
                        CollectiveAxis.Y: CollectiveLevel.GRID,
                        CollectiveAxis.Z: CollectiveLevel.GRID,
                    },
                ),
            ]
        )
        scales.append(reg.result())
    scale = AllocOp.build_with_fresh_id(RmemTensorType.build([], F32Type()))
    loads.extend([scale, ElementwiseOp.build(scale.result(), scales, "mul")])
    kernel.body.ops[0:0] = loads
    # Every fragment is scaled once, including sliced writeback and FP32 output.
    targets = [
        op
        for op in find_ops_of_type(module, ElementwiseOp)
        if op.fn == "convert"
        and isinstance(op.src(0).type, WgrmemTensorType)
        and op.src(0).type.dtype == F32Type()
    ]
    if not targets:
        targets = [
            op
            for op in find_ops_of_type(module, MoveOp)
            if isinstance(op.src.type, WgrmemTensorType)
            and op.src.type.dtype == F32Type()
        ]
    for op in targets:
        src = op.src(0) if isinstance(op, ElementwiseOp) else op.src
        assert op.parent_block is not None
        idx = op.parent_block.ops.index(op)
        op.parent_block.ops.insert(
            idx, ElementwiseOp.build(src, [src, scale.result()], "mul")
        )


def make_fp8_gemm_kernel(**config):
    """Compile FP8 with Ayla's cooperative GEMM pipeline."""
    from ..ptx_codegen import make_ptx_kernel

    return make_ptx_kernel(auto_matmul_4096_wgmma256_wg_tma_store_module(**config))
