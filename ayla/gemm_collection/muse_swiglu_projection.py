"""Standalone Muse Glimmer dual-projection SwiGLU kernel."""
from __future__ import annotations

from .. import ptx_runtime, torch_custom_call
from ..base.ir import Block, Op, Value
from ..gpu_dialect import (
    AllocOp,
    BarrierOp,
    BF16Type,
    BlockIdsOp,
    CollectiveAxis,
    CollectiveLevel,
    ElementwiseOp,
    F32Type,
    GmemTensorType,
    KernelOp,
    MatmulOp,
    ModuleOp,
    MoveOp,
    TileOp,
    WrmemTensorType,
)
from ..jit_compiler import ShapeSpecializingJITCompiler
from ..passes.canonicalize import Canonicalize
from ..passes.fuse_async import FuseAsync
from ..passes.lower_to_async import LowerToAsync
from ..passes.schedule_async import ScheduleAsync
from ..passes.stage_register_writeback import StageRegisterWriteback
from ..ptx_codegen import make_ptx_kernel
from ..torch_custom_call import DEFAULT_NAMESPACE, OperandKind
from .gemm import schedule_register_matmul


_HIDDEN = 6_656
_INTERMEDIATE = 19_968
_PACKED_INTERMEDIATE = 2 * _INTERMEDIATE
_M_TILE = 64
_N_TILE = 192
_K_TILE = 64
_OPERAND_STAGES = 3


def _tile(
    value: Value,
    sizes: list[int],
    indices: list[Value | int],
    *,
    maybe_partial: bool = False,
) -> TileOp:
    typ = GmemTensorType.build(
        sizes,
        value.type.dtype,
        strides=list(value.type.strides),
    )
    return TileOp.build(
        value, indices, typ, maybe_partial=maybe_partial,
    )


def build_muse_swiglu_projection(
    lhs: Value,
    packed_weight: Value,
    output: Value,
) -> tuple[list[Op], MoveOp]:
    """Build the concrete pre-tiled two-GEMM projection region."""
    m = lhs.type.sizes[0]
    if (
        lhs.type.sizes != [m, _HIDDEN]
        or packed_weight.type.sizes != [_PACKED_INTERMEDIATE, _HIDDEN]
        or output.type.sizes != [m, _INTERMEDIATE]
    ):
        raise ValueError("unexpected Muse SwiGLU operand shapes")
    block_ids = BlockIdsOp.build()
    gate_weight = _tile(
        packed_weight, [_INTERMEDIATE, _HIDDEN], [0, 0],
    )
    up_weight = _tile(
        packed_weight, [_INTERMEDIATE, _HIDDEN], [1, 0],
    )
    partial_m = m is None or m % _M_TILE != 0
    lhs_tile = _tile(
        lhs, [_M_TILE, _HIDDEN], [block_ids.y, 0],
        maybe_partial=partial_m,
    )
    gate_tile = _tile(
        gate_weight.result(), [_N_TILE, _HIDDEN], [block_ids.x, 0],
    )
    up_tile = _tile(
        up_weight.result(), [_N_TILE, _HIDDEN], [block_ids.x, 0],
    )
    output_tile = _tile(
        output, [_M_TILE, _N_TILE], [block_ids.y, block_ids.x],
        maybe_partial=partial_m,
    )

    accumulator_type = WrmemTensorType.build(
        [_M_TILE, _N_TILE], F32Type(), warps=4,
        layout="wgmma_layout_0",
    )
    epilogue_type = WrmemTensorType.build(
        [_M_TILE, _N_TILE], output.type.dtype, warps=4,
        layout="wgmma_layout_0",
    )
    gate = AllocOp.build_with_fresh_id(accumulator_type, init=0.0)
    up = AllocOp.build_with_fresh_id(accumulator_type, init=0.0)
    gate_activation = AllocOp.build_with_fresh_id(epilogue_type)
    product = AllocOp.build_with_fresh_id(epilogue_type)
    collective = {
        CollectiveAxis.X: CollectiveLevel.WG,
        CollectiveAxis.Y: CollectiveLevel.GRID,
        CollectiveAxis.Z: CollectiveLevel.GRID,
    }
    gate_matmul = MatmulOp.build(
        lhs_tile.result(), gate_tile.result(), gate.result(),
        collective=collective, attrs={"label": "gate"},
    )
    up_matmul = MatmulOp.build(
        lhs_tile.result(), up_tile.result(), up.result(),
        collective=collective, attrs={"label": "up"},
    )
    round_gate = ElementwiseOp.build(
        gate_activation.result(), [gate.result()], "convert",
    )
    widen_gate = ElementwiseOp.build(
        gate.result(), [gate_activation.result()], "convert",
    )
    silu = ElementwiseOp.build(
        gate_activation.result(), [gate.result()], "silu",
        attrs={"label": "gate_epi"},
    )
    round_up = ElementwiseOp.build(
        product.result(), [up.result()], "convert",
    )
    multiply = ElementwiseOp.build(
        product.result(), [product.result(), gate_activation.result()], "mul",
        attrs={"label": "up_epi"},
    )
    writeback = MoveOp.build(
        output_tile.result(), product.result(),
        collective=collective,
        attrs={
            "label": "up_dst_wrmem",
            "l2_cache_hint": "evict_last",
        },
    )

    return [
        block_ids,
        gate_weight,
        up_weight,
        lhs_tile,
        gate_tile,
        up_tile,
        output_tile,
        gate,
        up,
        gate_activation,
        product,
        gate_matmul,
        BarrierOp.build("thread"),
        up_matmul,
        round_gate,
        widen_gate,
        silu,
        round_up,
        multiply,
        writeback,
    ], writeback


def schedule_muse_swiglu_projection(
    module: ModuleOp, writeback: MoveOp,
) -> ModuleOp:
    """Schedule the pre-tiled projection region in ``module``."""
    move_attrs = {
        "lhs": {"l2_cache_hint": "evict_last", "l2_promotion": "128B"},
        "rhs": {"l2_cache_hint": "evict_first", "l2_promotion": "256B"},
    }
    for label in ("gate", "up"):
        schedule_register_matmul(
            label=label,
            operand_stages=_OPERAND_STAGES,
            operand_k_tile=_K_TILE,
            unsafe_reuse_smem=True,
            operand_move_attrs=move_attrs,
            loop_label_prefix=label,
        ).run(module)

    writeback.attrs["tma"] = True
    StageRegisterWriteback(
        unsafe_reuse_smem=True,
        label_filter="up_dst_wrmem",
    ).run(module)
    LowerToAsync().run(module)
    FuseAsync().run(module)
    ScheduleAsync().run(module)
    return Canonicalize().run(module)


def muse_swiglu_projection_module(*, m: int) -> ModuleOp:
    """Build ``silu(a @ gate.T) * (a @ up.T)`` for packed gate/up weights.

    This is intentionally a concrete two-hero builder. Both projections use
    the same block tile coordinates, retain their accumulators in WRMEM, and
    materialize only the product.
    """
    if m <= 0:
        raise ValueError("M must be positive")

    bf16 = BF16Type()
    lhs = Value("lhs", GmemTensorType.build([m, _HIDDEN], bf16))
    packed_weight = Value(
        "packed_weight",
        GmemTensorType.build([_PACKED_INTERMEDIATE, _HIDDEN], bf16),
    )
    output = Value(
        "output", GmemTensorType.build([m, _INTERMEDIATE], bf16),
    )
    ops, writeback = build_muse_swiglu_projection(
        lhs, packed_weight, output,
    )
    kernel = KernelOp.build(
        f"muse_swiglu_projection_m{m}",
        Block(
            arguments=[lhs, packed_weight, output],
            ops=ops,
        ),
        grid_dim=(_INTERMEDIATE // _N_TILE, (m + _M_TILE - 1) // _M_TILE, 1),
        block_dim=(128, 1, 1),
        num_warp_groups=1,
        ptx_optimization_level=4,
    )
    module = ModuleOp.build(Block(ops=[kernel]))
    return schedule_muse_swiglu_projection(module, writeback)


def muse_swiglu_projection_template() -> ModuleOp:
    """Build the dynamic-M template used by shape-specialized integration."""
    bf16 = BF16Type()
    lhs = Value("lhs", GmemTensorType.build([None, _HIDDEN], bf16))
    packed_weight = Value(
        "packed_weight",
        GmemTensorType.build([_PACKED_INTERMEDIATE, _HIDDEN], bf16),
    )
    output = Value(
        "output", GmemTensorType.build([None, _INTERMEDIATE], bf16),
    )
    ops, _ = build_muse_swiglu_projection(lhs, packed_weight, output)
    kernel = KernelOp.build(
        "dynamic_m_muse_swiglu_projection_out",
        Block(arguments=[lhs, packed_weight, output], ops=ops),
        grid_dim=(_INTERMEDIATE // _N_TILE, None, 1),
        block_dim=(128, 1, 1),
        num_warp_groups=1,
        ptx_optimization_level=4,
    )
    return ModuleOp.build(Block(ops=[kernel]))


def _compile_muse_swiglu_projection(module: ModuleOp):
    [kernel] = module.body.ops
    assert isinstance(kernel, KernelOp)
    m = kernel.signature[0].sizes[0]
    assert m is not None
    kernel.grid_dim = (
        _INTERMEDIATE // _N_TILE,
        (m + _M_TILE - 1) // _M_TILE,
        1,
    )
    [writeback] = [
        op for op in kernel.body.ops
        if isinstance(op, MoveOp)
        and op.attrs.get("label") == "up_dst_wrmem"
    ]
    schedule_muse_swiglu_projection(module, writeback)
    return ptx_runtime.make_ptx_launcher(make_ptx_kernel(module))


def make_dynamic_m_muse_swiglu_projection(
    *,
    namespace: str = DEFAULT_NAMESPACE,
):
    """Create a cached functional projection op for dynamic M."""
    compiler = ShapeSpecializingJITCompiler(
        muse_swiglu_projection_template(),
        _compile_muse_swiglu_projection,
    )
    return torch_custom_call.make_torch_custom_call(
        compiler,
        [OperandKind.R, OperandKind.R, OperandKind.RESULT],
        namespace=namespace,
        name="dynamic_m_muse_swiglu_projection",
        infer_result_shapes=lambda lhs, _weight: [(lhs.shape[0], _INTERMEDIATE)],
    )
