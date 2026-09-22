"""Shape-routed kernels for the Muse Glimmer SwiGLU MLP."""
from __future__ import annotations

import torch

from ..base.ir import Block, Op, Value
from ..gpu_dialect import (
    AllocOp,
    BF16Type,
    BarrierOp,
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
from .gemm import (
    auto_matmul_4096_wgmma256_wg_tma_store_module,
    schedule_register_matmul,
    schedule_tiled_matmul,
)
from ..passes.canonicalize import Canonicalize
from ..passes.fuse_async import FuseAsync
from ..passes.lower_to_async import LowerToAsync
from ..passes.schedule_async import ScheduleAsync
from ..passes.stage_register_writeback import StageRegisterWriteback
from ..passes.tile_matmul import TileMatmul
from ..ptx_codegen import make_ptx_kernel
from ..ptx_runtime import make_ptx_launcher
from ..torch_custom_call import DEFAULT_NAMESPACE, OperandKind, make_torch_custom_call


_HIDDEN = 6_656
_INTERMEDIATE = 19_968
_PACKED_INTERMEDIATE = 2 * _INTERMEDIATE
MUSE_SWIGLU_M_VALUES = (1, 4, 32, 64, 512, 2048)

_LARGE_M_VALUES = (512, 2048)
_OUTPUT_M_TILE = 64
_OUTPUT_N_TILE = 192
_RESIDENT_BLOCKS = 104


def _retained_cache_policy(
    *, keep_lhs: bool,
) -> dict[str, dict[str, object]]:
    if keep_lhs:
        return {
            "lhs": {
                "l2_cache_hint": "evict_last",
                "l2_promotion": "256B",
            },
            "rhs": {"l2_cache_hint": "evict_first"},
            "dst": {"l2_cache_hint": "evict_first"},
        }
    return {
        "lhs": {"l2_promotion": "256B"},
        "rhs": {"l2_promotion": "256B"},
        "dst": {"l2_cache_hint": "evict_first"},
    }


# Retained winners from the historical Muse cold-cache sweep. These are
# benchmark choices, not general-purpose dynamic-M heuristics.
MUSE_SWIGLU_LARGE_M_CONFIGS: dict[tuple[int, str], dict[str, object]] = {
    (512, "projection"): {
        "output_m_tile": 128,
        "output_n_tile": 208,
        "operand_k_tile": 64,
        "operand_stages": 4,
        "output_tma_inner_size": None,
        "operand_move_attrs": _retained_cache_policy(keep_lhs=True),
        "ptx_optimization_level": 3,
    },
    (512, "down"): {
        "output_m_tile": 128,
        "output_n_tile": 208,
        "operand_k_tile": 64,
        "operand_stages": 4,
        "output_tma_inner_size": None,
        "operand_move_attrs": _retained_cache_policy(keep_lhs=False),
        "ptx_optimization_level": 3,
    },
    (2048, "down"): {
        "output_m_tile": 192,
        "output_n_tile": 192,
        "operand_k_tile": 64,
        "operand_stages": 3,
        "output_slices": 3,
        "output_tma_inner_size": 64,
        "operand_move_attrs": _retained_cache_policy(keep_lhs=False),
        "ptx_optimization_level": 2,
    },
}


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


def _build_swiglu_projection(
    lhs: Value,
    packed_weight: Value,
    output: Value,
) -> tuple[list[Op], MoveOp]:
    """Build the paired projection and SwiGLU region for this benchmark."""
    m = lhs.type.sizes[0]
    block_ids = BlockIdsOp.build()
    gate_weight = _tile(
        packed_weight, [_INTERMEDIATE, _HIDDEN], [0, 0],
    )
    up_weight = _tile(
        packed_weight, [_INTERMEDIATE, _HIDDEN], [1, 0],
    )
    lhs_tile = _tile(
        lhs,
        [_OUTPUT_M_TILE, _HIDDEN],
        [block_ids.y, 0],
        maybe_partial=m is None or m % _OUTPUT_M_TILE != 0,
    )
    gate_tile = _tile(
        gate_weight.result(), [_OUTPUT_N_TILE, _HIDDEN], [block_ids.x, 0],
    )
    up_tile = _tile(
        up_weight.result(), [_OUTPUT_N_TILE, _HIDDEN], [block_ids.x, 0],
    )
    output_tile = _tile(
        output,
        [_OUTPUT_M_TILE, _OUTPUT_N_TILE],
        [block_ids.y, block_ids.x],
        maybe_partial=m is None or m % _OUTPUT_M_TILE != 0,
    )

    accumulator_type = WrmemTensorType.build(
        [_OUTPUT_M_TILE, _OUTPUT_N_TILE],
        F32Type(),
        warps=4,
        layout="wgmma_layout_0",
    )
    epilogue_type = WrmemTensorType.build(
        [_OUTPUT_M_TILE, _OUTPUT_N_TILE],
        output.type.dtype,
        warps=4,
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
        lhs_tile.result(),
        gate_tile.result(),
        gate.result(),
        collective=collective,
        attrs={"label": "gate"},
    )
    up_matmul = MatmulOp.build(
        lhs_tile.result(),
        up_tile.result(),
        up.result(),
        collective=collective,
        attrs={"label": "up"},
    )
    round_gate = ElementwiseOp.build(
        gate_activation.result(), [gate.result()], "convert",
    )
    widen_gate = ElementwiseOp.build(
        gate.result(), [gate_activation.result()], "convert",
    )
    silu = ElementwiseOp.build(
        gate_activation.result(),
        [gate.result()],
        "silu",
        attrs={"label": "gate_epi"},
    )
    round_up = ElementwiseOp.build(
        product.result(), [up.result()], "convert",
    )
    multiply = ElementwiseOp.build(
        product.result(),
        [product.result(), gate_activation.result()],
        "mul",
        attrs={"label": "up_epi"},
    )
    writeback = MoveOp.build(
        output_tile.result(),
        product.result(),
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


def _schedule_swiglu_projection(
    module: ModuleOp,
    writeback: MoveOp,
) -> None:
    move_attrs = {
        "lhs": {
            "l2_cache_hint": "evict_last",
            "l2_promotion": "128B",
        },
        "rhs": {
            "l2_cache_hint": "evict_first",
            "l2_promotion": "256B",
        },
    }
    for label in ("gate", "up"):
        schedule_register_matmul(
            label=label,
            operand_stages=3,
            operand_k_tile=64,
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
    Canonicalize().run(module)


def _schedule_down_projection(
    module: ModuleOp,
    *,
    operand_stages: int,
) -> None:
    TileMatmul(
        op_dims=["M", "N"],
        tile_sizes=[_OUTPUT_M_TILE, 64],
        iteration_bindings=[
            None,
            (CollectiveLevel.BLOCK, CollectiveAxis.X),
        ],
        for_loop_labels=["down_m", None],
        op_label_filter="down",
        allow_partial_tiles=True,
    ).run(module)
    schedule_tiled_matmul(
        label="down",
        output_m_tile=_OUTPUT_M_TILE,
        output_n_tile=64,
        operand_stages=operand_stages,
        operand_k_tile=64,
        output_tma_inner_size=None,
        unsafe_reuse_smem=True,
        operand_move_attrs={
            "lhs": {
                "l2_cache_hint": "evict_last",
                "l2_promotion": "128B",
            },
            "rhs": {
                "l2_cache_hint": "evict_first",
                "l2_promotion": "256B",
            },
            "dst": {"l2_cache_hint": "evict_first"},
        },
        loop_label_prefix="down",
    ).run(module)


def muse_swiglu_module(*, m: int) -> ModuleOp:
    """Build the one-wave fused Muse MLP used for small M."""
    if m not in (1, 4, 32, 64):
        raise ValueError(f"unsupported small M={m}")
    down_operand_stages = 5 if m <= 32 else 6

    bf16 = BF16Type()
    lhs = Value("lhs", GmemTensorType.build([m, _HIDDEN], bf16))
    packed_weight = Value(
        "packed_weight",
        GmemTensorType.build([_PACKED_INTERMEDIATE, _HIDDEN], bf16),
    )
    down_weight = Value(
        "down_weight",
        GmemTensorType.build([_HIDDEN, _INTERMEDIATE], bf16),
    )
    hidden = Value(
        "hidden", GmemTensorType.build([m, _INTERMEDIATE], bf16),
    )
    output = Value("output", GmemTensorType.build([m, _HIDDEN], bf16))
    sync = Value("sync", GmemTensorType.build([2], F32Type()))
    down = MatmulOp.build(
        hidden,
        down_weight,
        output,
        collective={axis: CollectiveLevel.GRID for axis in CollectiveAxis},
        init=0.0,
        attrs={"label": "down"},
    )
    projection_ops, projection_writeback = _build_swiglu_projection(
        lhs, packed_weight, hidden,
    )
    kernel = KernelOp.build(
        f"muse_glimmer_swiglu_m{m}",
        Block(
            arguments=[
                lhs, packed_weight, down_weight, hidden, output, sync,
            ],
            ops=[
                *projection_ops,
                BarrierOp.build("grid", sync=sync),
                down,
            ],
        ),
        grid_dim=(_RESIDENT_BLOCKS, 1, 1),
        block_dim=(128, 1, 1),
        num_warp_groups=1,
        ptx_optimization_level=4,
    )
    module = ModuleOp.build(Block(ops=[kernel]))
    _schedule_swiglu_projection(module, projection_writeback)
    _schedule_down_projection(module, operand_stages=down_operand_stages)
    ScheduleAsync().run(module)
    return Canonicalize().run(module)


def _torch_projection_swiglu(
    lhs: torch.Tensor,
    packed_weight: torch.Tensor,
) -> torch.Tensor:
    projection = lhs @ packed_weight.T
    gate = projection[:, :_INTERMEDIATE]
    up = projection[:, _INTERMEDIATE:]
    return torch.nn.functional.silu(gate) * up


def _swiglu_from_projection(projection: torch.Tensor) -> torch.Tensor:
    gate = projection[:, :_INTERMEDIATE]
    up = projection[:, _INTERMEDIATE:]
    return torch.nn.functional.silu(gate) * up


def _gemm_launcher(
    *,
    m: int,
    n: int,
    k: int,
    config: dict[str, object],
):
    return make_ptx_launcher(make_ptx_kernel(
        auto_matmul_4096_wgmma256_wg_tma_store_module(
            m=m, n=n, k=k, **config,
        ),
    ))


def _large_m_ayla_swiglu(m: int):
    down = _gemm_launcher(
        m=m,
        n=_HIDDEN,
        k=_INTERMEDIATE,
        config=MUSE_SWIGLU_LARGE_M_CONFIGS[(m, "down")],
    )
    if m == 2048:
        # At M=2048 the archived Ayla projection is not profitable with the
        # current, simplified GEMM builder. Preserve the normal backend
        # fallback and replace only the down projection, which still wins.
        projection_swiglu = torch.compile(
            _torch_projection_swiglu,
            fullgraph=True,
            dynamic=False,
        )
    else:
        projection = _gemm_launcher(
            m=m,
            n=_PACKED_INTERMEDIATE,
            k=_HIDDEN,
            config=MUSE_SWIGLU_LARGE_M_CONFIGS[(m, "projection")],
        )
        projection_output = torch.empty(
            (m, _PACKED_INTERMEDIATE),
            device="cuda",
            dtype=torch.bfloat16,
        )
        activation = torch.compile(
            _swiglu_from_projection,
            fullgraph=True,
            dynamic=False,
        )

        def projection_swiglu(lhs, packed_weight):
            projection(lhs, packed_weight, projection_output)
            return activation(projection_output)

    def run(lhs, packed_weight, down_weight, output):
        down(projection_swiglu(lhs, packed_weight), down_weight, output)

    return run


def make_muse_swiglu_launcher(m: int):
    """Build the tuned out-parameter route for one supported Muse M."""
    if m not in MUSE_SWIGLU_M_VALUES:
        raise ValueError(
            f"unsupported M={m}; expected one of {MUSE_SWIGLU_M_VALUES}"
        )
    if m in _LARGE_M_VALUES:
        return _large_m_ayla_swiglu(m)

    hidden = torch.empty(
        (m, _INTERMEDIATE), device="cuda", dtype=torch.bfloat16,
    )
    sync = torch.zeros(2, device="cuda", dtype=torch.float32)
    launcher = make_ptx_launcher(make_ptx_kernel(
        muse_swiglu_module(m=m),
    ))

    def run(lhs, packed_weight, down_weight, output):
        launcher(lhs, packed_weight, down_weight, hidden, output, sync)

    return run


class _MuseSwiGLULauncher:
    """Adapt one tuned route to the repository's functional custom-op API."""

    def __init__(self, m: int) -> None:
        self.name = f"muse_swiglu_m{m}"
        bf16 = BF16Type()
        self.signature = [
            GmemTensorType.build([m, _HIDDEN], bf16),
            GmemTensorType.build([_PACKED_INTERMEDIATE, _HIDDEN], bf16),
            GmemTensorType.build([_HIDDEN, _INTERMEDIATE], bf16),
            GmemTensorType.build([m, _HIDDEN], bf16),
        ]
        self._run = make_muse_swiglu_launcher(m)

    def __call__(self, *tensors: torch.Tensor) -> None:
        self._run(*tensors)


def make_muse_swiglu(
    m: int,
    *,
    namespace: str = DEFAULT_NAMESPACE,
):
    """Create the functional custom op selected for a concrete Muse M."""
    return make_torch_custom_call(
        _MuseSwiGLULauncher(m),
        [OperandKind.R, OperandKind.R, OperandKind.R, OperandKind.RESULT],
        namespace=namespace,
    )
