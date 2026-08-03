"""Select grouped STMatrix stores from layout-compatible generic moves."""
from __future__ import annotations

from ..base.ir import Op
from ..base.lowering_pass import LoweringPass
from ..gpu_dialect import (
    BF16Type,
    F16Type,
    ModuleOp,
    MoveOp,
    SmemTensorType,
    StMatrixStoreOp,
    WrmemTensorType,
    wgmma_layout_0_static_offsets,
)


class SelectStMatrix(LoweringPass):
    """Rewrite eligible WRMEM→SMEM moves to an explicit target IR op."""

    def __init__(self) -> None:
        super().__init__()
        self.register(MoveOp, type(self).lower_move)

    def lower_move(self, op: MoveOp) -> tuple[list[Op], list[object]] | None:
        dst_type = op.dst.type
        src_type = op.src.type
        if not (
            isinstance(dst_type, SmemTensorType)
            and isinstance(src_type, WrmemTensorType)
            and src_type.layout == "wgmma_layout_0"
            and isinstance(src_type.dtype, (BF16Type, F16Type))
            and dst_type.dtype == src_type.dtype
            and list(dst_type.sizes) == list(src_type.sizes)
            and dst_type.physical_layout is not None
        ):
            return None

        layout = dst_type.physical_layout
        assert layout is not None and layout.get("kind") == "blocked_padding"
        m_dim, n_dim = src_type.sizes
        if m_dim % 64 or n_dim % 16 or not (8 <= n_dim <= 256):
            return None

        num_registers = src_type.num_elements_per_thread // 2
        if num_registers % 4:
            return None
        block = int(layout["block"])
        padding = int(layout["padding"])
        offsets = []
        for reg in range(0, num_registers, 4):
            di, dj = wgmma_layout_0_static_offsets(reg, n_dim)
            physical_dj = (dj // block) * (block + padding) + dj % block
            offsets.append(
                (di * dst_type.strides[0] + physical_dj)
                * dst_type.dtype.byte_size
            )
        return [StMatrixStoreOp.build(
            self.value_map[op.dst], self.value_map[op.src], offsets,
            attrs=dict(op.attrs),
        )], []


def select_stmatrix(module: ModuleOp) -> ModuleOp:
    return SelectStMatrix().run(module)
