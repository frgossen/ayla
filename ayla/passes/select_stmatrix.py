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
            and dst_type.physical_layout is not None
        ):
            return None

        assert dst_type.blocked_layout is not None
        m_dim, n_dim = src_type.sizes
        col_start = int(op.attrs.get("wgmma_col_start", 0))
        dst_m, dst_n = dst_type.sizes
        if dst_m != m_dim or col_start < 0 or col_start + dst_n > n_dim:
            return None
        if m_dim % 64 or n_dim % 16 or not (8 <= n_dim <= 256):
            return None

        num_registers = src_type.num_elements_per_thread // 2
        if num_registers % 4:
            return None
        register_starts = []
        offsets = []
        for reg in range(0, num_registers, 4):
            di, dj = wgmma_layout_0_static_offsets(reg, n_dim)
            if not col_start <= dj < col_start + dst_n:
                continue
            logical_dj = dj - col_start
            register_starts.append(reg)
            offsets.append(
                dst_type.element_offset([di, logical_dj]) * dst_type.dtype.byte_size
            )
        return [StMatrixStoreOp.build(
            self.value_map[op.dst], self.value_map[op.src],
            register_starts, offsets,
            attrs=dict(op.attrs),
        )], []


def select_stmatrix(module: ModuleOp) -> ModuleOp:
    return SelectStMatrix().run(module)
