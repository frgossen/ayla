"""Compose a logical M192 tile from native M128 and M64 updates."""

from ..base.ir import Op, Value, clone_type
from ..base.lowering_pass import LoweringPass, lowers
from ..gpu_dialect import MatmulOp, MemTensorType, TileOp
from ._utils import label_matches


class ComposeMatmulRows(LoweringPass):
    """Split after operand promotion so both updates share each loaded B panel.

    The destination is still address-backed here. Loop-scoped promotion gives
    each disjoint row region its own native TMEM accumulator and retains both
    through the reduction, without representing 192 logical rows in TMEM.
    """

    label_filter: str

    def __init__(self, *, label_filter: str):
        super().__init__(label_filter=label_filter)

    @lowers(MatmulOp)
    def lower_matmul(self, op: MatmulOp) -> tuple[list[Op], list[object]] | None:
        if not label_matches(op, self.label_filter):
            return None
        assert isinstance(op.lhs.type, MemTensorType)
        if op.lhs.type.sizes[0] != 192:
            raise ValueError("M192 composition requires a 192-row operand")
        result: list[Op] = []
        for rows, origin in ((128, 0), (64, 128)):

            def view(value: Value) -> Value:
                assert isinstance(value.type, MemTensorType)
                shape = [rows, value.type.sizes[1]]
                tile = TileOp.build(
                    value, [origin // rows, 0], clone_type(value.type, {"sizes": shape})
                )
                result.append(tile)
                return tile.result()

            lhs, dst = view(op.lhs), view(op.dst)
            init = op.init
            if init is not None:
                assert isinstance(init.type, MemTensorType)
                if init.type.rank:
                    init = view(init)
            result.append(
                MatmulOp.build(
                    lhs=lhs,
                    rhs=op.rhs,
                    dst=dst,
                    init=init,
                    collective=op.collective,
                    attrs=op.extra_attrs,
                )
            )
        return result, []
