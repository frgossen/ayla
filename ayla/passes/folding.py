"""Folding pass: constant folding and local simplifications.

Folds index arithmetic where operands are known constants,
eliminates trivial operations like `x * 1`, `x + 0`, `0 * n`,
`x * 0`, and removes no-op tile ops.
"""
from __future__ import annotations

from ..gpu_dialect import (
    ForOp,
    IndexAddOp,
    IndexConstantOp,
    IndexDivOp,
    IndexModOp,
    IndexMulOp,
    ModuleOp,
    PtrAddOp,
    TileOp,
)
from ..ir import Op, Value, clone_op
from ..lowering_pass import LoweringPass


class Folding(LoweringPass):

    def __init__(self) -> None:
        super().__init__()
        self.register(ForOp, type(self).lower_for)
        self.register(IndexAddOp, type(self).lower_add)
        self.register(IndexMulOp, type(self).lower_mul)
        self.register(IndexDivOp, type(self).lower_div)
        self.register(IndexModOp, type(self).lower_mod)
        self.register(PtrAddOp, type(self).lower_ptr_add)
        self.register(TileOp, type(self).lower_tile)

    def lower_for(
        self, op: ForOp,
    ) -> tuple[list[Op], list[object]] | None:
        if op.trip_count != 1:
            return None
        assert len(op.body.blocks) == 1
        self.lower_region(op.body)
        iv_const = IndexConstantOp.build(value=op.lb)
        vmap: dict[Value, Value] = {op.iv: iv_const.result()}
        cloned = [clone_op(body_op, vmap) for body_op in op.body.entry_block.ops]
        return [iv_const, *cloned], []

    def _const_val(self, value: Value) -> int | None:
        mapped = self.value_map.get(value, value)
        if isinstance(mapped, Value) and isinstance(mapped.definition, IndexConstantOp):
            return mapped.definition.value
        return None

    def _decompose_add(self, value: Value) -> tuple[Value | None, int]:
        """Strip the constant part from a value.

        If value is a constant, returns (None, const).
        If value is an add with a constant rhs, returns (lhs, const).
        Otherwise returns (value, 0).
        """
        mapped = self.value_map.get(value, value)
        const = self._const_val(value)
        if const is not None:
            return None, const
        if isinstance(mapped.definition, IndexAddOp):
            rhs_const = self._const_val(mapped.definition.rhs)
            if rhs_const is not None:
                lhs = self.value_map.get(mapped.definition.lhs, mapped.definition.lhs)
                return lhs, rhs_const
        return mapped, 0

    def lower_add(
        self, op: IndexAddOp,
    ) -> tuple[list[Op], list[object]] | None:
        lhs_non_const, lhs_const = self._decompose_add(op.lhs)
        rhs_non_const, rhs_const = self._decompose_add(op.rhs)
        const_sum = lhs_const + rhs_const

        # Nothing to fold if neither side had a constant.
        lhs_has_const = lhs_non_const is None or lhs_const != 0
        rhs_has_const = rhs_non_const is None or rhs_const != 0
        if not lhs_has_const and not rhs_has_const:
            return None
        # Already normalized: rhs is a bare constant, lhs has no deeper constant,
        # and the constant isn't 0 (which should be eliminated).
        if lhs_non_const is not None and rhs_non_const is None and lhs_const == 0 and rhs_const != 0:
            return None

        # Both sides fully constant.
        if lhs_non_const is None and rhs_non_const is None:
            c = IndexConstantOp.build(const_sum)
            return [c], [c.result()]

        # Build sum from non-constant parts.
        new_ops: list[Op] = []
        if lhs_non_const is None:
            result = rhs_non_const
        elif rhs_non_const is None:
            result = lhs_non_const
        else:
            add = IndexAddOp.build(lhs_non_const, rhs_non_const)
            new_ops.append(add)
            result = add.result()

        # Append combined constant on the rhs.
        if const_sum != 0:
            c = IndexConstantOp.build(const_sum)
            add = IndexAddOp.build(result, c.result())
            new_ops.extend([c, add])
            result = add.result()

        return new_ops, [result]

    def _decompose_mul(self, value: Value) -> tuple[Value | None, int]:
        """Strip the constant factor from a value.

        If value is a constant, returns (None, const).
        If value is a mul with a constant rhs, returns (lhs, const).
        Otherwise returns (value, 1).
        """
        mapped = self.value_map.get(value, value)
        const = self._const_val(value)
        if const is not None:
            return None, const
        if isinstance(mapped.definition, IndexMulOp):
            rhs_const = self._const_val(mapped.definition.rhs)
            if rhs_const is not None:
                lhs = self.value_map.get(mapped.definition.lhs, mapped.definition.lhs)
                return lhs, rhs_const
        return mapped, 1

    def lower_mul(
        self, op: IndexMulOp,
    ) -> tuple[list[Op], list[object]] | None:
        lhs_non_const, lhs_const = self._decompose_mul(op.lhs)
        rhs_non_const, rhs_const = self._decompose_mul(op.rhs)
        const_prod = lhs_const * rhs_const

        # Nothing to fold if neither side had a constant.
        lhs_has_const = lhs_non_const is None or lhs_const != 1
        rhs_has_const = rhs_non_const is None or rhs_const != 1
        if not lhs_has_const and not rhs_has_const:
            return None
        # Already normalized: rhs is a bare constant, lhs has no deeper constant,
        # and the constant isn't a special value (0 or 1) that could be simplified.
        if lhs_non_const is not None and rhs_non_const is None and lhs_const == 1 and rhs_const not in (0, 1):
            return None

        # Zero kills everything.
        if const_prod == 0:
            c = IndexConstantOp.build(0)
            return [c], [c.result()]

        # Both sides fully constant.
        if lhs_non_const is None and rhs_non_const is None:
            c = IndexConstantOp.build(const_prod)
            return [c], [c.result()]

        # Build product from non-constant parts.
        new_ops: list[Op] = []
        if lhs_non_const is None:
            result = rhs_non_const
        elif rhs_non_const is None:
            result = lhs_non_const
        else:
            mul = IndexMulOp.build(lhs_non_const, rhs_non_const)
            new_ops.append(mul)
            result = mul.result()

        # Append combined constant on the rhs.
        if const_prod != 1:
            c = IndexConstantOp.build(const_prod)
            mul = IndexMulOp.build(result, c.result())
            new_ops.extend([c, mul])
            result = mul.result()

        return new_ops, [result]


    def lower_div(
        self, op: IndexDivOp,
    ) -> tuple[list[Op], list[object]] | None:
        lhs = self.value_map[op.lhs]
        lhs_val = self._const_val(op.lhs)
        rhs_val = self._const_val(op.rhs)

        if lhs_val is not None and rhs_val is not None and rhs_val != 0:
            new_op = IndexConstantOp.build(lhs_val // rhs_val)
            return [new_op], [new_op.result()]
        if lhs_val == 0:
            new_op = IndexConstantOp.build(0)
            return [new_op], [new_op.result()]
        if rhs_val == 1:
            return [], [lhs]
        return None

    def lower_mod(
        self, op: IndexModOp,
    ) -> tuple[list[Op], list[object]] | None:
        lhs_val = self._const_val(op.lhs)
        rhs_val = self._const_val(op.rhs)

        if lhs_val is not None and rhs_val is not None and rhs_val != 0:
            new_op = IndexConstantOp.build(lhs_val % rhs_val)
            return [new_op], [new_op.result()]
        if lhs_val == 0:
            new_op = IndexConstantOp.build(0)
            return [new_op], [new_op.result()]
        return None

    def lower_ptr_add(
        self, op: PtrAddOp,
    ) -> tuple[list[Op], list[object]] | None:
        ptr = self.value_map[op.ptr]
        offset_val = self._const_val(op.byte_offset)
        if offset_val == 0:
            return [], [ptr]
        return None


    def lower_tile(
        self, op: TileOp,
    ) -> tuple[list[Op], list[object]] | None:
        arg = self.value_map[op.arg]
        if list(op.result().type.sizes) == list(arg.type.sizes):
            return [], [arg]
        return None


def folding(module: ModuleOp) -> ModuleOp:
    """Convenience entry point."""
    return Folding().run(module)
