"""Folding pass: constant folding and local simplifications.

Folds index arithmetic where operands are known constants,
eliminates trivial operations like `x * 1`, `x + 0`, `0 * n`,
`x * 0`, and removes no-op tile ops.
"""
from __future__ import annotations

from ..gpu_dialect import (
    AllocOp,
    ForOp,
    IndexAddOp,
    IndexConstantOp,
    IndexDivOp,
    IndexModOp,
    IndexMulOp,
    MatmulOp,
    ModuleOp,
    PtrAddOp,
    TileOp,
    WrmemTensorType,
)
from ..base.ir import Block, Op, Value, clone_op, iter_ops
from ..base.lowering_pass import LoweringPass


def _first_matmul(ops: list[Op], dst: Value) -> MatmulOp | None:
    for op in ops:
        if isinstance(op, MatmulOp) and op.dst is dst:
            return op
        if isinstance(op, ForOp):
            continue
        for region in op.regions:
            for block in region.blocks:
                if found := _first_matmul(block.ops, dst):
                    return found
    return None


def _fold_accumulator_init(block: Block) -> None:
    """Replace a zero-filled WGMMA accumulator with first-iteration scale-d."""
    for index, alloc in enumerate(block.ops):
        if not (isinstance(alloc, AllocOp)
                and isinstance(alloc.result().type, WrmemTensorType)
                and alloc.init == 0):
            continue
        accumulator = alloc.result()
        users = {id(op) for op in accumulator.users()}
        first_user = next((
            op for op in block.ops[index + 1:]
            if any(id(nested) in users for nested in iter_ops(op))
        ), None)
        if not (isinstance(first_user, ForOp)
                and first_user.lb == 0 and first_user.trip_count > 0):
            continue
        matmul = _first_matmul(first_user.body.entry_block.ops, accumulator)
        if matmul is not None and matmul.scale_d is None:
            matmul.operands.append(first_user.iv)
            alloc.attrs.pop("init")


class Folding(LoweringPass):

    def __init__(self, *, fold_accumulator_init: bool = False) -> None:
        super().__init__()
        self.fold_accumulator_init = fold_accumulator_init
        self.register(ForOp, type(self).lower_for)
        self.register(IndexAddOp, type(self).lower_add)
        self.register(IndexMulOp, type(self).lower_mul)
        self.register(IndexDivOp, type(self).lower_div)
        self.register(IndexModOp, type(self).lower_mod)
        self.register(PtrAddOp, type(self).lower_ptr_add)
        self.register(TileOp, type(self).lower_tile)

    def lower_block(self, block: Block) -> None:
        if self.fold_accumulator_init:
            _fold_accumulator_init(block)
        super().lower_block(block)

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
        # Distribute a constant factor over a normalized ``x + c``.  Besides
        # folding ``c * factor`` at compile time, this exposes the shared
        # ``x * factor`` term to CSE across sibling affine coordinates.
        lhs = self.value_map.get(op.lhs, op.lhs)
        rhs = self.value_map.get(op.rhs, op.rhs)
        lhs_factor = self._const_val(op.lhs)
        rhs_factor = self._const_val(op.rhs)
        add_value, factor = (
            (rhs, lhs_factor) if lhs_factor is not None
            else (lhs, rhs_factor)
        )
        if factor is not None and isinstance(add_value.definition, IndexAddOp):
            add = add_value.definition
            add_const = self._const_val(add.rhs)
            if add_const is not None:
                factor_op = IndexConstantOp.build(factor)
                product = IndexMulOp.build(
                    self.value_map.get(add.lhs, add.lhs), factor_op.result(),
                )
                offset = IndexConstantOp.build(add_const * factor)
                result = IndexAddOp.build(product.result(), offset.result())
                return [factor_op, product, offset, result], [result.result()]

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
