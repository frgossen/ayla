"""Folding pass: constant folding and local simplifications.

Folds index arithmetic where operands are known constants,
eliminates trivial operations like `x * 1`, `x + 0`, `0 * n`,
`x * 0`, removes no-op tile ops, resolves a `select` whose index is known,
inlines a `for` that runs once, and erases one that never runs.

One fact beyond plain constants: inside a ``for``, the induction variable is
congruent to ``lb`` modulo ``step``. That is enough to fold ``(iv + c) % m``
whenever ``m`` divides the step, even though ``iv`` itself is not a constant --
which is how a per-iteration buffer choice becomes a static one after
unrolling multiplies the step.

A second fact bounds hardware IDs by the enclosing kernel's launch
dimensions, and loop variables by their loop. When x is known to stay below m,
x % m folds to x and x / m folds to 0. Tiling passes produce this pattern when
they split one hardware ID into several tile indices and the hardware axis
already has exactly as many IDs as there are tiles. This relies on launch
dimensions never changing afterwards, which the kernel launch-dimension
setters enforce by rejecting changes to known axes.
"""
from __future__ import annotations

from typing import cast

from ..gpu_dialect import (
    HW_ID_LEVELS,
    BlockIdOp,
    CollectiveAxis,
    ForOp,
    IndexAddOp,
    IndexConstantOp,
    IndexDivOp,
    IndexModOp,
    IndexMulOp,
    KernelOp,
    MemTensorType,
    ModuleOp,
    PtrAddOp,
    SelectOp,
    ThreadIdOp,
    TileOp,
    WarpGroupIdOp,
    launch_extent,
)
from ..base.ir import (
    Block,
    Op,
    Value,
    clone_op,
    clone_type,
    find_ancestor_of_type,
)
from ..base.lowering_pass import LoweringPass, lowers


def _iv_loop(value: Value) -> ForOp | None:
    """The loop ``value`` is the induction variable of, if it is one.

    An induction variable is a block argument, so it has no defining op -- the
    loop it belongs to is the one holding that block, which the parent links
    reach directly. Returning the loop rather than its bounds keeps `ub`,
    `trip_count` and the rest available to callers that need them.
    """
    block = value.definition
    if not isinstance(block, Block):
        return None
    loop = block.parent_op
    if not isinstance(loop, ForOp) or value is not loop.iv:
        return None
    return loop


class Folding(LoweringPass):
    @lowers(ForOp)
    def lower_for(
        self, op: ForOp,
    ) -> tuple[list[Op], list[object]] | None:
        # `(ub - lb) // step` floors, so an empty range reads as <= 0 for either
        # sign of step. The body never runs, so nothing it defines can be
        # referenced from outside and nothing it does is observable.
        if op.trip_count <= 0:
            return [], []
        if op.trip_count != 1:
            return None
        self.lower_block(op.body)
        iv_const = IndexConstantOp.build(value=op.lb)
        vmap: dict[Value, Value] = {op.iv: iv_const.result()}
        cloned = [clone_op(body_op, vmap) for body_op in op.body.ops]
        return [iv_const, *cloned], []

    @lowers(SelectOp)
    def lower_select(
        self, op: SelectOp,
    ) -> tuple[list[Op], list[object]] | None:
        index = self._const_val(op.index)
        if index is None:
            return None
        cases = [self.value_map.get(case, case) for case in op.cases]
        if not 0 <= index < len(cases):
            raise ValueError(
                f"select: index {index} out of range for {len(cases)} cases"
            )
        return [], [cases[index]]

    def _const_val(self, value: Value) -> int | None:
        mapped = self.value_map.get(value, value)
        assert isinstance(mapped, Value)
        if isinstance(mapped.definition, IndexConstantOp):
            return mapped.definition.value
        return None

    def _range(self, value: Value) -> tuple[int, int] | None:
        """Inclusive bounds of a non-negative index, if they are known.

        Hardware IDs range over the enclosing kernel's launch, an induction
        variable over its loop, and arithmetic over its operands' ranges.
        """
        # Folding only ever maps values to values.
        value = cast(Value, self.value_map.get(value, value))
        definition = value.definition
        if isinstance(definition, IndexConstantOp):
            c = definition.value
            return (c, c) if c >= 0 else None
        if (loop := _iv_loop(value)) is not None:
            return (loop.lb, loop.ub - 1) if loop.lb >= 0 else None
        if isinstance(definition, (BlockIdOp, ThreadIdOp, WarpGroupIdOp)):
            axis = (
                definition.axis
                if isinstance(definition, (BlockIdOp, ThreadIdOp))
                else CollectiveAxis(definition.results.index(value))
            )
            kernel = find_ancestor_of_type(definition, KernelOp)
            assert kernel is not None
            level = HW_ID_LEVELS[type(definition)]
            extent = launch_extent(kernel, level, axis)
            return None if extent is None else (0, extent - 1)
        if not isinstance(
            definition, (IndexAddOp, IndexMulOp, IndexDivOp, IndexModOp),
        ):
            return None
        rhs = self._range(definition.rhs)
        if rhs is None:
            return None
        rhs_min, rhs_max = rhs
        lhs = self._range(definition.lhs)
        # A remainder is bounded even when its numerator is unknown.
        if isinstance(definition, IndexModOp):
            if rhs_min == 0:
                return None
            upper = rhs_max - 1
            if lhs is not None:
                _, lhs_max = lhs
                upper = min(lhs_max, upper)
            return (0, upper)
        if lhs is None:
            return None
        lhs_min, lhs_max = lhs
        match definition:
            case IndexAddOp():
                return (lhs_min + rhs_min, lhs_max + rhs_max)
            case IndexMulOp():
                return (lhs_min * rhs_min, lhs_max * rhs_max)
            case IndexDivOp():
                if rhs_min == 0:
                    return None
                return (lhs_min // rhs_max, lhs_max // rhs_min)
        return None

    def _decompose_add(self, value: Value) -> tuple[Value | None, int]:
        """Strip the constant part from a value.

        If value is a constant, returns (None, const).
        If value is an add with a constant rhs, returns (lhs, const).
        Otherwise returns (value, 0).
        """
        mapped = self.value_map.get(value, value)
        assert isinstance(mapped, Value)
        const = self._const_val(value)
        if const is not None:
            return None, const
        if isinstance(mapped.definition, IndexAddOp):
            rhs_const = self._const_val(mapped.definition.rhs)
            if rhs_const is not None:
                lhs = self.value_map.get(
                    mapped.definition.lhs, mapped.definition.lhs,
                )
                assert isinstance(lhs, Value)
                return lhs, rhs_const
        return mapped, 0

    @lowers(IndexAddOp)
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
            assert result is not None
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
        assert isinstance(mapped, Value)
        const = self._const_val(value)
        if const is not None:
            return None, const
        if isinstance(mapped.definition, IndexMulOp):
            rhs_const = self._const_val(mapped.definition.rhs)
            if rhs_const is not None:
                lhs = self.value_map.get(
                    mapped.definition.lhs, mapped.definition.lhs,
                )
                assert isinstance(lhs, Value)
                return lhs, rhs_const
        return mapped, 1

    @lowers(IndexMulOp)
    def lower_mul(
        self, op: IndexMulOp,
    ) -> tuple[list[Op], list[object]] | None:
        # Distribute a constant factor over a normalized ``x + c``.  Besides
        # folding ``c * factor`` at compile time, this exposes the shared
        # ``x * factor`` term to CSE across sibling affine coordinates.
        lhs = self.value_map.get(op.lhs, op.lhs)
        rhs = self.value_map.get(op.rhs, op.rhs)
        assert isinstance(lhs, Value)
        assert isinstance(rhs, Value)
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
                add_lhs = self.value_map.get(add.lhs, add.lhs)
                assert isinstance(add_lhs, Value)
                product = IndexMulOp.build(add_lhs, factor_op.result())
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
            assert result is not None
            c = IndexConstantOp.build(const_prod)
            mul = IndexMulOp.build(result, c.result())
            new_ops.extend([c, mul])
            result = mul.result()

        return new_ops, [result]


    @lowers(IndexDivOp)
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
        if rhs_val is None:
            return None
        bounds = self._range(op.lhs)
        if bounds is None:
            return None
        _, upper = bounds
        if upper < rhs_val:
            new_op = IndexConstantOp.build(0)
            return [new_op], [new_op.result()]
        return None

    @lowers(IndexModOp)
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

        # An induction variable only takes values lb + i*step, so (iv + c) % m
        # is (lb + c) % m for any m dividing the step.
        base, offset = self._decompose_add(op.lhs)
        loop = _iv_loop(base) if rhs_val and base is not None else None
        if loop is not None and rhs_val is not None and loop.step % rhs_val == 0:
            new_op = IndexConstantOp.build((loop.lb + offset) % rhs_val)
            return [new_op], [new_op.result()]

        # A value already below the modulus is its own remainder.
        if rhs_val is None:
            return None
        bounds = self._range(op.lhs)
        if bounds is None:
            return None
        _, upper = bounds
        if upper < rhs_val:
            return [], [self.value_map[op.lhs]]
        return None

    @lowers(PtrAddOp)
    def lower_ptr_add(
        self, op: PtrAddOp,
    ) -> tuple[list[Op], list[object]] | None:
        ptr = self.value_map[op.ptr]
        assert isinstance(ptr, Value)
        offset_val = self._const_val(op.byte_offset)
        if offset_val == 0:
            return [], [ptr]
        return None


    @lowers(TileOp)
    def lower_tile(
        self, op: TileOp,
    ) -> tuple[list[Op], list[object]] | None:
        arg = self.value_map[op.arg]
        assert isinstance(arg, Value)
        result_type = op.result().type
        arg_type = arg.type
        assert isinstance(result_type, MemTensorType)
        assert isinstance(arg_type, MemTensorType)
        if list(result_type.sizes) == list(arg_type.sizes):
            return [], [arg]
        indices: list[int | Value] = []
        changed = False
        for index in op.indices:
            if isinstance(index, int):
                indices.append(index)
                continue
            mapped = self.value_map.get(index, index)
            assert isinstance(mapped, Value)
            constant = self._const_val(index)
            if constant is None:
                indices.append(mapped)
            else:
                indices.append(constant)
                changed = True
        if changed:
            attrs = {
                key: value for key, value in op.attrs.items()
                if key not in ("static_indices", "maybe_partial")
            }
            tile = TileOp.build(
                arg, indices, clone_type(op.result().type), attrs=attrs,
                maybe_partial=op.maybe_partial,
            )
            return [tile], [tile.result()]
        return None


def folding(module: ModuleOp) -> ModuleOp:
    """Convenience entry point."""
    return Folding().run(module)
