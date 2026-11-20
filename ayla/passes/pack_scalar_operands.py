"""Group private scalar FMA operands without speculating memory effects."""

from ..base.ir import Block, clone_type
from ..base.pipeline import Pass
from ..gpu_dialect import (
    AllocOp, CollectiveLevel, F32Type, ForOp, IndexAddOp, IndexConstantOp, IndexDivOp,
    MatmulOp, MoveOp, RmemTensorType, SmemTensorType, TileOp,
    effective_operands, is_side_effect_free,
)


def _affine(value):
    if isinstance(value, int):
        return None, value
    op = value.definition
    if isinstance(op, IndexConstantOp):
        return None, op.value
    if isinstance(op, IndexAddOp):
        left, right = _affine(op.lhs), _affine(op.rhs)
        if left[0] is None:
            return right[0], left[1] + right[1]
        if right[0] is None:
            return left[0], left[1] + right[1]
    return value, 0


def _aligned_range(root, offset, columns):
    if type(columns) is not int or columns < 4:
        return False
    if root is None:
        return offset % 4 == 0 and 0 <= offset and offset + 4 <= min(columns, 1 << 32)
    body = root.definition
    loop = body.parent_op if isinstance(body, Block) else None
    if not isinstance(loop, ForOp) or root is not loop.iv or loop.trip_count <= 0:
        return False
    last = loop.lb + (loop.trip_count - 1) * loop.step
    return (loop.step > 0 and loop.step % 4 == 0 and (loop.lb + offset) % 4 == 0
            and 0 <= loop.lb + offset and last + offset + 4 <= min(columns, 1 << 32))


def _candidate(ops, mm):
    if (not isinstance(mm, MatmulOp) or mm.init is not None or not mm.accumulate
            or any(level != CollectiveLevel.THREAD for level in mm.collective.values())
            or mm.attrs.get("participant_warps")
            or any(not isinstance(v.type, RmemTensorType)
                   or not isinstance(v.type.dtype, F32Type) or v.type.rank != 2
                   for v in (mm.lhs, mm.rhs, mm.dst))
            or mm.lhs.type.sizes[-1] != 1 or mm.rhs.type.sizes[-1] != 1):
        return None
    loads = []
    for arg in (mm.lhs, mm.rhs):
        if not isinstance(arg.definition, AllocOp):
            return None
        load = next((op for op in reversed(ops[:ops.index(mm)])
                     if isinstance(op, MoveOp) and op.dst is arg), None)
        if (load is None or not isinstance(load.src.type, SmemTensorType)
                or load.src.type.rank != 2 or load.src.type.strides[-1] != 1
                or load.src.type.dtype != arg.type.dtype or load.indices or load.multicast
                or load.attrs.get("participant_warps")
                or any(level != CollectiveLevel.THREAD for level in load.collective.values())
                or not isinstance(load.src.definition, TileOp)
                or load.src.definition.maybe_partial
                or load.src.definition.result().type.sizes != arg.type.sizes):
            return None
        loads.append(load)
    return loads


def pack_scalar_operand_block(block, *, vectorize=True):
    """Pack four K1 FMAs only when their private register history is closed.

    Required loads already share one straight-line completion scope. The
    frame may contain only those loads/FMAs and independent pure definitions.
    Fresh operands replace the old register WAR chain; the accumulator keeps
    each output's original K order. No barrier, guard, issue or other memory
    effect is crossed. Tail frames and unproved affine groups remain intact.
    """
    packed = 0
    while True:
        ops = list(block.ops)
        mms = [op for op in ops if _candidate(ops, op) is not None]
        found = False
        for first in range(len(mms) - 3):
            group = mms[first:first + 4]
            loads = [_candidate(ops, mm) for mm in group]
            if any(mm.dst is not group[0].dst or mm.collective != group[0].collective
                   or mm.extra_attrs != group[0].extra_attrs for mm in group):
                continue
            registers = {arg for mm in group for arg in (mm.lhs, mm.rhs)}
            if group[0].dst in registers or any(mm.lhs is mm.rhs for mm in group):
                continue
            members = set(group) | {load for pair in loads for load in pair}
            start, end = min(ops.index(op) for op in members), ops.index(group[-1])
            if any(op not in members and (not is_side_effect_free(op)
                    or set(effective_operands(op)) & (registers | {group[0].dst}))
                   for op in ops[start:end + 1]):
                continue
            # An outside consumer must see a later, retained overwrite, not
            # one of the removed loads. Reject escaping and aliased views.
            closed = True
            for reg in registers:
                for user in reg.users():
                    if user in members:
                        continue
                    if user.parent_block is not block:
                        closed = False
                        break
                    if isinstance(user, MoveOp) and user.dst is reg:
                        if user.src is reg:
                            closed = False
                            break
                        continue
                    if not isinstance(user, MatmulOp):
                        closed = False
                        break
                    latest = next((op for op in reversed(ops[:ops.index(user)])
                                   if isinstance(op, MoveOp) and op.dst is reg), None)
                    if latest in members or latest is None:
                        closed = False
                        break
            if not closed:
                continue
            views = []
            for operand in (0, 1):
                tiles = [pair[operand].src.definition for pair in loads]
                first_tile = tiles[0]
                parent = first_tile.arg
                affine = [_affine(tile.indices[-1]) for tile in tiles]
                root, offset = affine[0]
                if (any(tile.arg is not parent or tile.indices[0] != first_tile.indices[0]
                        or tile.result().type != first_tile.result().type
                        or tile.extra_attrs != first_tile.extra_attrs
                        or pair[operand].extra_attrs != loads[0][operand].extra_attrs
                        or pair[operand].collective != loads[0][operand].collective
                        for tile, pair in zip(tiles, loads))
                        or affine != [(root, offset + i) for i in range(4)]
                        or not _aligned_range(root, offset, parent.type.sizes[-1])):
                    break
                views.append(first_tile)
            if len(views) != 2:
                continue
            additions, operands = [], []
            for tile, original in zip(views, loads[0]):
                column = tile.indices[-1]
                if isinstance(column, int):
                    index = column // 4
                else:
                    four = IndexConstantOp.build(4)
                    div = IndexDivOp.build(column, four.result())
                    additions.extend([four, div])
                    index = div.result()
                typ = clone_type(tile.result().type)
                typ.sizes = [typ.sizes[0], 4]
                view = TileOp.build(tile.arg, [tile.indices[0], index], typ, attrs=tile.extra_attrs)
                reg = AllocOp.build_with_fresh_id(RmemTensorType.build(typ.sizes, typ.dtype))
                attrs = dict(original.extra_attrs)
                attrs.setdefault("vectorize_scalar_shared_loads", vectorize)
                load = MoveOp.build(reg.result(), view.result(), collective=original.collective, attrs=attrs)
                additions.extend([view, reg, load])
                operands.append(reg.result())
            additions.append(MatmulOp.build(*operands, group[0].dst,
                                            collective=group[0].collective,
                                            attrs=group[0].extra_attrs))
            replacement = []
            for op in ops:
                if op is group[0]:
                    replacement.extend(additions)
                if op not in members:
                    replacement.append(op)
            block.ops = replacement
            packed += 1
            found = True
            break
        if not found:
            return packed


class PackScalarOperands(Pass):
    def __init__(self, *, vectorize=True):
        if type(vectorize) is not bool:
            raise ValueError("scalar operand vectorization must be a boolean")
        super().__init__(vectorize=vectorize)

    def run_block(self, block):
        pack_scalar_operand_block(block, vectorize=self.vectorize)
        super().run_block(block)
