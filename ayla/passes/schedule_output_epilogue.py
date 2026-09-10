"""Overlap a tiled output tail with the next asynchronous matmul."""

from __future__ import annotations

from ..base.ir import Block, Op, Value
from ..base.pipeline import Pass, op_label_matches
from ..gpu_dialect import (
    AsyncOp,
    AwaitOp,
    MatmulOp,
    ModuleOp,
    TileOp,
    effective_operands,
    is_elementwise,
    root_buffer,
)


def _last_wgmma_window(ops: list[Op]) -> tuple[AsyncOp, AwaitOp] | None:
    """The last complete WGMMA issue/await pair in ``ops``."""
    for wait_index in range(len(ops) - 1, -1, -1):
        wait = ops[wait_index]
        if not isinstance(wait, AwaitOp) or wait.lock.type.kind != "wgmma":
            continue
        issue = next((
            op for op in reversed(ops[:wait_index])
            if isinstance(op, AsyncOp) and op.lock is wait.lock
        ), None)
        if issue is not None:
            return issue, wait
    return None


def _trailing_tile_dim(value: Value) -> int | None:
    """The one dimension in which a static tile ends at its source bound."""
    tile = value.definition
    if not isinstance(tile, TileOp):
        return None
    dims = []
    for dim in range(value.type.rank):
        index = tile.static_indices[dim]
        source_size = tile.arg.type.sizes[dim]
        tile_size = value.type.sizes[dim]
        if (
            isinstance(index, int)
            and isinstance(source_size, int)
            and isinstance(tile_size, int)
            and tile_size < source_size
            and (index + 1) * tile_size == source_size
        ):
            dims.append(dim)
    return dims[0] if len(dims) == 1 else None


def _matches_tail(values: set[Value], value: Value) -> bool:
    """Whether a value is tracked directly or aliases its trailing slice."""
    if value in values:
        return True
    dim = _trailing_tile_dim(value)
    if dim is None:
        return False
    root = root_buffer(value)
    return any(
        _trailing_tile_dim(candidate) == dim
        and candidate.type.rank == value.type.rank
        and root_buffer(candidate) is root
        and candidate.type.sizes[dim] == value.type.sizes[dim]
        for candidate in values
    )


class ScheduleOutputEpilogue(Pass):
    """Move selected output tails into the next WGMMA wait window.

    The selected element-wise op must consume one proper trailing static tile.
    Its exact dataflow, including an aliased block-wide view used by a TMA
    store, is followed through the store await. Every tail except the final one
    moves immediately before the last WGMMA wait before the next tail.

    This pass deliberately supports one trailing tile. It does not choose tile
    sizes, split epilogues, or model a general stage schedule.
    """

    def __init__(self, label_filter: str) -> None:
        self.label_filter = label_filter

    @staticmethod
    def _group(ops: list[Op], seed: Op) -> list[Op]:
        """Follow a selected tail through its output-store await."""
        values = set(effective_operands(seed))
        group = [seed]
        for op in ops[ops.index(seed) + 1:]:
            operands = effective_operands(op)
            if not any(_matches_tail(values, value) for value in operands):
                continue
            group.append(op)
            values.update(operands)
            values.update(op.results)
            if isinstance(op, AwaitOp):
                return group
        raise ValueError(
            "ScheduleOutputEpilogue: selected tail has no output-store await"
        )

    def run_block(self, block: Block) -> None:
        super().run_block(block)
        ops = block.ops
        tails = [
            op for op in ops
            if is_elementwise(op)
            and op_label_matches(op, self.label_filter)
            and any(
                _trailing_tile_dim(value) is not None for value in op.operands
            )
        ]
        if len(tails) < 2:
            return

        groups = [self._group(ops, tail) for tail in tails]
        if any(set(a) & set(b) for a, b in zip(groups, groups[1:])):
            raise ValueError("ScheduleOutputEpilogue: overlapping tail groups")

        for group, next_group in zip(groups, groups[1:]):
            group_set = set(group)
            end = max(ops.index(op) for op in group)
            next_start = ops.index(next_group[0])
            window = _last_wgmma_window(ops[end + 1:next_start])
            if window is None:
                raise ValueError(
                    "ScheduleOutputEpilogue: no next WGMMA wait between tails"
                )
            wgmma, insertion = window
            tail_roots = {
                root_buffer(value)
                for value in effective_operands(group[0])
                if _trailing_tile_dim(value) is not None
            }
            wgmma_roots = {
                root_buffer(inner.dst)
                for inner in wgmma.body.ops
                if isinstance(inner, MatmulOp)
            }
            if tail_roots & wgmma_roots:
                raise ValueError(
                    "ScheduleOutputEpilogue: tail and next WGMMA share an "
                    "accumulator"
                )
            moved = [op for op in ops if op in group_set]
            remaining = [op for op in ops if op not in group_set]
            insert_at = remaining.index(insertion)
            remaining[insert_at:insert_at] = moved
            ops[:] = remaining


def schedule_output_epilogue(
    module: ModuleOp, *, label_filter: str,
) -> ModuleOp:
    return ScheduleOutputEpilogue(label_filter).run(module)
