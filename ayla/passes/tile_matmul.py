from __future__ import annotations

from ..gpu_dialect import (
    BlockIdsOp,
    CollectiveAxis,
    CollectiveLevel,
    ForOp,
    IfOp,
    IndexConstantOp,
    IndexLtOp,
    IndexType,
    KernelOp,
    MatmulOp,
    ModuleOp,
    ThreadIdsOp,
    TileOp,
    WarpGroupIdOp,
    is_elementwise,
)
from ..base.ir import Block, Op, Value, clone_type, find_ops
from ..base.lowering_pass import LoweringPass, lowers
from ._utils import label_matches


_PARALLEL_ID_OPS = {
    CollectiveLevel.BLOCK: BlockIdsOp,
    CollectiveLevel.THREAD: ThreadIdsOp,
    CollectiveLevel.WG: WarpGroupIdOp,
}


class TileMatmul(LoweringPass):
    """Tile every ``matmul`` along ``op_dims`` by matching ``tile_sizes``.

    Four parallel lists describe the tiling:
      - ``op_dims[i]``: one of ``"M" | "N" | "K"`` (matmul shapes
        are lhs=[M, K], rhs=[N, K], dst=[M, N]).
      - ``tile_sizes[i]``: tile size for that dim.
      - ``iteration_bindings[i]``: where the iteration lives — ``None`` for a
        sequential ``for`` loop, or a ``(CollectiveLevel, CollectiveAxis)``
        pair for a hardware dimension. An existing parallel launch extent may
        be larger than the logical tile count; excess workers are guarded. A
        smaller extent remains a launch conflict; express multiple waves with
        an explicit outer ``for`` tiling level.
      - ``for_loop_labels[i]``: the label for a generated ``for`` loop, or
        ``None`` for an iteration mapped to a hardware dimension.

    Nesting order is outer→inner along the list.

    The matmul is the hero: the dims name *its* axes, and everything else the
    tiling touches is found from it. ``epilogue_label_filter`` opts
    element-wise ops that consume its destination into the same tiling, so
    they are cut to the same tile rather than left reading the whole tensor.
    Left ``None``, no op is drawn in and only the matmul is tiled.

    An epilogue joins every loop whose dim it shares and stays outside every
    loop it does not: it has M and N but no K, so tiling K puts the matmul in
    a loop and leaves the epilogue after it, running once per tile rather than
    once per K step.

    ``allow_partial_tiles`` permits a non-divisible block-level tile by
    rounding the logical tile count up. Addressing remains full-tile; callers
    must use memory operations with defined out-of-bounds behavior, such as
    TMA.
    """

    def __init__(
        self,
        op_dims: list[str],
        tile_sizes: list[int],
        iteration_bindings: list[
            tuple[CollectiveLevel, CollectiveAxis] | None
        ],
        for_loop_labels: list[str | None],
        op_label_filter: str | None = None,
        epilogue_label_filter: str | None = None,
        allow_partial_tiles: bool = False,
    ) -> None:
        super().__init__(
            op_dims=list(op_dims),
            tile_sizes=list(tile_sizes),
            iteration_bindings=list(iteration_bindings),
            for_loop_labels=list(for_loop_labels),
            op_label_filter=op_label_filter,
            epilogue_label_filter=epilogue_label_filter,
            allow_partial_tiles=allow_partial_tiles,
        )
        assert (
            len(op_dims)
            == len(tile_sizes)
            == len(iteration_bindings)
            == len(for_loop_labels)
        ), (
            "TileMatmul: op_dims, tile_sizes, iteration_bindings, "
            "for_loop_labels "
            f"must have the same length; got {len(op_dims)}, "
            f"{len(tile_sizes)}, {len(iteration_bindings)}, "
            f"{len(for_loop_labels)}"
        )
        assert all(
            binding is None or label is None
            for binding, label in zip(iteration_bindings, for_loop_labels)
        ), "TileMatmul: only sequential iterations may have loop labels"
        parallel_bindings = [
            binding for binding in iteration_bindings if binding is not None
        ]
        # TODO: Linearize multiple matmul tile dimensions onto one hardware
        # axis (for example, map the M x N tile space onto block.x).
        if len(parallel_bindings) != len(set(parallel_bindings)):
            raise ValueError(
                "TileMatmul: one hardware dimension cannot tile multiple "
                "matmul dimensions"
            )
        for binding in parallel_bindings:
            level, axis = binding
            if (
                level not in _PARALLEL_ID_OPS
                or level == CollectiveLevel.WG and axis != CollectiveAxis.X
            ):
                raise ValueError(
                    f"TileMatmul: unsupported iteration binding {binding!r}"
                )
        self._kernel: KernelOp

    @lowers(KernelOp)
    def lower_kernel(
        self, op: KernelOp,
    ) -> tuple[list[Op], list[object]]:
        self._kernel = op
        heroes = find_ops(op, self._is_hero)
        bounds = {hero: self._iteration_bounds(hero) for hero in heroes}
        self._resolve_launch_extents(bounds)
        self._rewrite_block(op.body, bounds)
        return [op], []

    def _rewrite_block(
        self,
        block: Block,
        bounds: dict[MatmulOp, list[int]],
    ) -> None:
        """Replace each hero and the epilogue behind it with a tiled region.

        Nested blocks first, so a matmul already inside a loop from an earlier
        tiling is rewritten before the loop holding it is rebuilt.
        """
        for op in block.ops:
            for nested in op.blocks:
                self._rewrite_block(nested, bounds)

        ops = list(block.ops)
        new_ops: list[Op] = []
        i = 0
        while i < len(ops):
            op = ops[i]
            if not self._is_hero(op):
                new_ops.append(op)
                i += 1
                continue
            epilogue = self._epilogue(ops, i)
            new_ops.extend(self._tile(op, epilogue, bounds[op]))
            i += 1 + len(epilogue)
        block.ops = new_ops

    def _is_hero(self, op: Op) -> bool:
        return isinstance(op, MatmulOp) and label_matches(
            op, self.op_label_filter,
        )

    def _epilogue(self, ops: list[Op], hero: int) -> list[Op]:
        """The ops directly behind ``ops[hero]`` that its tiling should absorb.

        Contiguous only, and each has to earn its place: element-wise, opted
        in by ``epilogue_label_filter``, and reading the hero's destination.
        Anything else ends the run -- an op left where it is must not be
        stepped over, or absorbing what follows would reorder the two.
        """
        found: list[Op] = []
        dst = ops[hero].dst
        i = hero + 1
        while i < len(ops):
            op = ops[i]
            if self.epilogue_label_filter is None or not is_elementwise(op):
                break
            if not label_matches(op, self.epilogue_label_filter):
                break
            # In place over the hero's destination is the only shape supported:
            # one operand pattern, one set of dims, no broadcast to reason about.
            if len(op.operands) != 2 or not all(v is dst for v in op.operands):
                break
            found.append(op)
            i += 1
        return found

    def _iteration_bounds(self, hero: MatmulOp) -> list[int]:
        lhs, rhs, _ = hero.operands
        sizes_by_dim = {
            "M": lhs.type.sizes[0],
            "N": rhs.type.sizes[0],
            "K": lhs.type.sizes[1],
        }
        ubs: list[int] = []
        for op_dim, tile_size, iteration_binding in zip(
            self.op_dims, self.tile_sizes, self.iteration_bindings,
        ):
            if op_dim not in sizes_by_dim:
                raise ValueError(
                    f"TileMatmul: op_dim must be 'M', 'N', or 'K'; "
                    f"got {op_dim!r}"
                )
            op_dim_size = sizes_by_dim[op_dim]
            is_partial = op_dim_size % tile_size != 0
            if is_partial and not (
                self.allow_partial_tiles
                and (
                    iteration_binding is None
                    or iteration_binding[0] == CollectiveLevel.BLOCK
                )
                ):
                raise ValueError(
                    f"TileMatmul: op_dim {op_dim} size "
                    f"({op_dim_size}) is not divisible by tile "
                    f"size ({tile_size})"
                )
            ub = (op_dim_size + tile_size - 1) // tile_size
            ubs.append(ub)
            sizes_by_dim[op_dim] = tile_size
        return ubs

    def _resolve_launch_extents(
        self, bounds: dict[MatmulOp, list[int]],
    ) -> None:
        required: dict[tuple[CollectiveLevel, CollectiveAxis], int] = {}
        for ubs in bounds.values():
            for binding, ub in zip(self.iteration_bindings, ubs):
                if binding is not None:
                    required[binding] = max(required.get(binding, 0), ub)

        for binding, extent in required.items():
            current = self._launch_extent(binding)
            if current is not None and current > extent:
                continue
            self._accumulate_constraint(binding, extent)

    def _launch_extent(
        self, binding: tuple[CollectiveLevel, CollectiveAxis],
    ) -> int | None:
        level, axis = binding
        if level == CollectiveLevel.BLOCK:
            return self._kernel.grid_dim[axis]
        if level == CollectiveLevel.THREAD:
            return self._kernel.block_dim[axis]
        assert level == CollectiveLevel.WG and axis == CollectiveAxis.X
        return self._kernel.num_warp_groups

    def _tile(
        self,
        hero: MatmulOp,
        epilogue: list[Op],
        ubs: list[int],
    ) -> list[Op]:
        lhs, rhs, dst = hero.operands

        new_ops: list[Op] = []
        parallel_ids: dict[
            CollectiveLevel, BlockIdsOp | ThreadIdsOp | WarpGroupIdOp
        ] = {}
        for level, op_type in (
            (CollectiveLevel.BLOCK, BlockIdsOp),
            (CollectiveLevel.THREAD, ThreadIdsOp),
            (CollectiveLevel.WG, WarpGroupIdOp),
        ):
            if any(
                binding is not None and binding[0] == level
                for binding in self.iteration_bindings
            ):
                ids = op_type.build()
                parallel_ids[level] = ids
                new_ops.append(ids)

        ivs: list[Value] = []
        for iteration_binding, loop_label in zip(
            self.iteration_bindings, self.for_loop_labels,
        ):
            if iteration_binding is None:
                ivs.append(Value(name=loop_label, type=IndexType()))
            else:
                level, axis = iteration_binding
                ivs.append(parallel_ids[level].results[axis])

        # The outermost loop over a dim the epilogue does not have. It belongs
        # after that one, and so outside every loop nested in it.
        outside_of = min(
            (
                i for i, dim in enumerate(self.op_dims)
                if dim not in ("M", "N")
                and self.iteration_bindings[i] is None
            ),
            default=None,
        )

        body_ops: list[Op] = []
        ops, lhs = _build_tile_op(
            lhs, ["M", "K"],
            op_dims=self.op_dims,
            tile_idxs=ivs,
            tile_sizes=self.tile_sizes,
        )
        body_ops.extend(ops)
        ops, rhs = _build_tile_op(
            rhs, ["N", "K"],
            op_dims=self.op_dims,
            tile_idxs=ivs,
            tile_sizes=self.tile_sizes,
        )
        body_ops.extend(ops)
        dst_ops, dst = _build_tile_op(
            dst, ["M", "N"],
            op_dims=self.op_dims,
            tile_idxs=ivs,
            tile_sizes=self.tile_sizes,
        )
        # The destination is cut by M and N, never by K, so its tile stands
        # outside the loops over K -- the same place the epilogue stands, and
        # where it has to be for an epilogue there to name it. Only that far:
        # M may be a `for` level too, and the tile reads its iv.
        hoist_dst = outside_of is not None
        if not hoist_dst:
            body_ops.extend(dst_ops)
        for iteration_binding in self.iteration_bindings:
            if iteration_binding is not None:
                level, axis = iteration_binding
                hero.narrow_collective_level(
                    axis,
                    level,
                )
        body_ops.append(
            MatmulOp.build(
                lhs=lhs,
                rhs=rhs,
                dst=dst,
                collective=hero.collective,
                attrs=hero.extra_attrs,
                init=hero.init,
            ),
        )

        tiled_epilogue = [
            type(op)(operands=[dst, dst], results=[], attrs=op.attrs)
            for op in epilogue
        ]
        if outside_of is None:
            body_ops.extend(tiled_epilogue)
            tiled_epilogue = []

        for i, iv, ub, binding, label in zip(
            reversed(range(len(self.iteration_bindings))),
            reversed(ivs), reversed(ubs), reversed(self.iteration_bindings),
            reversed(self.for_loop_labels),
        ):
            if binding is not None:
                launch_extent = self._launch_extent(binding)
                if launch_extent is not None and launch_extent > ub:
                    limit = IndexConstantOp.build(ub)
                    condition = IndexLtOp.build(iv, limit.result())
                    body_ops = [
                        limit,
                        condition,
                        IfOp.build(condition.result(), Block(ops=body_ops)),
                    ]
                continue
            body = Block(arguments=[iv], ops=body_ops)
            attrs = {"label": label} if label is not None else None
            body_ops = [ForOp.build(lb=0, ub=ub, body=body, attrs=attrs)]
            if i == outside_of:
                if hoist_dst:
                    body_ops[:0] = dst_ops
                body_ops.extend(tiled_epilogue)
                tiled_epilogue = []

        assert not tiled_epilogue, "TileMatmul: epilogue was never placed"
        new_ops.extend(body_ops)
        return new_ops

    def _accumulate_constraint(
        self,
        iteration_binding: tuple[CollectiveLevel, CollectiveAxis],
        ub: int,
    ) -> None:
        level, axis = iteration_binding
        if level == CollectiveLevel.BLOCK:
            required = [None, None, None]
            required[axis] = ub
            self._kernel.grid_dim = _resolve_launch_dim_constraints(
                list(self._kernel.grid_dim), required,
            )
        elif level == CollectiveLevel.THREAD:
            required = [None, None, None]
            required[axis] = ub
            self._kernel.block_dim = _resolve_launch_dim_constraints(
                list(self._kernel.block_dim), required,
            )
        else:
            assert level == CollectiveLevel.WG and axis == CollectiveAxis.X
            [num_warp_groups] = _resolve_launch_dim_constraints(
                [self._kernel.num_warp_groups], [ub],
            )
            self._kernel.num_warp_groups = num_warp_groups
            self._kernel.block_dim = _resolve_launch_dim_constraints(
                list(self._kernel.block_dim), [ub * 128, 1, 1],
            )


def _resolve_launch_dim_constraints(
    a_list: list[int | None], b_list: list[int | None],
) -> list[int | None]:
    """Symmetric per-axis merge of two ``int | None`` lists.

    For each axis: if either side is ``None``, take the other; if they
    agree, take it; otherwise raise.
    """
    assert len(a_list) == len(b_list)
    result: list[int | None] = []
    for a, b in zip(a_list, b_list):
        if a is None:
            result.append(b)
        elif b is None:
            result.append(a)
        elif a == b:
            result.append(a)
        else:
            raise ValueError(
                f"launch dim conflict: {a_list} vs {b_list}"
            )
    return result


def _build_tile_op(
    arg: Value,
    dim_names: list[str],
    *,
    op_dims: list[str],
    tile_idxs: list[Value],
    tile_sizes: list[int],
) -> tuple[list[Op], Value]:
    """Build TileOps that contract the dims of ``arg`` named in
    ``op_dims`` to the matching ``tile_sizes`` at the matching
    ``tile_idxs``. Each tiling step produces a new value that
    subsequent steps operate on, so the same dim can be tiled
    multiple times (e.g. K=64 then K=16).
    """
    assert len(dim_names) == len(arg.type.sizes), (
        f"_build_tile_op: dim_names ({dim_names}) must have one entry "
        f"per arg dim ({arg.type.sizes})"
    )
    assert len(op_dims) == len(tile_idxs) == len(tile_sizes), (
        "_build_tile_op: op_dims, tile_idxs, tile_sizes must have the "
        "same length"
    )
    name_to_idx = {n: i for i, n in enumerate(dim_names)}
    ops: list[Op] = []
    cur = arg
    maybe_partial_dims: set[str] = set()
    for name, iv, ts in zip(op_dims, tile_idxs, tile_sizes):
        d = name_to_idx.get(name)
        if d is None:
            continue
        maybe_partial = name in maybe_partial_dims or cur.type.sizes[d] % ts != 0
        if maybe_partial:
            maybe_partial_dims.add(name)
        new_type = clone_type(cur.type)
        indices: list[Value | int] = [0] * len(new_type.sizes)
        indices[d] = iv
        new_type.sizes[d] = ts
        tile_op = TileOp.build(
            cur, indices, new_type, maybe_partial=maybe_partial,
        )
        ops.append(tile_op)
        cur = tile_op.result()
    return ops, cur


def tile_matmul(
    module: ModuleOp,
    op_dims: list[str],
    tile_sizes: list[int],
    iteration_bindings: list[
        tuple[CollectiveLevel, CollectiveAxis] | None
    ],
    for_loop_labels: list[str | None],
    op_label_filter: str | None = None,
    epilogue_label_filter: str | None = None,
    allow_partial_tiles: bool = False,
) -> ModuleOp:
    """Convenience entry point."""
    return TileMatmul(
        op_dims, tile_sizes, iteration_bindings, for_loop_labels,
        op_label_filter, epilogue_label_filter, allow_partial_tiles,
    ).run(module)
