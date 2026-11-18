"""Tile the parallel and reduction dimensions of reductions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast, overload

if TYPE_CHECKING:
    from ..jit_compiler import PtxSequencePlan

from ..base.ir import (
    Block,
    Op,
    Value,
    clone_op,
    clone_type,
    find_ancestor_of_type,
)
from ..base.lowering_pass import LoweringPass, lowers
from ..gpu_dialect import (
    build_hardware_id,
    AddrMemTensorType,
    BlockIdOp,
    F32Type,
    GmemTensorType,
    MatmulOp,
    MemoryEffect,
    ReshapeOp,
    is_side_effect_free,
    AllocOp,
    CollectiveAxis,
    CollectiveLevel,
    ConstantOp,
    ElementwiseOp,
    ForOp,
    IndexConstantOp,
    IndexAddOp,
    IndexDivOp,
    IndexLtOp,
    IndexModOp,
    IndexMulOp,
    IfOp,
    KernelOp,
    MemTensorType,
    ModuleOp,
    ReduceOp,
    RmemTensorType,
    WgrmemTensorType,
    TileOp,
    root_buffer,
)
from ._utils import select_region, tile_region, label_matches, fresh_alloc


def _materialize_tile(
    arg: Value,
    dims: list[int],
    tile_sizes: dict[int, int],
    ivs: dict[int, Value],
) -> tuple[list[Op], Value]:
    if not dims:
        return [], arg

    typ = clone_type(arg.type)
    assert isinstance(typ, MemTensorType)
    indices: list[Value | int] = [0] * typ.rank
    sizes = list(typ.sizes)
    for dim in dims:
        sizes[dim] = tile_sizes[dim]
        indices[dim] = ivs[dim]
    typ.sizes = sizes
    tile_op = TileOp.build(arg, indices, typ, maybe_partial=isinstance(typ, AddrMemTensorType) and any(
        cast(int, cast(MemTensorType, arg.type).sizes[dim]) % tile_sizes[dim] for dim in dims))
    return [tile_op], tile_op.result()


def _materialize_partials(
    reduction_fn: str,
    reduction_dst: Value,
    num_partials: int,
) -> tuple[list[Op], Value]:
    # TODO: Resolve gmem allocations through buffer promotion, buffer reuse,
    # or scratch gmem kernel arguments. See the design notes:
    # https://docs.google.com/document/d/1UG1GMse251yAtngItIoH6W46qGGJE3VitYRJ0kw_0BM/edit?tab=t.0#heading=h.fi4redhw9yo6
    assert isinstance(reduction_dst.type, MemTensorType)
    reduction_dim = reduction_dst.type.rank - 1
    if isinstance(reduction_dst.definition, (AllocOp, Block)):
        sizes = list(reduction_dst.type.sizes)
        sizes[reduction_dim] = num_partials
        attrs: dict[str, object] = {"sizes": sizes}
        if isinstance(reduction_dst.type, AddrMemTensorType):
            attrs["strides"] = AddrMemTensorType.canonical_strides(sizes)
        if isinstance(reduction_dst.type, WgrmemTensorType) and num_partials != 1:
            attrs["layout"] = "wgmma_layout_0"
        typ = clone_type(reduction_dst.type, attrs)
        neutral_element = ReduceOp.NEUTRAL_ELEMENTS[reduction_fn]
        assert neutral_element is not None
        alloc = fresh_alloc(typ, reduction_dst)
        neutral = ConstantOp.build(
            neutral_element,
            RmemTensorType.build([], reduction_dst.type.dtype),
        )
        fill = ElementwiseOp.build(
            alloc.result(),
            srcs=[neutral.result()],
            fn="fill",
        )
        return [alloc, neutral, fill], alloc.result()

    tile = reduction_dst.definition
    assert isinstance(tile, TileOp), (
        "TileReduce reduction tiling only supports TileOp destination view chains"
    )
    ops, replayed_arg = _materialize_partials(
        reduction_fn,
        tile.arg,
        num_partials,
    )
    tile_type = tile.result().type
    replayed_type = replayed_arg.type
    assert isinstance(tile_type, MemTensorType)
    assert isinstance(replayed_type, MemTensorType)
    sizes = list(tile_type.sizes)
    sizes[reduction_dim] = replayed_type.sizes[reduction_dim]
    typ = clone_type(replayed_type, {"sizes": sizes})
    replayed = TileOp.build(
        replayed_arg,
        list(tile.indices),
        typ,
        maybe_partial=tile.maybe_partial,
    )
    ops.append(replayed)
    return ops, replayed.result()


def _materialize_iteration_space(
    dim: int,
    bounds: dict[int, int],
    parallel_iteration_space: (
        tuple[CollectiveLevel, CollectiveAxis] | None
    ),
    divisor: int,
) -> tuple[list[Op], Value, ForOp | None]:
    if parallel_iteration_space is None:
        loop = ForOp.build(
            lb=0,
            ub=bounds[dim],
            attrs={"label": f"d{dim}"},
        )
        return [loop], loop.iv, loop

    level, axis = parallel_iteration_space
    id_ops, parallel_id = build_hardware_id(level, axis)
    cdivisor = IndexConstantOp.build(divisor)
    div = IndexDivOp.build(
        parallel_id, cdivisor.result(),
    )
    cbound = IndexConstantOp.build(bounds[dim])
    mod = IndexModOp.build(
        div.result(), cbound.result(),
    )
    return [*id_ops, cdivisor, div, cbound, mod], mod.result(), None


def _narrow_collective(
    collective: dict[CollectiveAxis, CollectiveLevel],
    parallel_iteration_space: (
        tuple[CollectiveLevel, CollectiveAxis] | None
    ),
) -> dict[CollectiveAxis, CollectiveLevel]:
    result = dict(collective)
    if parallel_iteration_space is not None:
        level, axis = parallel_iteration_space
        assert level.join(collective[axis]) == collective[axis]
        result[axis] = level
    return result


def _clone_hero_with_epilogue(
    hero: ReduceOp, epilogue: list[Op], dst: Value, src: Value,
    collective: dict[CollectiveAxis, CollectiveLevel],
) -> list[Op]:
    return [
        clone_op(
            hero, {hero.dst: dst, hero.src: src}, {"collective": collective},
        ),
        *tile_region(epilogue, hero.src, {hero.dst: dst}, [], collective),
    ]


@dataclass
class ReductionModules:
    """Ordered launches sharing an uninitialized partial-result workspace."""

    partials: ModuleOp
    reduction: ModuleOp
    workspace_shape: tuple[int, ...]

    def launch_plan(self) -> PtxSequencePlan:
        """Build ordered launches with one automatically allocated F32 workspace.

        ``launch = stages.launch_plan()()`` accepts the original kernel
        arguments. Scratch storage is retained and reused by the launcher.
        """
        from ..jit_compiler import PtxRoutePlan, PtxSequencePlan, ScratchBuffer
        from ..ptx_codegen import make_ptx_kernel

        kernel = cast(KernelOp, self.partials.body.ops[0])
        workspace = ScratchBuffer(GmemTensorType.build(self.workspace_shape, F32Type()))
        bindings = [*range(len(kernel.signature) - 1), workspace]
        return PtxSequencePlan(
            (PtxRoutePlan(make_ptx_kernel(phase), bindings)
             for phase in (self.partials, self.reduction)),
            reuse_scratch=True,
        )


def _replay_address_view(value: Value, root: Value, replacement: Value,
                         extra_shape: tuple[int, ...] = ()) -> tuple[list[Op], Value]:
    """Replay a logical view on storage with new strides and trailing axes."""
    if value is root:
        return [], replacement
    op = value.definition
    if not isinstance(op, (TileOp, ReshapeOp)):
        raise ValueError('partition destinations must be Tile/Reshape views of a global buffer')
    ops, arg = _replay_address_view(op.arg, root, replacement, extra_shape)
    typ = cast(AddrMemTensorType, arg.type)
    shape = (*cast(MemTensorType, value.type).static_shape, *extra_shape)
    if isinstance(op, TileOp):
        view = TileOp.build(arg, [*op.indices, *([0] * len(extra_shape))],
                            clone_type(typ, {'sizes': list(shape)}),
                            attrs=op.extra_attrs, maybe_partial=op.maybe_partial)
    else:
        strides = typ.reshape_strides(shape)
        if strides is None:
            raise ValueError('partition workspace cannot preserve this reshape address order')
        view = ReshapeOp.build(arg, clone_type(typ, {'sizes': list(shape), 'strides': list(strides)}))
    return [*ops, view], view.result()


def _partition_reduction(
    module: ModuleOp,
    tile_size: int,
    axis: CollectiveAxis,
    atomic: bool,
    hero_label_filter: str | None,
    workspace_alignment: int,
    num_partials: int | None,
) -> ReductionModules | ModuleOp:
    """Materialize parallel reduction tiles and their launch boundary.

    ReduceOp and MatmulOp differ only in their source operands, combination
    function, and initializer representation. Workspace, launch ownership,
    cloning, and consumer placement use the same transformation.
    """
    from .verify_gpu_dialect import VerifyGpuDialect

    VerifyGpuDialect().run(module)
    if len(module.body.ops) != 1 or not isinstance(module.body.ops[0], KernelOp):
        raise ValueError("reduction partitioning requires exactly one kernel")
    kernel = module.body.ops[0]
    heroes = [
        op
        for op in kernel.body.ops
        if isinstance(op, (ReduceOp, MatmulOp)) and label_matches(op, hero_label_filter)
    ]
    if len(heroes) != 1:
        raise ValueError(
            "reduction partitioning requires one selected top-level producer"
        )
    hero = heroes[0]
    sources = [hero.src] if isinstance(hero, ReduceOp) else [hero.lhs, hero.rhs]
    types = [cast(MemTensorType, v.type) for v in [*sources, hero.dst]]
    if any(
        not isinstance(t, GmemTensorType) or t.is_dynamic or t.rank == 0 for t in types
    ):
        raise ValueError("reduction partitioning requires static global-memory tensors")
    n = types[0].static_shape[-1]
    p = types[-1].static_shape[-1] if isinstance(hero, ReduceOp) else 1
    if (
        type(tile_size) is not int
        or tile_size < 1
        or tile_size % p
    ):
        raise ValueError(
            "reduction tiles must preserve the destination's modulo partitions"
        )
    if (
        hero.collective[axis] != CollectiveLevel.GRID
        or kernel.cluster_dim[axis] not in (None, 1)
    ):
        raise ValueError("reduction partitioning requires a grid-collective, unclustered axis")
    if type(workspace_alignment) is not int or workspace_alignment < 4 or workspace_alignment % 4:
        raise ValueError('workspace alignment must be a positive multiple of the F32 element size')
    index = kernel.body.ops.index(hero)
    if any(not is_side_effect_free(op) for op in kernel.body.ops[:index]):
        raise ValueError("reduction partitioning requires a side-effect-free prologue")
    epilogue = select_region(kernel.body.ops, index + 1, ".*")
    if len(epilogue) != len(kernel.body.ops) - index - 1:
        raise ValueError("all consumers must belong to the final reduction phase")
    fn = hero.fn if isinstance(hero, ReduceOp) else "sum"
    identity = ReduceOp.NEUTRAL_ELEMENTS[fn]
    if identity is None:
        raise ValueError("partitioning requires a reduction identity")
    tensor_init = None
    if isinstance(hero, ReduceOp):
        init = hero.init
    elif hero.init is None:
        init = None if hero.accumulate else 0.0
    elif cast(MemTensorType, hero.init.type).rank != 0:
        tensor_init, init = hero.init, 0.0
    elif isinstance(hero.init.definition, ConstantOp):
        init = hero.init.definition.value
        if not isinstance(init, (int, float)):
            raise ValueError("partitioned initializer must be scalar")
    else:
        raise ValueError(
            "partitioned contraction requires a constant scalar or tensor initializer"
        )
    parts = math.ceil(n / tile_size)
    if num_partials is not None:
        if type(num_partials) is not int or num_partials < parts:
            raise ValueError('num_partials must cover every source tile')
        parts = num_partials
    if atomic:
        if types[-1].dtype != F32Type():
            raise ValueError("atomic reduction requires an F32 destination")
        if epilogue:
            raise ValueError(
                "atomic reduction requires epilogues in a subsequent kernel"
            )
        if not isinstance(hero, MatmulOp):
            raise NotImplementedError(
                "atomic writeback currently supports WGMMA contraction fragments"
            )
    partials = cast(ModuleOp, clone_op(module, {}))
    pkernel = cast(KernelOp, partials.body.ops[0])
    phero = cast(ReduceOp | MatmulOp, pkernel.body.ops[index])
    pkernel.attrs["name"] = f"{kernel.name}_partials"
    grid = list(pkernel.grid_dim)
    grid_extent = grid[axis] or 1
    grid[axis] = grid_extent * parts
    pkernel.attrs["grid_dim"] = grid
    prologue = list(pkernel.body.ops[:index])
    if grid_extent > 1:
        # Preserve the existing grid coordinate while giving each coordinate
        # a contiguous set of reduction partitions on the same physical axis.
        expanded = []
        for op in prologue:
            expanded.append(op)
            if isinstance(op, BlockIdOp) and op.axis == axis:
                count = IndexConstantOp.build(parts)
                coordinate = IndexDivOp.build(op.value, count.result())
                for user, operand in list(op.value.uses):
                    if user is not coordinate:
                        user.operands[operand] = coordinate.result()
                expanded.extend([count, coordinate])
        prologue = expanded
    partition_id = BlockIdOp.build(axis)
    partition_ops: list[Op] = [partition_id]
    partition = partition_id.value
    if grid_extent > 1:
        count = IndexConstantOp.build(parts)
        remainder = IndexModOp.build(partition, count.result())
        partition_ops.extend([count, remainder])
        partition = remainder.result()
    psources = [phero.src] if isinstance(phero, ReduceOp) else [phero.lhs, phero.rhs]
    tiles = [
        TileOp.build(
            value,
            [*([0] * (typ.rank - 1)), partition],
            clone_type(typ, {"sizes": [*typ.sizes[:-1], tile_size]}),
            maybe_partial=parts * tile_size > n,
        )
        for value, typ in zip(psources, types[:-1], strict=True)
    ]
    collective = dict(phero.collective)
    collective[axis] = CollectiveLevel.BLOCK
    attrs = phero.extra_attrs
    attrs.pop("accumulate", None)
    attrs.pop("dst_tiling_scheme", None)
    output_ops: list[Op] = []
    shape = types[-1].static_shape
    destination_root = root_buffer(phero.dst)
    assert destination_root is not None and isinstance(destination_root.type, GmemTensorType)
    root_shape = destination_root.type.static_shape
    row_elements = workspace_alignment // 4
    aligned_shape = (*root_shape[:-1], ((root_shape[-1] + row_elements - 1) // row_elements) * row_elements)
    workspace_shape = (parts, *aligned_shape)
    if atomic:
        destination = phero.dst
        attrs["atomic_add"] = True
    else:
        scratch = Value(
            "reduction_partials", GmemTensorType.build(list(workspace_shape), F32Type())
        )
        pkernel.body.append_argument(scratch)
        partial = TileOp.build(
            scratch,
            [partition, *([0] * len(root_shape))],
            GmemTensorType.build([1, *aligned_shape], F32Type()),
        )
        view = ReshapeOp.build(
            partial.result(), GmemTensorType.build(list(aligned_shape), F32Type())
        )
        output_ops.extend([partial, view])
        logical = TileOp.build(view.result(), [0] * len(root_shape),
                               clone_type(view.result().type, {'sizes': list(root_shape)}))
        output_ops.append(logical)
        replayed, destination = _replay_address_view(phero.dst, destination_root, logical.result())
        output_ops.extend(replayed)
        attrs["dst_tiling_scheme"] = MemoryEffect.fresh_tiling_scheme_id()
    mapping = {old: tile.result() for old, tile in zip(psources, tiles, strict=True)}
    mapping[phero.dst] = destination
    if isinstance(phero, MatmulOp):
        zero = ConstantOp.build(identity, RmemTensorType.build([], F32Type()))
        output_ops.append(zero)
        reduced = MatmulOp.build(
            mapping[phero.lhs],
            mapping[phero.rhs],
            destination,
            collective,
            attrs=attrs,
            init=zero.result(),
        )
    else:
        attrs["label"] = "reduce_partial"
        reduced = ReduceOp.build(
            destination, tiles[0].result(), fn, collective, init=identity, attrs=attrs
        )
    pkernel.body.ops = [
        *prologue,
        *partition_ops,
        *tiles,
        *output_ops,
        reduced,
    ]
    if atomic:
        return partials
    reduction = cast(ModuleOp, clone_op(module, {}))
    rkernel = cast(KernelOp, reduction.body.ops[0])
    rhero = cast(ReduceOp | MatmulOp, rkernel.body.ops[index])
    rkernel.attrs.update(
        name=f"{kernel.name}_final",
        grid_dim=list(kernel.grid_dim),
        block_dim=[None] * 3,
        cluster_dim=[None] * 3,
        num_warp_groups=None,
    )
    scratch_view = Value(
        "reduction_partials",
        GmemTensorType.build(
            [*root_shape, parts],
            F32Type(),
            strides=[*GmemTensorType.canonical_strides(aligned_shape), math.prod(aligned_shape)],
        ),
    )
    rkernel.body.append_argument(scratch_view)
    reduction_root = root_buffer(rhero.dst)
    assert reduction_root is not None
    replayed, partial_view = _replay_address_view(rhero.dst, reduction_root, scratch_view, (parts,))
    dst_type = cast(GmemTensorType, rhero.dst.type)
    dst_view = ReshapeOp.build(
        rhero.dst,
        clone_type(dst_type, {"sizes": [*shape, 1], "strides": [*dst_type.strides, 1]}),
    )
    reduced = ReduceOp.build(
        dst_view.result(),
        partial_view,
        fn,
        rhero.collective,
        init=init,
        attrs={"label": "reduce_final"},
    )
    # Restore the producer's logical rank for general consumer graphs. The
    # extra workspace reduction axis is an implementation detail of this phase.
    restore = ReshapeOp.build(dst_view.result(), clone_type(dst_type))
    restore.attrs["label"] = "reduce_epilogue"
    mapping = {rhero.dst: restore.result()}
    consumer_ops: list[Op] = [restore]
    if tensor_init is not None:
        rinit = cast(MatmulOp, rhero).init
        assert rinit is not None
        initial = AllocOp.build_with_fresh_id(
            clone_type(dst_type, {"dtype": F32Type()}),
            attrs={"label": "reduce_epilogue"},
        )
        consumer_ops.extend(
            [
                initial,
                ElementwiseOp.build(
                    initial.result(),
                    [rinit],
                    "convert",
                    attrs={"label": "reduce_epilogue"},
                ),
                ElementwiseOp.build(
                    restore.result(),
                    [restore.result(), initial.result()],
                    "add",
                    attrs={"label": "reduce_epilogue"},
                ),
            ]
        )
    for op in rkernel.body.ops[index + 1 :]:
        cloned = clone_op(op, mapping, {"label": "reduce_epilogue"})
        mapping.update(zip(op.results, cloned.results, strict=True))
        consumer_ops.append(cloned)
    rkernel.body.ops = [*rkernel.body.ops[:index], *replayed, dst_view, reduced, *consumer_ops]
    return ReductionModules(partials, reduction, workspace_shape)

class TileReduce(LoweringPass):
    """Tile reduction iteration spaces and selected epilogues.

    ``tile_sizes`` maps each tiled dimension to its tile size. The
    ``parallel_iteration_space`` applies to every tiled dimension, or is
    ``None`` to create sequential iteration spaces.
    Nested tiling is expressed by composing multiple pass invocations.
    A dimension without a parallel iteration space is lowered to a sequential
    loop whose label is derived from the dimension.
    The vector accumulation option selects vector accumulation instead of
    materialized partial reductions. ``partial_size`` selects the accumulator
    width (for example 32 for one scalar per warp lane). By default it equals
    the source tile width. Register-source loops should be unrolled before
    PTX lowering so register views have static offsets.
    ``linearize`` distributes the product of parallel tile counts across
    grid and thread, warp, or warp-group IDs. ``linearize_grid`` includes
    existing grid axes (for example a batch axis) in that flattened space.
    """

    tile_sizes: dict[int, int]
    parallel_iteration_space: tuple[CollectiveLevel, CollectiveAxis] | None
    hero_label_filter: str | None
    epilogue_label_filter: str | None
    vector_accumulation: bool
    partial_size: int | None
    linearize: bool
    linearize_grid: tuple[CollectiveAxis, ...]

    def __init__(
        self,
        tile_sizes: dict[int, int],
        parallel_iteration_space: (tuple[CollectiveLevel, CollectiveAxis] | None),
        hero_label_filter: str | None = None,
        epilogue_label_filter: str | None = None,
        vector_accumulation: bool = False,
        partial_size: int | None = None,
        linearize: bool = False,
        linearize_grid: tuple[CollectiveAxis, ...] = (),
    ) -> None:
        super().__init__(
            tile_sizes=dict(tile_sizes),
            parallel_iteration_space=parallel_iteration_space,
            hero_label_filter=hero_label_filter,
            epilogue_label_filter=epilogue_label_filter,
            vector_accumulation=vector_accumulation,
            partial_size=partial_size,
            linearize=linearize,
            linearize_grid=linearize_grid,
        )
        self._linear_spaces: dict[Op, tuple[Value, dict[int, int], int]] = {}

    @overload
    def split(self, module: ModuleOp, *, atomic: Literal[False] = False, workspace_alignment: int = 4, num_partials: int | None = None) -> ReductionModules: ...

    @overload
    def split(self, module: ModuleOp, *, atomic: Literal[True], workspace_alignment: int = 4, num_partials: int | None = None) -> ModuleOp: ...

    @overload
    def split(self, module: ModuleOp, *, atomic: bool, workspace_alignment: int = 4, num_partials: int | None = None) -> ReductionModules | ModuleOp: ...

    def split(self, module: ModuleOp, *, atomic: bool = False, workspace_alignment: int = 4, num_partials: int | None = None) -> ReductionModules | ModuleOp:
        """Tile a reduction or contraction across blocks in ordered phases.

        Configure one last-source-axis tile and a block binding. Both launches
        retain the original arguments and append a shared F32 workspace.
        Consumers run once in the final phase, through normal tiling/promotion.
        Atomic contractions instead require initialized F32 output and return
        one launch; consumers need a subsequent phase.
        ``workspace_alignment`` pads each workspace row to a byte alignment.
        Existing grid coordinates are preserved when the partition binding
        shares their physical axis.
        ``num_partials`` may include empty partitions to retain a chosen
        hardware concurrency; padded source elements use the reduction identity.
        """
        if self.parallel_iteration_space is None or self.parallel_iteration_space[0] != CollectiveLevel.BLOCK or len(self.tile_sizes) != 1 or self.vector_accumulation or self.partial_size is not None:
            raise ValueError('split requires one reduction tile and a block binding')
        dim, tile_size = next(iter(self.tile_sizes.items()))
        heroes = [op for kernel in module.body.ops if isinstance(kernel, KernelOp)
                  for op in kernel.body.ops if isinstance(op, (ReduceOp, MatmulOp))
                  and label_matches(op, self.hero_label_filter)]
        if len(heroes) != 1:
            raise ValueError('split requires one selected reduction or contraction')
        source = heroes[0].src if isinstance(heroes[0], ReduceOp) else heroes[0].lhs
        if dim != cast(MemTensorType, source.type).rank - 1:
            raise ValueError('split must tile the last source reduction axis')
        return _partition_reduction(module, tile_size, self.parallel_iteration_space[1],
                                    atomic, self.hero_label_filter, workspace_alignment, num_partials)

    def run(self, module: ModuleOp) -> ModuleOp:
        self._linear_spaces.clear()
        return super().run(module)

    @lowers(KernelOp)
    def lower_kernel(
        self, op: KernelOp,
    ) -> tuple[list[Op], list[object]]:
        if self.linearize:
            self._prepare_linear_space(op)
        self._rewrite_block(op.body)
        return [op], []

    def _prepare_linear_space(self, kernel: KernelOp) -> None:
        """Distribute a flattened tile space over grid and intra-block IDs."""
        from ..base.ir import find_ops_of_type

        heroes = [op for op in kernel.body.ops if isinstance(op, ReduceOp)
                  and label_matches(op, self.hero_label_filter)]
        if len(heroes) != 1 or self.parallel_iteration_space is None:
            raise ValueError('linear tiling requires one selected top-level reduction and a hardware binding')
        hero = heroes[0]
        level, axis = self.parallel_iteration_space
        if level not in (CollectiveLevel.THREAD, CollectiveLevel.THREAD_32, CollectiveLevel.WG):
            raise ValueError('linear tiling requires a thread, warp, or warp-group binding')
        if hero.collective[axis] != CollectiveLevel.GRID:
            raise ValueError('linear tiling must begin at grid scope')
        self._assert_preconditions(hero, [])
        group_size = 128 if level == CollectiveLevel.WG else 32 if level == CollectiveLevel.THREAD_32 else 1
        threads = kernel.block_dim[axis] or 128
        if threads % group_size or any(kernel.block_dim[a] not in (None, 1) for a in CollectiveAxis if a != axis):
            raise ValueError('linear tiling requires complete groups on one block axis')
        if axis in self.linearize_grid or len(set(self.linearize_grid)) != len(self.linearize_grid):
            raise ValueError('the distributed axis cannot also be a flattened grid axis')
        if any(kernel.cluster_dim[a] not in (None, 1) for a in (axis, *self.linearize_grid)):
            raise ValueError('linear tiling cannot flatten a clustered axis')
        bounds = {dim: math.ceil(cast(MemTensorType, hero.src.type).static_shape[dim] / size)
                  for dim, size in self.tile_sizes.items()}
        count = math.prod(bounds.values())
        ops, local = build_hardware_id(level, axis)
        block = BlockIdOp.build(axis)
        groups = IndexConstantOp.build(threads // group_size)
        base = IndexMulOp.build(block.value, groups.result())
        linear = IndexAddOp.build(base.result(), local)
        ops.extend([block, groups, base, linear])
        for grid_axis in reversed(self.linearize_grid):
            extent = kernel.grid_dim[grid_axis] or 1
            stride = IndexConstantOp.build(count)
            divide = IndexDivOp.build(linear.result(), stride.result())
            size = IndexConstantOp.build(extent)
            coordinate = IndexModOp.build(divide.result(), size.result())
            for old in find_ops_of_type(kernel, BlockIdOp):
                if old.axis == grid_axis:
                    for user, operand in list(old.value.uses):
                        user.operands[operand] = coordinate.result()
            ops.extend([stride, divide, size, coordinate])
            count *= extent
        grid = list(kernel.grid_dim)
        launch_count = math.ceil(count / (threads // group_size))
        if grid[axis] not in (None, launch_count):
            raise ValueError('linear tiling conflicts with the existing grid extent')
        grid[axis] = launch_count
        for grid_axis in self.linearize_grid:
            grid[grid_axis] = 1
        # These axes are deliberately remapped together with their ID uses.
        # The public setter only fills unknown dimensions; it cannot express
        # a coordinate-space rewrite.
        kernel.attrs["grid_dim"] = grid
        block_dims = list(kernel.block_dim)
        block_dims[axis] = threads
        kernel.block_dim = block_dims
        kernel.body.ops = [*ops, *kernel.body.ops]
        self._linear_spaces[hero] = linear.result(), bounds, count

    def _rewrite_block(self, block: Block) -> None:
        for op in block.ops:
            for nested in op.blocks:
                self._rewrite_block(nested)

        ops = list(block.ops)
        new_ops: list[Op] = []
        i = 0
        while i < len(ops):
            op = ops[i]
            if not (
                isinstance(op, ReduceOp) and label_matches(op, self.hero_label_filter)
            ):
                new_ops.append(op)
                i += 1
                continue
            epilogue = select_region(ops, i + 1, self.epilogue_label_filter)
            new_ops.extend(
                self._rewrite_hero_with_epilogue(
                    op,
                    epilogue,
                )
            )
            i += 1 + len(epilogue)
        block.ops = new_ops

    def _rewrite_hero_with_epilogue(
        self, hero: ReduceOp, epilogue: list[Op],
    ) -> list[Op]:
        self._assert_preconditions(hero, epilogue)
        if not self.tile_sizes:
            return [hero, *epilogue]
        if self.linearize:
            return self._rewrite_linear(hero, epilogue)
        src_type = hero.src.type
        assert isinstance(src_type, MemTensorType)
        bounds = {
            dim: math.ceil(cast(int, src_type.sizes[dim]) / tile_size)
            for dim, tile_size in self.tile_sizes.items()
        }
        self._update_launch_dimensions(hero, bounds)

        if src_type.rank - 1 not in self.tile_sizes:
            return self._rewrite_hero_with_epilogue_parallel(
                hero, epilogue, bounds,
            )
        if self.vector_accumulation:
            return self._rewrite_hero_with_epilogue_reduction_accumulate(
                hero, epilogue, bounds,
            )
        return self._rewrite_hero_with_epilogue_reduction_hierarchical(
            hero, epilogue, bounds,
        )

    def _rewrite_linear(self, hero: ReduceOp, epilogue: list[Op]) -> list[Op]:
        linear, bounds, count = self._linear_spaces[hero]
        ops: list[Op] = []
        ivs = {}
        divisor = 1
        for dim in reversed(list(self.tile_sizes)):
            stride = IndexConstantOp.build(divisor)
            divide = IndexDivOp.build(linear, stride.result())
            size = IndexConstantOp.build(bounds[dim])
            coordinate = IndexModOp.build(divide.result(), size.result())
            ops.extend([stride, divide, size, coordinate])
            ivs[dim] = coordinate.result()
            divisor *= bounds[dim]
        src_ops, src = _materialize_tile(hero.src, list(self.tile_sizes), self.tile_sizes, ivs)
        dst_ops, dst = _materialize_tile(hero.dst, list(self.tile_sizes), self.tile_sizes, ivs)
        collective = _narrow_collective(hero.collective, self.parallel_iteration_space)
        body = [*src_ops, *dst_ops, clone_op(hero, {hero.dst: dst, hero.src: src}, {'collective': collective})]
        body.extend(tile_region(epilogue, hero.src, {hero.dst: dst, hero.src: src},
                                [(dim, size, ivs[dim]) for dim, size in self.tile_sizes.items()], collective))
        extent = IndexConstantOp.build(count)
        active = IndexLtOp.build(linear, extent.result())
        return [*ops, extent, active, IfOp.build(active.result(), Block(ops=body))]

    def _update_launch_dimensions(
        self,
        hero: ReduceOp,
        bounds: dict[int, int],
    ) -> None:
        if self.parallel_iteration_space is None:
            return
        launch_dim = math.prod(bounds.values())
        kernel = find_ancestor_of_type(hero, KernelOp)
        assert kernel is not None
        level, axis = self.parallel_iteration_space
        parent = hero.collective[axis]
        if parent in (CollectiveLevel.WG, CollectiveLevel.THREAD_32):
            parent_threads = 128 if parent == CollectiveLevel.WG else 32
            child_threads = 32 if level == CollectiveLevel.THREAD_32 else 1
            assert launch_dim * child_threads == parent_threads, (
                "nested intra-block tiling must cover every child of its parent"
            )
            return
        if level == CollectiveLevel.THREAD_32:
            assert axis == CollectiveAxis.X
            launch_dim *= 32
        elif level == CollectiveLevel.WG:
            assert axis == CollectiveAxis.X
            launch_dim *= 128
        if level == CollectiveLevel.BLOCK:
            launch_dims = list(kernel.grid_dim)
        else:
            launch_dims = list(kernel.block_dim)
        assert launch_dims[axis] in (None, launch_dim), (
            f"launch dim conflict: {launch_dims[axis]} vs {launch_dim}"
        )
        launch_dims[axis] = launch_dim
        if level == CollectiveLevel.BLOCK:
            kernel.grid_dim = launch_dims
        else:
            kernel.block_dim = launch_dims

    def _rewrite_hero_with_epilogue_parallel(
        self, hero: ReduceOp, epilogue: list[Op], bounds: dict[int, int],
    ) -> list[Op]:
        new_ops: list[Op] = []
        body = new_ops
        ivs: dict[int, Value] = {}
        divisor = 1
        parallel_dims = list(self.tile_sizes)
        for dim in parallel_dims:
            iteration_ops, iv, inner_loop = _materialize_iteration_space(
                dim, bounds, self.parallel_iteration_space, divisor,
            )
            ivs[dim] = iv
            body.extend(iteration_ops)
            divisor *= bounds[dim]
            if inner_loop is not None:
                body = inner_loop.body.ops

        src_ops, src = _materialize_tile(
            hero.src, parallel_dims, self.tile_sizes, ivs,
        )
        dst_ops, dst = _materialize_tile(
            hero.dst, parallel_dims, self.tile_sizes, ivs,
        )
        body.extend([*src_ops, *dst_ops])
        collective = (
            _narrow_collective(hero.collective, self.parallel_iteration_space)
            if parallel_dims else dict(hero.collective)
        )
        body.append(clone_op(hero, {hero.dst: dst, hero.src: src}, {'collective': collective}))
        body.extend(tile_region(
            epilogue, hero.src, {hero.dst: dst, hero.src: src},
            [(dim, self.tile_sizes[dim], ivs[dim]) for dim in parallel_dims], collective,
        ))
        return new_ops

    def _rewrite_hero_with_epilogue_reduction_hierarchical(
        self,
        hero: ReduceOp,
        epilogue: list[Op],
        bounds: dict[int, int],
    ) -> list[Op]:
        src_type = hero.src.type
        assert isinstance(src_type, MemTensorType)
        new_ops, partials = _materialize_partials(
            hero.fn,
            hero.dst,
            bounds[src_type.rank - 1]
            * cast(int, cast(MemTensorType, hero.dst.type).sizes[-1]),
        )
        iteration_ops, iv, inner_loop = _materialize_iteration_space(
            src_type.rank - 1,
            bounds,
            self.parallel_iteration_space,
            divisor=1,
        )
        new_ops.extend(iteration_ops)
        body = inner_loop.body.ops if inner_loop is not None else new_ops
        ivs = {src_type.rank - 1: iv}
        partials_ops, partial = _materialize_tile(
            partials,
            [src_type.rank - 1],
            {
                src_type.rank - 1: cast(
                    int, cast(MemTensorType, hero.dst.type).sizes[-1]
                )
            },
            ivs,
        )
        src_ops, src = _materialize_tile(
            hero.src,
            [src_type.rank - 1],
            self.tile_sizes,
            ivs,
        )
        body.extend(
            [
                *partials_ops,
                *src_ops,
                self._build_partial_reduce(hero, partial, src),
            ]
        )
        new_ops.extend(
            _clone_hero_with_epilogue(
                hero,
                epilogue,
                hero.dst,
                partials,
                dict(hero.collective),
            )
        )
        return new_ops

    def _rewrite_hero_with_epilogue_reduction_accumulate(
        self,
        hero: ReduceOp,
        epilogue: list[Op],
        bounds: dict[int, int],
    ) -> list[Op]:
        src_type = hero.src.type
        assert isinstance(src_type, MemTensorType)
        new_ops, accumulator = _materialize_partials(
            hero.fn,
            hero.dst,
            self.partial_size or self.tile_sizes[src_type.rank - 1],
        )
        iteration_ops, iv, loop = _materialize_iteration_space(
            src_type.rank - 1,
            bounds,
            None,
            divisor=1,
        )
        assert loop is not None
        new_ops.extend(iteration_ops)
        src_ops, src = _materialize_tile(
            hero.src,
            [src_type.rank - 1],
            self.tile_sizes,
            {src_type.rank - 1: iv},
        )
        loop.body.ops.extend(
            [
                *src_ops,
                self._build_partial_reduce(hero, accumulator, src),
            ]
        )
        new_ops.extend(
            _clone_hero_with_epilogue(
                hero,
                epilogue,
                hero.dst,
                accumulator,
                dict(hero.collective),
            )
        )
        return new_ops

    def _build_partial_reduce(
        self, hero: ReduceOp, dst: Value, src: Value,
    ) -> ReduceOp:
        attrs = hero.extra_attrs
        label = attrs.get("label")
        if isinstance(label, str):
            attrs["label"] = f"{label}_partial"
        return ReduceOp.build(
            dst, src, fn=hero.fn,
            collective=_narrow_collective(
                hero.collective, self.parallel_iteration_space,
            ),
            attrs=attrs,
        )

    def _assert_preconditions(
        self,
        hero: Op,
        epilogue: list[Op],
    ) -> None:
        assert isinstance(hero, ReduceOp)
        assert (
            hero.fn in ReduceOp.NEUTRAL_ELEMENTS
            and ReduceOp.NEUTRAL_ELEMENTS[hero.fn] is not None
        ), "TileReduce requires a reduction with a neutral element"

        src_type = hero.src.type
        assert isinstance(src_type, MemTensorType)
        rank = src_type.rank
        assert all(0 <= dim < rank for dim in self.tile_sizes), (
            f"TileReduce dimensions must be in [0, {rank})"
        )
        reduction_dim = rank - 1 if rank - 1 in self.tile_sizes else None
        parallel_dims = [dim for dim in self.tile_sizes if dim != reduction_dim]
        assert not parallel_dims or reduction_dim is None, (
            "TileReduce cannot tile parallel and reduction dimensions in one invocation"
        )
        if self.partial_size is not None:
            assert self.vector_accumulation, "partial_size requires vector accumulation"
            assert reduction_dim is not None
            assert 1 <= self.partial_size <= self.tile_sizes[reduction_dim], (
                "partial_size must fit the source tile"
            )
            assert self.partial_size >= cast(
                int, cast(MemTensorType, hero.dst.type).sizes[-1]
            ), "partial_size cannot be smaller than the final destination"
        if self.vector_accumulation:
            assert reduction_dim is not None, (
                "TileReduce vector accumulation requires reduction-dimension tiling"
            )
            assert self.parallel_iteration_space is None, (
                "TileReduce does not implement parallel vector accumulation; "
                "it would require atomic accumulator updates and is not "
                "currently considered useful"
            )
        for dim, tile_size in self.tile_sizes.items():
            size = cast(int, src_type.sizes[dim])
            assert 0 < tile_size and (isinstance(src_type, AddrMemTensorType) or tile_size <= size), (
                f"TileReduce requires dimension {dim} tile size to be in "
                f"[1, {size}]; got {tile_size}"
            )
            assert isinstance(src_type, AddrMemTensorType) or size % tile_size == 0, (
                f"TileReduce requires dimension {dim} size {size} to be "
                f"divisible by tile size {tile_size}"
            )
        if reduction_dim is not None:
            dst_size = cast(int, cast(MemTensorType, hero.dst.type).sizes[-1])
            assert self.tile_sizes[reduction_dim] % dst_size == 0, (
                "reduction tiles must preserve the destination's modulo partition"
            )
            if self.vector_accumulation:
                accumulator_size = self.partial_size or self.tile_sizes[reduction_dim]
                assert accumulator_size % dst_size == 0, (
                    "partial_size must preserve the destination's modulo partition"
                )
            current = hero.dst
            while not isinstance(current.definition, (AllocOp, Block)):
                tile = current.definition
                assert isinstance(tile, TileOp), (
                    "TileReduce reduction tiling only supports TileOp "
                    "destination view chains"
                )
                current = tile.arg

        if self.parallel_iteration_space is None:
            return

        level, axis = self.parallel_iteration_space
        # A block partition may coexist with clusters, but its physical grid
        # must still contain complete clusters.
        kernel = find_ancestor_of_type(hero, KernelOp)
        assert kernel is not None
        assert level != CollectiveLevel.CLUSTER, (
            "TileReduce does not support cluster-aware iteration spaces"
        )
        if level == CollectiveLevel.BLOCK:
            count = math.prod(
                math.ceil(cast(int, src_type.sizes[d]) / t) for d, t in self.tile_sizes.items()
            )
            assert count % (kernel.cluster_dim[axis] or 1) == 0, (
                "TileReduce block extent must be divisible by the cluster dimension"
            )

        # Nested intra-block levels use relative IDs (the modulo in each
        # iteration space) while retaining the outer launch extent.
        current_level = hero.collective[axis]
        assert (
            self.linearize and current_level == CollectiveLevel.GRID and level < CollectiveLevel.BLOCK
            or
            current_level == CollectiveLevel.GRID
            and level == CollectiveLevel.BLOCK
            or current_level == CollectiveLevel.BLOCK
            and level < CollectiveLevel.BLOCK
            or current_level in (CollectiveLevel.WG, CollectiveLevel.THREAD_32)
            and level < current_level
        ), (
            "TileReduce only supports grid-to-block and "
            "block-to-intra-block parallel tiling"
        )
        if self.linearize:
            assert reduction_dim is None and isinstance(src_type, AddrMemTensorType) and isinstance(hero.dst.type, AddrMemTensorType), (
                'linear tiling requires address-memory operands and only parallel dimensions'
            )


def tile_reduce(
    module: ModuleOp,
    tile_sizes: dict[int, int],
    parallel_iteration_space: (tuple[CollectiveLevel, CollectiveAxis] | None),
    hero_label_filter: str | None = None,
    epilogue_label_filter: str | None = None,
    vector_accumulation: bool = False,
    partial_size: int | None = None,
    linearize: bool = False,
    linearize_grid: tuple[CollectiveAxis, ...] = (),
) -> ModuleOp:
    return TileReduce(
        tile_sizes,
        parallel_iteration_space,
        hero_label_filter,
        epilogue_label_filter,
        vector_accumulation,
        partial_size,
        linearize,
        linearize_grid,
    ).run(module)
