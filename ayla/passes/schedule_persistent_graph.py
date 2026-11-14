"""Compose graph phases on a resident tensor-core CTA grid."""

from __future__ import annotations

from typing import Any, Literal, cast
import math

from ..base.ir import Block, Op, Value, clone_op, clone_type, find_ops_of_type
from ..base.pipeline import Pass
from ..gemm_jit import select_persistent_gemm_schedule
from ..gemm_schedule_table import (
    PERSISTENT_SWIGLU_SCHEDULES, PERSISTENT_SWIGLU_WAVE_TAILS,
    PERSISTENT_SWIGLU_OUTPUT_PREFETCH,
    PERSISTENT_SWIGLU_OPERAND_MOVE_ATTRS,
)
from ..forward_schedule_table import PERSISTENT_ROW_PREFETCH_SCHEDULES, PERSISTENT_WARP_ROW_THREADS, STATIC_NORMALIZATION_ROW_THREADS, STATIC_GEMM_LAYOUT_OVERRIDES
from ..gemm_collection.swiglu import schedule_swiglu_projection, schedule_swiglu_projection_wave_tail
from ..gpu_dialect import (
    BarrierOp,
    ConstantOp,
    F32Type,
    BF16Type,
    GmemTensorType,
    KernelOp,
    MatmulOp,
    MemTensorType,
    ModuleOp,
    ReshapeOp,
    TileOp,
    ElementwiseOp,
    MoveOp,
    memory_effects,
    root_buffer,
    SmemTensorType,
    AllocOp,
    ReduceOp,
    GatherOp,
    BlockIdOp,
    CollectiveAxis,
    IndexConstantOp,
    IndexLtOp,
    IfOp,
)
from ..pipelines.tile_matmul_pipeline import _schedule_matmul
from .schedule_scalar_graph import ScheduleScalarGraph
from ._utils import can_reorder
from .pack_smem_allocs import ReuseSmemAllocs


class SchedulePersistentGraph(Pass):
    """Run every graph phase on the same cooperative grid.

    Tensor-core GEMMs start from measured library heuristics, adapted to a
    shared CTA size and rolled grid. Compatible pointwise and row reductions
    stay in registers. Schedule adaptations still require performance tuning.
    """

    def __init__(
        self,
        *,
        blocks: int,
        threads: int = 256,
        external_arguments: int = 0,
        shared_memory_kib: int = 220,
        tile_scheduler: str | None = None,
        tile_order: str = "source",
    ):
        super().__init__()
        if type(blocks) is not int or blocks < 1:
            raise ValueError("persistent grids require a positive resident block count")
        if type(threads) is not int or threads not in (256, 384, 512):
            raise ValueError(
                "persistent tensor-core scheduling requires two, three or four warpgroups per CTA"
            )
        if type(shared_memory_kib) is not int or shared_memory_kib < 1:
            raise ValueError("shared-memory budget must be positive")
        self.shared_memory_kib = shared_memory_kib
        if tile_scheduler not in (None, "static", "dynamic"):
            raise ValueError("tile_scheduler must be None, 'static' or 'dynamic'")
        self.tile_scheduler = tile_scheduler
        if tile_order not in ("source", "memory"):
            raise ValueError("tile_order must be 'source' or 'memory'")
        self.tile_order = tile_order
        self.blocks, self.threads, self.external_arguments = (
            blocks,
            threads,
            external_arguments,
        )

    def run(self, module: ModuleOp) -> ModuleOp:
        [source] = module.body.ops
        assert isinstance(source, KernelOp)
        sync = Value("forward_sync", GmemTensorType.build([2], F32Type()))
        source.body.append_argument(sync)
        arguments = [
            Value(arg.name, clone_type(arg.type)) for arg in source.body.arguments
        ]
        structural = [
            op
            for op in source.body.ops
            if isinstance(op, (ConstantOp, TileOp, ReshapeOp))
        ]
        effects = [op for op in source.body.ops if op not in structural]
        combined: list[Op] = []
        choices = []
        row_regions: dict[tuple[int, ...], int] = {}
        row_choices = []
        swiglu_phases = []
        groups = _map_groups(
            _swiglu_groups(
                _pointwise_groups(effects),
                set(source.body.arguments[: self.external_arguments]),
            )
        )
        groups = _row_copy_groups(groups, set(source.body.arguments[: self.external_arguments]))
        groups = _row_epilogue_groups(groups)
        groups = _gemm_map_groups(
            groups, set(source.body.arguments[: self.external_arguments]),
            rows_per_cta=self.threads // 2,
        )
        groups = _hoist_independent_copies(groups)
        groups = _inline_row_gathers(
            groups, set(source.body.arguments[: self.external_arguments])
        )
        for index, group in enumerate(groups):
            active_blocks = self.blocks
            vmap: dict[Value, Value] = {}
            phase_args = [
                Value(arg.name, clone_type(arg.type)) for arg in source.body.arguments
            ]
            vmap.update(zip(source.body.arguments, phase_args, strict=True))
            phase_ops = [clone_op(op, vmap) for op in structural]
            phase_effects = [clone_op(effect, vmap) for effect in group]
            phase_ops.extend(phase_effects)
            phase = ModuleOp.build(
                Block(
                    ops=[
                        KernelOp.build(
                            source.name,
                            Block(arguments=phase_args, ops=phase_ops),
                            grid_dim=(None, 1, 1),
                        )
                    ]
                )
            )
            phase_effect = phase_effects[0]
            if isinstance(phase_effect, MatmulOp) and len(group) > 1 and phase_effect.attrs.get("fused_map_epilogue") is not True:
                if (index + 1 < len(groups)
                        and isinstance(groups[index + 1][0], MatmulOp)
                        and root_buffer(cast(MatmulOp, groups[index + 1][0]).lhs) is root_buffer(cast(ElementwiseOp, group[-1]).dst)):
                    swiglu_phases.append(index)
                final = phase_effects[-1]
                assert isinstance(final, ElementwiseOp)
                width = cast(MemTensorType, final.dst.type).static_shape[1]
                widths = (208, 192, 128) if self.threads == 256 else (128, 96)
                n_tile = next((tile for tile in widths if width % tile == 0), 128)
                m, k = cast(MemTensorType, phase_effect.lhs.type).static_shape
                profile = (
                    PERSISTENT_SWIGLU_SCHEDULES.get((self.blocks, m, width, k))
                    if self.threads == 256 else None
                )
                stages, cyclic, tail, block_store = (
                    3 if self.threads == 512 else 4, True, None, False
                )
                if profile is not None:
                    n_tile, stages, cyclic, tail, block_store = profile
                wave_tail = PERSISTENT_SWIGLU_WAVE_TAILS.get(
                    ((self.blocks, m, width, k), (n_tile, stages, cyclic, tail, block_store))
                ) if self.threads == 256 else None
                fixed_static = self.tile_scheduler == "static" and self.tile_order == "source"
                if self.tile_scheduler is not None and not fixed_static:
                    # Work stealing and producer pools change tile ownership.
                    wave_tail = None
                projection_options: dict[str, Any] = {}
                # Operand cache hints describe these loads and do not depend
                # on the next tile's owner or on output-wave scheduling.
                if self.threads == 256:
                    move_attrs = PERSISTENT_SWIGLU_OPERAND_MOVE_ATTRS.get(
                        ((self.blocks, m, width, k), (n_tile, stages, cyclic, tail, block_store))
                    )
                    if move_attrs is not None:
                        projection_options["operand_move_attrs"] = move_attrs
                if wave_tail is not None:
                    projection_options.update(tail_n_tile=wave_tail[0], tail_operand_stages=wave_tail[1], grid_sync=phase_args[-1])
                    projection_options["output_prefetch_tiles"] = PERSISTENT_SWIGLU_OUTPUT_PREFETCH.get(
                        ((self.blocks, m, width, k), (n_tile, stages, cyclic, tail, block_store)), 0
                    ) if self.tile_scheduler is None or fixed_static else 0
                projection_scheduler = schedule_swiglu_projection_wave_tail if wave_tail is not None else schedule_swiglu_projection
                phase = projection_scheduler(
                    phase,
                    phase_effect.lhs,
                    phase_effect.rhs,
                    final.dst,
                    blocks=self.blocks,
                    threads=self.threads,
                    n_tile=n_tile,
                    operand_stages=stages,
                    cyclic_operand_pipeline=cyclic,
                    k_tail_tiles=tail,
                    block_store=block_store,
                    **projection_options,
                )
            elif isinstance(phase_effect, MatmulOp):
                lhs, rhs = (
                    cast(MemTensorType, phase_effect.lhs.type),
                    cast(MemTensorType, phase_effect.rhs.type),
                )
                m, k = lhs.static_shape
                n, _ = rhs.static_shape
                seed = select_persistent_gemm_schedule(
                    m, n, k, blocks=self.blocks, threads=self.threads
                )
                if seed is None or seed.builder_config is None:
                    raise ValueError(f"no tensor-core schedule for GEMM {m}x{n}x{k}")
                config: dict[str, Any] = dict(seed.builder_config)
                if self.tile_scheduler == "static" and self.tile_order == "source":
                    config.update(STATIC_GEMM_LAYOUT_OVERRIDES.get((self.blocks, self.threads, m, n, k), {}))
                active_blocks = config.get("output_grid_size", self.blocks)
                if type(active_blocks) is not int or not 0 < active_blocks <= self.blocks:
                    raise ValueError("phase grid must fit the resident cooperative grid")
                if len(group) > 1:
                    config["epilogue_label_filter"] = "forward_epi"
                    phase_effect = _inline_gemm_epilogue(
                        cast(KernelOp, phase.body.ops[0]), len(structural), phase_effects
                    )
                phase_effect.attrs["label"] = "mm"
                template = cast(ModuleOp, clone_op(phase, {}))
                while True:
                    phase = _schedule_matmul(
                        cast(ModuleOp, clone_op(template, {})), **config
                    )
                    # Measure the physical arena while retaining logical
                    # allocation identities for synchronization insertion.
                    budget_phase = phase
                    if config.get("unsafe_reuse_smem") is True:
                        budget_phase = ReuseSmemAllocs().run(cast(ModuleOp, clone_op(phase, {})))
                    shared = sum(
                        math.prod(
                            cast(SmemTensorType, alloc.result().type).static_shape
                        )
                        * cast(
                            BF16Type | F32Type,
                            cast(MemTensorType, alloc.result().type).dtype,
                        ).byte_size
                        for alloc in find_ops_of_type(budget_phase, AllocOp)
                        if isinstance(alloc.result().type, SmemTensorType)
                    )
                    if shared <= self.shared_memory_kib * 1024:
                        break
                    if config["operand_stages"] <= 2:
                        raise ValueError(
                            "GEMM schedule exceeds the persistent shared-memory budget"
                        )
                    config["operand_stages"] -= 1
                choices.append(
                    {
                        "shape": [m, n, k],
                        "phase": sum(isinstance(op, BarrierOp) and op.scope == "grid" for op in combined),
                        "source_shape": seed.source_shape,
                        "configuration": config,
                    }
                )
            elif all(
                isinstance(op, (ElementwiseOp, MoveOp, ReduceOp, GatherOp))
                for op in group
            ):
                reads_elsewhere = {
                    root
                    for op in effects
                    if op not in group
                    for effect in memory_effects(op) or []
                    if effect.kind == "R"
                    for root in effect.roots
                }
                # FX live outputs have no IR reads; retain their stores as well.
                externally_live = set(source.body.arguments[: self.external_arguments])
                live = {
                    vmap[op.dst] if op.dst in vmap else op.dst
                    for op in cast(list[ElementwiseOp | MoveOp | ReduceOp | GatherOp], group)
                    if root_buffer(op.dst) in reads_elsewhere | externally_live
                    and not (isinstance(op, GatherOp)
                             and op.attrs.get("inline_row_gather") is True)
                }
                reductions = [op for op in group if isinstance(op, ReduceOp)]
                # Map regions index each destination through its own views;
                # their output shapes need not broadcast to one common shape.
                shape = _group_shape(group) if reductions else ()
                prefetch_slots = 0
                warp_row_threads = 32
                if reductions and len(shape) == 2 and any(
                    op.attrs.get("row_map_epilogue") is True
                    and isinstance(cast(MemTensorType, cast(ElementwiseOp | MoveOp | ReduceOp | GatherOp, op).dst.type).dtype, BF16Type)
                    for op in group
                ):
                    warp_row_threads = PERSISTENT_WARP_ROW_THREADS.get(
                        (self.blocks, self.threads, shape[0], shape[1]), 32,
                    )
                if reductions:
                    occurrence = row_regions.get(shape, 0)
                    row_regions[shape] = occurrence + 1
                    matrices = [value.type for op in group for value in op.operands[1:]
                                if isinstance(value.type, GmemTensorType)
                                and value.type.static_shape == shape]
                    if (len(shape) == 2 and len(reductions) == 2 and matrices
                            and isinstance(matrices[0].dtype, BF16Type)):
                        prefetch_slots = PERSISTENT_ROW_PREFETCH_SCHEDULES.get(
                            (self.blocks, self.threads, shape[0], shape[1], occurrence), 0
                        )
                    if self.tile_scheduler is not None:
                        prefetch_slots = 0
                    row_choices.append({"shape": list(shape), "occurrence": occurrence,
                                        "prefetch_slots": prefetch_slots,
                                        "warp_row_threads": warp_row_threads})
                scheduler = ScheduleScalarGraph(
                    self.threads, self.blocks, phase_args[-1], vector_width=8,
                    row_threads=STATIC_NORMALIZATION_ROW_THREADS.get((self.blocks, self.threads, *shape))
                    if reductions and len(shape) == 2 and shape[1] > 1024
                    and self.tile_scheduler == "static" and self.tile_order == "source" else None,
                    row_prefetch_slots=prefetch_slots,
                    tile_tasks=self.tile_scheduler is not None,
                    warp_row_threads=warp_row_threads,
                )
                if reductions:
                    phase = scheduler.row_region(
                        phase,
                        cast(list[ElementwiseOp | MoveOp | ReduceOp | GatherOp], phase_effects),
                        live,
                        shape,
                    )
                else:
                    phase = scheduler.map_region(
                        phase,
                        cast(list[ElementwiseOp | MoveOp | GatherOp], phase_effects),
                        live,
                    )
            else:
                phase = ScheduleScalarGraph(
                    self.threads, self.blocks, phase_args[-1], vector_width=8,
                    tile_tasks=self.tile_scheduler is not None,
                ).run(phase)
            [kernel] = phase.body.ops
            assert isinstance(kernel, KernelOp)
            if tuple(kernel.block_dim) != (self.threads, 1, 1):
                raise ValueError(
                    f"phase {index}: inconsistent CTA size {kernel.block_dim}"
                )
            if sum(isinstance(op, MatmulOp) for op in effects) > 1:
                # Every CTA drains its TMA stores before the cooperative grid
                # barrier. Distinct phase lifetimes therefore cannot overlap.
                for allocation in find_ops_of_type(kernel, AllocOp):
                    if isinstance(allocation.result().type, SmemTensorType):
                        lifetime = allocation.attrs.get("reuse_group", "phase")
                        allocation.attrs["reuse_group"] = (
                            f"forward_phase_{index}_{lifetime}"
                        )
            mapping = dict(zip(kernel.body.arguments, arguments, strict=True))
            # Give each phase's loops/locks their own labels, retaining the
            # same SSA argument binding for scratch and live outputs.
            phase_body: list[Op] = []
            for op in kernel.body.ops:
                if isinstance(op, BarrierOp) and op.scope == "grid" and op.sync is None:
                    continue
                phase_body.append(clone_op(op, mapping))
            if active_blocks < self.blocks:
                # This condition is uniform within each CTA. Idle CTAs skip
                # its block barriers, then join the full cooperative barrier.
                block = BlockIdOp.build(CollectiveAxis.X)
                bound = IndexConstantOp.build(active_blocks)
                predicate = IndexLtOp.build(block.value, bound.result())
                combined.extend([
                    block, bound, predicate,
                    IfOp.build(predicate.result(), Block(ops=phase_body)),
                ])
            else:
                combined.extend(phase_body)
            if index + 1 < len(groups) or isinstance(phase_effect, MatmulOp):
                combined.append(
                    BarrierOp.build(
                        "grid", sync=arguments[-1], attrs={"drain_bulk": True}
                    )
                )
        kernel = KernelOp.build(
            source.name,
            Block(arguments=arguments, ops=combined),
            grid_dim=(self.blocks, 1, 1),
            block_dim=(self.threads, 1, 1),
            num_warp_groups=self.threads // 128,
            ptx_optimization_level=source.ptx_optimization_level,
            attrs={"cooperative": True},
        )
        kernel.attrs["forward_gemm_schedules"] = choices
        kernel.attrs["forward_row_schedules"] = row_choices
        kernel.attrs["forward_swiglu_phases"] = swiglu_phases
        result = ModuleOp.build(Block(ops=[kernel]))
        if self.tile_scheduler is not None:
            from .schedule_tile_graph import ScheduleTileGraph

            result = ScheduleTileGraph(
                cast(Literal["static", "dynamic"], self.tile_scheduler),
                tile_order=self.tile_order,
            ).run(result)
        return result


def _hoist_independent_copies(groups: list[list[Op]]) -> list[list[Op]]:
    """Share a map phase with a copy that can cross every intervening effect.

    Independent live outputs need their stores but not a separate grid barrier.
    Map regions give each destination its own loop, so their shapes may differ.
    Preserve copy snapshots by checking reads, writes and ordering effects,
    including those in the destination phase.
    """
    result = [list(group) for group in groups]
    for index in range(len(result) - 1, -1, -1):
        group = result[index]
        if len(group) != 1 or not isinstance(group[0], MoveOp):
            continue
        copy = group[0]
        # A matching head region already loads/stores the same row geometry.
        # Other map regions retain their independent destination loops.
        candidates = sorted(
            range(index), key=lambda i: not any(
                op.attrs.get("row_map_epilogue") is True for op in result[i]
            )
        )
        for earlier in candidates:
            candidate = result[earlier]
            if not candidate or not all(
                isinstance(op, (ElementwiseOp, MoveOp, GatherOp, ReduceOp)) for op in candidate
            ):
                continue
            if any(isinstance(op, ReduceOp) for op in candidate):
                if not any(op.attrs.get("row_map_epilogue") is True for op in candidate):
                    continue
                rows, width = _group_shape(candidate)
                heads = _row_heads(candidate, rows)
                shapes = {(rows, width), (rows // heads, heads * width),
                          (rows // heads, heads, width)}
                if (cast(MemTensorType, copy.src.type).static_shape not in shapes
                    or cast(MemTensorType, copy.dst.type).static_shape not in shapes):
                    continue
            if all(can_reorder(op, copy)
                   for between in result[earlier:index] for op in between):
                candidate.append(copy)
                result[index] = []
                break
    return [group for group in result if group]


def _row_heads(group: list[Op], rows: int) -> int:
    return next((
        typ.static_shape[1] for op in group
        if op.attrs.get("row_map_epilogue") is True
        and (typ := cast(MemTensorType, cast(ElementwiseOp | MoveOp, op).dst.type)).rank == 3
        and typ.static_shape[0] * typ.static_shape[1] == rows
    ), 1)


def _inline_row_gathers(
    groups: list[list[Op]], externally_live: set[Value],
) -> list[list[Op]]:
    """Read private gathered coefficients directly in head row epilogues.

    Every consumer must preserve the token axis through its views. Keep the
    materialization for other consumers, partial writers, and source/index
    updates that would change the original gather snapshot.
    """
    result = [list(group) for group in groups]
    external_roots = {root_buffer(value) for value in externally_live}
    for index, group in enumerate(result):
        if len(group) != 1 or not isinstance(group[0], GatherOp):
            continue
        gather = group[0]
        root = root_buffer(gather.dst)
        dst = cast(MemTensorType, gather.dst.type)
        if gather.dst is not root or root in external_roots or dst.rank != 2:
            continue
        readers: list[int] = []
        safe = True
        for position, candidate in enumerate(result):
            if position == index:
                continue
            for op in candidate:
                effects = memory_effects(op)
                if effects is None:
                    safe = False
                    break
                if any(effect.kind == "W" and root in effect.roots for effect in effects):
                    safe = False
                if any(effect.kind == "R" and root in effect.roots for effect in effects):
                    if position not in readers:
                        readers.append(position)
        if not safe or not readers:
            continue
        for position in readers:
            candidate = result[position]
            if (position <= index
                or not any(isinstance(op, ReduceOp) for op in candidate)
                or not any(op.attrs.get("row_map_epilogue") is True for op in candidate)):
                safe = False
                break
            rows, _ = _group_shape(candidate)
            if dst.static_shape[0] != rows // _row_heads(candidate, rows):
                safe = False
            for op in candidate:
                for value in op.operands[1:]:
                    if not isinstance(value.type, MemTensorType) or root_buffer(value) is not root:
                        continue
                    # Only reshape the inner axes and slice columns. A shifted
                    # token axis would require another position-index load.
                    while value is not gather.dst:
                        view = value.definition
                        if (not isinstance(view, (TileOp, ReshapeOp))
                            or cast(MemTensorType, value.type).rank < 2
                            or cast(MemTensorType, value.type).static_shape[0] != dst.static_shape[0]
                            or isinstance(view, TileOp) and (view.maybe_partial or view.indices[0] != 0)):
                            safe = False
                            break
                        value = view.arg
        inputs = {root_buffer(gather.src), root_buffer(gather.indices)}
        crossed = [op for candidate in result[index + 1:max(readers) + 1] for op in candidate]
        if not safe or any(
            memory_effects(op) is None or any(
                effect.kind == "W" and inputs.intersection(effect.roots)
                for effect in memory_effects(op) or []
            ) for op in crossed
        ):
            continue
        gather.attrs["inline_row_gather"] = True
        for position in readers:
            result[position].insert(0, gather)
        result[index] = []
    return [group for group in result if group]


def _broadcast_shape(
    left: tuple[int, ...], right: tuple[int, ...]
) -> tuple[int, ...] | None:
    rank = max(len(left), len(right))
    left, right = (1,) * (rank - len(left)) + left, (1,) * (rank - len(right)) + right
    if any(a != b and a != 1 and b != 1 for a, b in zip(left, right, strict=True)):
        return None
    return tuple(max(a, b) for a, b in zip(left, right, strict=True))


def _group_shape(group: list[Op]) -> tuple[int, ...]:
    if any(op.attrs.get("row_map_epilogue") is True for op in group):
        return cast(MemTensorType, next(op.src for op in group if isinstance(op, ReduceOp)).type).static_shape
    shape: tuple[int, ...] = ()
    for op in cast(list[ElementwiseOp | MoveOp | ReduceOp], group):
        if isinstance(op, MoveOp) and op.attrs.get("inline_row_copy") is True:
            continue
        shape = cast(
            tuple[int, ...],
            _broadcast_shape(shape, cast(MemTensorType, op.dst.type).static_shape),
        )
    return shape


def _pointwise_groups(effects: list[Op]) -> list[list[Op]]:
    groups: list[list[Op]] = []
    group: list[Op] = []
    shape: tuple[int, ...] = ()
    written: set[Value] = set()
    roots: set[Value] = set()
    for op in effects:
        pointwise = isinstance(op, (ElementwiseOp, MoveOp, ReduceOp))
        merged = (
            _broadcast_shape(
                shape,
                cast(
                    MemTensorType,
                    op.src.type if isinstance(op, ReduceOp) else op.dst.type,
                ).static_shape,
            )
            if pointwise
            else None
        )
        reads = op.operands[1:] if pointwise else []
        alias = any(
            root_buffer(value) in roots and value not in written
            for value in reads
            if isinstance(value.type, MemTensorType)
        )
        if merged is None or alias:
            if group:
                groups.append(group)
            group, shape, written, roots = [], (), set(), set()
        if not pointwise:
            groups.append([op])
            continue
        group.append(op)
        shape = cast(
            tuple[int, ...],
            _broadcast_shape(shape, cast(MemTensorType, op.dst.type).static_shape),
        )
        written.add(op.dst)
        root = root_buffer(op.dst)
        if root is not None:
            roots.add(root)
    if group:
        groups.append(group)
    return groups


def _swiglu_groups(groups: list[list[Op]], external: set[Value]) -> list[list[Op]]:
    """Fuse only private, complementary BF16 gate/up slices and their epilogue."""
    result: list[list[Op]] = []
    for index, group in enumerate(groups):
        previous = groups[index - 1] if index else []
        hero = previous[0] if len(previous) == 1 else None
        if (
            isinstance(hero, MatmulOp)
            and len(group) == 3
            and all(isinstance(op, ElementwiseOp) for op in group)
        ):
            convert, silu, multiply = cast(list[ElementwiseOp], group)
            shape = cast(MemTensorType, hero.dst.type).static_shape
            if len(shape) != 2:
                result.append(list(group))
                continue
            n = shape[1] // 2
            gate = convert.src(0).definition
            up_value = next(
                (src for src in multiply.operands[1:] if src is not silu.dst), None
            )
            up = up_value.definition if up_value is not None else None
            private = {root_buffer(op.dst) for op in (hero, convert, silu)}
            reads_outside = any(
                effect.kind == "R" and private.intersection(effect.roots)
                for other in groups
                if other is not group and other is not previous
                for op in other
                for effect in memory_effects(op) or []
            )
            if (
                [op.fn for op in (convert, silu, multiply)]
                == ["convert", "silu", "mul"]
                and shape[1] == 2 * n
                and shape[0] % 128 == 0
                and any(n % tile == 0 for tile in (128, 192, 208))
                and cast(MemTensorType, hero.lhs.type).static_shape[1] % 64 == 0
                and not private.intersection(external)
                and not reads_outside
                and isinstance(gate, TileOp)
                and isinstance(up, TileOp)
                and gate.arg is hero.dst
                and up.arg is hero.dst
                and gate.indices == [0, 0]
                and up.indices == [0, 1]
                and cast(MemTensorType, gate.result().type).static_shape
                == (shape[0], n)
                and cast(MemTensorType, up.result().type).static_shape == (shape[0], n)
                and isinstance(cast(MemTensorType, hero.dst.type).dtype, BF16Type)
                and isinstance(cast(MemTensorType, convert.dst.type).dtype, F32Type)
                and isinstance(cast(MemTensorType, silu.dst.type).dtype, BF16Type)
                and isinstance(cast(MemTensorType, multiply.dst.type).dtype, BF16Type)
                and silu.src(0) is convert.dst
                and set(multiply.operands[1:]) == {silu.dst, up_value}
                and cast(MemTensorType, multiply.dst.type).static_shape == (shape[0], n)
            ):
                result[-1] = [*previous, *group]
                continue
        result.append(list(group))
    return result


def _map_groups(groups: list[list[Op]]) -> list[list[Op]]:
    """Combine maps across views when every aliased read has a complete writer."""
    result: list[list[Op]] = []
    written: set[Value] = set()
    roots: set[Value | None] = set()
    for group in groups:
        if not all(isinstance(op, (ElementwiseOp, MoveOp, GatherOp)) for op in group):
            result.append(group)
            written, roots = set(), set()
            continue
        safe = True
        local_written, local_roots = set(written), set(roots)
        for op in group:
            assert isinstance(op, (ElementwiseOp, MoveOp, GatherOp))
            for src in op.operands[1:]:
                if root_buffer(src) not in local_roots:
                    continue
                view = src
                while view not in local_written and isinstance(
                    view.definition, (TileOp, ReshapeOp)
                ):
                    view = view.definition.arg
                if view not in local_written:
                    safe = False
            local_written.add(op.dst)
            local_roots.add(root_buffer(op.dst))
        if written and safe:
            result[-1] = [*result[-1], *group]
        else:
            result.append(group)
            maps = cast(list[ElementwiseOp | MoveOp | GatherOp], group)
            local_written = {op.dst for op in maps}
            local_roots = {root_buffer(op.dst) for op in maps}
        written, roots = local_written, local_roots
    return result


def _row_epilogue_groups(groups: list[list[Op]]) -> list[list[Op]]:
    """Keep head-local maps across reshape/half-head views in row registers.

    Restrict fusion to two 64-column halves of a 128-column head. Each lane
    already owns the same columns of both halves, so no shuffle or inter-CTA
    dependency is introduced. Independent gathers retain their own phase.
    """
    result: list[list[Op]] = []
    index = 0
    while index < len(groups):
        group = groups[index]
        reductions = [op for op in group if isinstance(op, ReduceOp)]
        if len(reductions) != 1 or index + 1 == len(groups):
            result.append(group)
            index += 1
            continue
        shape = cast(MemTensorType, reductions[0].src.type).static_shape
        following = groups[index + 1]
        if len(shape) != 2 or shape[1] != 128 or not all(
            isinstance(op, (ElementwiseOp, MoveOp, GatherOp)) for op in following
        ):
            result.append(group)
            index += 1
            continue
        gathers = [op for op in following if isinstance(op, GatherOp)]
        written = {
            cast(ElementwiseOp | MoveOp | ReduceOp, op).dst for op in group
            if not (isinstance(op, MoveOp) and op.attrs.get("inline_row_copy") is True)
        }
        row_written = set(written)
        roots = {root_buffer(value) for value in written}
        fused: list[Op] = []
        remaining: list[Op] = []
        for op in following:
            if isinstance(op, GatherOp):
                continue
            assert isinstance(op, (ElementwiseOp, MoveOp))
            if any(root_buffer(src) in roots for src in op.operands[1:]):
                fused.append(op)
                written.add(op.dst)
                roots.add(root_buffer(op.dst))
            else:
                remaining.append(op)
        heads = {
            typ.static_shape[1] for op in fused
            if (typ := cast(MemTensorType, cast(ElementwiseOp | MoveOp, op).dst.type)).rank == 3
            and typ.static_shape[0] * typ.static_shape[1] == shape[0]
        }
        safe = (
            bool(fused) and len(heads) == 1
            and all(can_reorder(op, gather) for op in group for gather in gathers)
            and all(can_reorder(earlier, gather)
                    for position, gather in enumerate(following) if isinstance(gather, GatherOp)
                    for earlier in following[:position] if not isinstance(earlier, GatherOp))
            and all(can_reorder(other, op) for other in remaining for op in fused)
        )
        if safe:
            [head_count] = heads
            known = set(row_written)
            known_roots = {root_buffer(value) for value in known}
            for op in fused:
                for value in op.operands:
                    typ = cast(MemTensorType, value.type)
                    dims = typ.static_shape
                    safe &= (
                        typ.rank == 0
                        or typ.rank == 2 and dims in ((shape[0] // head_count, head_count * 128), shape)
                        or typ.rank == 3 and dims[0] * head_count == shape[0]
                        and dims[1] in (1, head_count) and dims[2] in (64, 128)
                    )
                    # An aliased in-region read must resolve to a complete
                    # writer; a containing buffer with partial writers does
                    # not have a single register value to read through.
                    parent = value
                    while isinstance(parent.definition, (TileOp, ReshapeOp)):
                        view = parent.definition
                        if isinstance(view, TileOp):
                            safe &= all(isinstance(at, int) for at in view.indices) and not any(view.indices[:-1])
                            safe &= view.indices[-1] == 0 or (
                                view.indices[-1] == 1
                                and cast(MemTensorType, parent.type).static_shape[-1] == 64
                            )
                        parent = view.arg
                    if value is op.operands[0] or root_buffer(value) not in known_roots:
                        continue
                    parent = value
                    while parent not in known and isinstance(parent.definition, (TileOp, ReshapeOp)):
                        parent = parent.definition.arg
                    safe &= parent in known
                known.add(cast(ElementwiseOp | MoveOp, op).dst)
                known_roots.add(root_buffer(cast(ElementwiseOp | MoveOp, op).dst))
        if not safe:
            result.append(group)
            index += 1
            continue
        if gathers:
            result.append(list(gathers))
        for op in fused:
            op.attrs["row_map_epilogue"] = True
        result.append([*group, *fused])
        if remaining:
            result.append(remaining)
        index += 2
    return result


def _inline_gemm_epilogue(
    kernel: KernelOp, structural_count: int, effects: list[Op]
) -> MatmulOp:
    """Version private maps in registers and retain the GEMM output transport."""
    destination = cast(ElementwiseOp, effects[-1]).dst
    original_hero = cast(MatmulOp, effects[0])
    hero = cast(MatmulOp, clone_op(original_hero, {original_hero.dst: destination}))
    rewritten = list(kernel.body.ops[:structural_count])
    rewritten.append(hero)
    rounded = AllocOp.build_with_fresh_id(
        clone_type(original_hero.dst.type), attrs={"label": "forward_epi"}
    )
    rewritten.append(rounded)
    rewritten.append(
        ElementwiseOp.build(
            rounded.result(), [hero.dst], "convert", attrs={"label": "forward_epi"}
        )
    )
    remap: dict[Value, Value] = {original_hero.dst: rounded.result()}
    for effect in cast(list[ElementwiseOp], effects[1:]):
        allocation = AllocOp.build_with_fresh_id(
            clone_type(effect.dst.type), attrs={"label": "forward_epi"}
        )
        rewritten.append(allocation)
        remap[effect.dst] = allocation.result()
        effect = cast(ElementwiseOp, clone_op(effect, remap))
        effect.attrs["label"] = "forward_epi"
        if (
            effect.fn in ("sigmoid", "silu")
            and cast(MemTensorType, effect.src(0).type).dtype != F32Type()
        ):
            widened = AllocOp.build_with_fresh_id(
                clone_type(effect.src(0).type, {"dtype": F32Type()}),
                attrs={"label": "forward_epi"},
            )
            rewritten.append(widened)
            rewritten.append(
                ElementwiseOp.build(
                    widened.result(),
                    [effect.src(0)],
                    "convert",
                    attrs={"label": "forward_epi"},
                )
            )
            effect.operands[1] = widened.result()
        rewritten.append(effect)
    rewritten.append(
        MoveOp.build(
            destination,
            remap[destination],
            collective=hero.collective,
            attrs={"label": "forward_epi"},
        )
    )
    kernel.body.ops = rewritten
    return hero


def _gemm_map_groups(groups: list[list[Op]], external: set[Value],
                     rows_per_cta: int = 128) -> list[list[Op]]:
    """Keep private, same-shape pointwise consumers in GEMM registers."""
    result: list[list[Op]] = []
    for index, group in enumerate(groups):
        previous = result[-1] if result else []
        if (
            len(previous) != 1
            or not isinstance(previous[0], MatmulOp)
            or not all(isinstance(op, ElementwiseOp) for op in group)
        ):
            result.append(group)
            continue
        hero = previous[0]
        region = [hero, *group]
        internal = {cast(MatmulOp | ElementwiseOp, op).dst for op in region[:-1]}
        readers = {
            root
            for other in groups
            for op in other
            if op not in region
            for effect in memory_effects(op) or []
            if effect.kind == "R"
            for root in effect.roots
        }
        shape = cast(MemTensorType, hero.dst.type).static_shape
        safe = (
            not internal & (external | readers)
            # Packed BF16 fragment input loads require complete CTA rows.
            # Keep partial rows in the guarded scalar map schedule.
            and shape[0] % rows_per_cta == 0
            and all(root_buffer(dst) is dst for dst in internal)
            and all(
                cast(MemTensorType, cast(ElementwiseOp, op).dst.type).static_shape
                == shape
                for op in group
            )
            and all(
                cast(ElementwiseOp, op).fn
                in ("convert", "fill", "sigmoid", "silu", "mul")
                for op in group
            )
            and all(
                cast(MemTensorType, src.type).rank == 0
                or (
                    cast(MemTensorType, src.type).rank == 2
                    and all(
                        s in (1, d)
                        for s, d in zip(
                            cast(MemTensorType, src.type).static_shape,
                            shape,
                            strict=True,
                        )
                    )
                )
                for op in group
                for src in op.operands[1:]
            )
            and any(hero.dst in op.operands[1:] for op in group)
            and all(
                not any(
                    root_buffer(src) in internal and src not in internal
                    for src in op.operands[1:]
                )
                for op in group
            )
            and root_buffer(cast(ElementwiseOp, group[-1]).dst)
            not in {
                root
                for op in region
                for effect in memory_effects(op) or []
                if effect.kind == "R"
                for root in effect.roots
            }
        )
        if safe:
            hero.attrs["fused_map_epilogue"] = True
            result[-1] = region
        else:
            result.append(group)
    return result


def _row_copy_groups(groups: list[list[Op]], external: set[Value]) -> list[list[Op]]:
    """Read private materialization inputs directly in the following row region."""
    result: list[list[Op]] = []
    for index, group in enumerate(groups):
        previous = groups[index - 1] if index else []
        copies: list[MoveOp] = []
        if (
            previous
            and all(
                isinstance(op, (ElementwiseOp, MoveOp, GatherOp)) for op in previous
            )
            and any(isinstance(op, ReduceOp) for op in group)
        ):
            reads = {
                root
                for op in group
                for effect in memory_effects(op) or []
                if effect.kind == "R"
                for root in effect.roots
            }
            writes = {
                root
                for op in group
                for effect in memory_effects(op) or []
                if effect.kind == "W"
                for root in effect.roots
            }
            for op in previous:
                if not isinstance(op, MoveOp):
                    continue
                root = root_buffer(op.dst)
                used_elsewhere = any(
                    effect.kind == "R" and root in effect.roots
                    for other in groups
                    if other is not previous and other is not group
                    for other_op in other
                    for effect in memory_effects(other_op) or []
                )
                compatible = (
                    isinstance(op.src.type, GmemTensorType)
                    and isinstance(op.dst.type, GmemTensorType)
                    and op.dst is root
                    and root in reads
                    and root not in external
                    and op.src.type.static_shape == op.dst.type.static_shape
                    and op.src.type.dtype == op.dst.type.dtype
                    and root_buffer(op.src) not in writes | {root}
                    and all(
                        can_reorder(op, other) for other in previous if other is not op
                    )
                )
                if compatible and not used_elsewhere:
                    copies.append(op)
        if copies:
            result[-1] = [op for op in result[-1] if op not in copies]
            if not result[-1]:
                result.pop()
            for op in copies:
                op.attrs["inline_row_copy"] = True
            result.append([*copies, *group])
        else:
            result.append(list(group))
    return result
