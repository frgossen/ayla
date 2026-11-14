"""Import matching, scheduling, and compilation for packed SwiGLU."""

from __future__ import annotations

from typing import cast

from ..base.ir import Block, Op, Value, clone_type, clone_op, find_ops_of_type
from ..fx_compiler import (
    FxRegionMatch,
    fx_region_to_module,
)
from ..jit_compiler import PtxRoutePlan, ScratchInit
from .gemm import (
    schedule_register_matmul,
    schedule_tiled_matmul,
)
from ..gpu_dialect import (
    AllocOp,
    AsyncOp,
    AwaitOp,
    root_buffers,
    join_collectives,
    BF16Type,
    BarrierOp,
    BlockIdOp,
    WarpGroupIdOp,
    ForOp,
    IndexConstantOp,
    IndexAddOp,
    IndexModOp,
    IndexDivOp,
    IndexLtOp,
    IfOp,
    MemoryEffect,
    ConstantOp,
    RmemTensorType,
    CollectiveAxis,
    CollectiveLevel,
    ElementwiseOp,
    F32Type,
    GmemTensorType,
    KernelOp,
    LockType,
    MatmulOp,
    ModuleOp,
    MoveOp,
    TileOp,
    ReshapeOp,
    ReinterpretOp,
    WgrmemTensorType,
    SmemTensorType,
    effective_operands,
    is_side_effect_free,
)
from ..pipelines.canonicalize import Canonicalize
from ..passes.fuse_async import FuseAsync
from ..passes.lower_to_async import LowerToAsync
from ..passes.promote_buffers import PromoteBuffers
from ..passes.schedule_async import ScheduleAsync
from ..passes.tile_matmul import TileMatmul
from ..ptx_codegen import make_ptx_kernel


def _gmem(value: Value) -> GmemTensorType:
    if not isinstance(value.type, GmemTensorType):
        raise ValueError(f"expected gmem value, got {value.type}")
    return value.type


def _static_pair(value: Value) -> tuple[int, int]:
    typ = _gmem(value)
    if typ.rank != 2 or any(size is None for size in typ.sizes):
        raise ValueError(f"expected a static rank-2 tensor, got {typ}")
    sizes = cast(tuple[int, int], typ.sizes)
    return sizes[0], sizes[1]


def _match_swiglu_chain(
    module: ModuleOp,
) -> tuple[KernelOp, MatmulOp, MatmulOp]:
    top_ops = module.body.ops
    if len(top_ops) != 1 or not isinstance(top_ops[0], KernelOp):
        raise ValueError("SwiGLU schedule requires one kernel")
    kernel = top_ops[0]
    if len(kernel.body.arguments) != 4:
        raise ValueError("SwiGLU schedule requires three inputs and one result")
    lhs, packed_weight, down_weight, output = kernel.body.arguments
    matmuls = find_ops_of_type(module, MatmulOp)
    tiles = find_ops_of_type(module, TileOp)
    elementwise = find_ops_of_type(module, ElementwiseOp)
    if len(matmuls) != 2 or len(tiles) != 2:
        raise ValueError("expected two matmuls and two projection slices")
    if [op.fn for op in elementwise] != ["convert", "silu", "mul"]:
        raise ValueError("expected convert, SiLU, and multiply epilogue")
    projection, down = matmuls
    gate_convert, silu, multiply = elementwise
    gate = next((tile for tile in tiles if tile.indices[1] == 0), None)
    up = next((tile for tile in tiles if tile is not gate), None)
    if gate is None or up is None:
        raise ValueError("expected gate and up projection slices")
    if not (
        projection.lhs is lhs
        and projection.rhs is packed_weight
        and gate.arg is projection.dst
        and up.arg is projection.dst
        and up.indices[1] == 1
        and gate_convert.src(0) is gate.result()
        and silu.src(0) is gate_convert.dst
        and set((multiply.src(0), multiply.src(1))) == {silu.dst, up.result()}
        and down.lhs is multiply.dst
        and down.rhs is down_weight
        and down.dst is output
    ):
        raise ValueError("module is not a packed SwiGLU GEMM chain")
    return kernel, projection, down


def is_swiglu_region(match: FxRegionMatch) -> bool:
    """Whether an FX region imports as the packed SwiGLU GEMM chain."""
    try:
        module = fx_region_to_module(match, name="swiglu_candidate")
        _match_swiglu_chain(module)
    except ValueError:
        return False
    return True


def _projection_ops(
    lhs: Value,
    packed_weight: Value,
    hidden: Value,
    *,
    m: int,
    k: int,
    intermediate: int,
    m_tile: int,
    n_tile: int,
) -> tuple[list[Op], MoveOp]:
    bx = BlockIdOp.build(CollectiveAxis.X)
    by = BlockIdOp.build(CollectiveAxis.Y)
    zero = ConstantOp.build(0.0, RmemTensorType.build([], F32Type()))
    packed_type = _gmem(packed_weight)
    lhs_type = _gmem(lhs)
    hidden_type = _gmem(hidden)
    gate_weight = TileOp.build(
        packed_weight,
        [0, 0],
        GmemTensorType.build(
            [intermediate, k],
            packed_type.dtype,
            strides=list(packed_type.strides),
        ),
    )
    up_weight = TileOp.build(
        packed_weight,
        [1, 0],
        GmemTensorType.build(
            [intermediate, k],
            packed_type.dtype,
            strides=list(packed_type.strides),
        ),
    )
    lhs_tile = TileOp.build(
        lhs,
        [by.value, 0],
        GmemTensorType.build(
            [m_tile, k],
            lhs_type.dtype,
            strides=list(lhs_type.strides),
        ),
        maybe_partial=m % m_tile != 0,
    )
    gate_tile = TileOp.build(
        gate_weight.result(),
        [bx.value, 0],
        GmemTensorType.build(
            [n_tile, k],
            packed_type.dtype,
            strides=list(packed_type.strides),
        ),
    )
    up_tile = TileOp.build(
        up_weight.result(),
        [bx.value, 0],
        GmemTensorType.build(
            [n_tile, k],
            packed_type.dtype,
            strides=list(packed_type.strides),
        ),
    )
    hidden_tile = TileOp.build(
        hidden,
        [by.value, bx.value],
        GmemTensorType.build(
            [m_tile, n_tile],
            hidden_type.dtype,
            strides=list(hidden_type.strides),
        ),
        maybe_partial=m % m_tile != 0,
    )

    accumulator_type = WgrmemTensorType.build(
        [m_tile, n_tile],
        F32Type(),
        warps=4,
        layout="wgmma_layout_0",
    )
    epilogue_type = WgrmemTensorType.build(
        [m_tile, n_tile],
        _gmem(hidden).dtype,
        warps=4,
        layout="wgmma_layout_0",
    )
    gate_acc = AllocOp.build_with_fresh_id(accumulator_type)
    up_acc = AllocOp.build_with_fresh_id(accumulator_type)
    activation = AllocOp.build_with_fresh_id(epilogue_type)
    product = AllocOp.build_with_fresh_id(epilogue_type)
    collective = {
        CollectiveAxis.X: CollectiveLevel.WG,
        CollectiveAxis.Y: CollectiveLevel.GRID,
        CollectiveAxis.Z: CollectiveLevel.GRID,
    }
    gate_matmul = MatmulOp.build(
        lhs_tile.result(),
        gate_tile.result(),
        gate_acc.result(),
        collective=collective,
        attrs={"label": "gate"},
    )
    up_matmul = MatmulOp.build(
        lhs_tile.result(),
        up_tile.result(),
        up_acc.result(),
        collective=collective,
        attrs={"label": "up"},
    )
    round_gate = ElementwiseOp.build(
        activation.result(),
        [gate_acc.result()],
        "convert",
    )
    widen_gate = ElementwiseOp.build(
        gate_acc.result(),
        [activation.result()],
        "convert",
    )
    activate = ElementwiseOp.build(
        activation.result(),
        [gate_acc.result()],
        "silu",
        attrs={"label": "gate_epi"},
    )
    round_up = ElementwiseOp.build(
        product.result(),
        [up_acc.result()],
        "convert",
    )
    multiply = ElementwiseOp.build(
        product.result(),
        [product.result(), activation.result()],
        "mul",
        attrs={"label": "up_epi"},
    )
    writeback = MoveOp.build(
        hidden_tile.result(),
        product.result(),
        collective=collective,
        attrs={"label": "up_dst_wrmem"},
    )
    return [
        bx,
        by,
        zero,
        gate_weight,
        up_weight,
        lhs_tile,
        gate_tile,
        up_tile,
        hidden_tile,
        gate_acc,
        up_acc,
        activation,
        product,
        ElementwiseOp.build(gate_acc.result(), [zero.result()], "fill"),
        gate_matmul,
        round_gate,
        widen_gate,
        activate,
        # Gate and up have distinct logical SMEM allocations that are overlaid
        # later by ReuseSmemAllocs. Keep ScheduleAsync from hoisting up's TMA
        # loads above gate completion and clobbering that shared arena. Thread
        # scope is only an IR ordering fence here and emits no PTX instruction.
        BarrierOp.build("thread"),
        ElementwiseOp.build(up_acc.result(), [zero.result()], "fill"),
        up_matmul,
        round_up,
        multiply,
        writeback,
    ], writeback


def _schedule_up_tail_continuations(
    ops: list[Op], issues: list[AsyncOp], *,
    continuation_mmas: int | None, epilogue_n_tile: int | None,
) -> list[Op]:
    # Keep some up MMAs after gate activation in the IR. With a single up
    # commit, ptxas can move its final drain into the activation to consume
    # and free up's registers early. A continuation lets it interleave
    # those MMAs with the activation instead. The accumulator dependency
    # is carried by WGMMA across commits; no scalar reads of up occur until
    # all up commits complete. Zero retains the original schedule for tuning.
    for issue in issues:
        if not all(str(inner.attrs.get("label", "")).startswith("up")
                   for inner in issue.body.ops):
            continue
        body = list(issue.body.ops)
        # Leave at most one K64 tile in the first commit. Larger peeled
        # tails supply more continuation MMAs for ptxas to interleave.
        reserve = (max(len(body) // 2, len(body) - 4)
                   if continuation_mmas is None else continuation_mmas)
        if reserve == 0:
            continue
        if reserve >= len(body):
            raise ValueError("up tail continuation must leave an initial MMA")
        wait = next(op for op in ops if isinstance(op, AwaitOp) and op.lock is issue.lock)
        if epilogue_n_tile is not None:
            # Static register views preserve the original BF16 round-trip.
            # Issue up continuations before successive gate fragments; up
            # registers are read by scalar code only after every commit.
            activate = next(op for op in ops if isinstance(op, ElementwiseOp)
                            and op.attrs.get("label") == "gate_epi")
            width = cast(int, activate.dst.type.sizes[1])
            if width % epilogue_n_tile:
                raise ValueError("tail epilogue tile must divide the register width")
            epi = [op for op in ops[ops.index(issue) + 1:ops.index(wait)]
                   if isinstance(op, ElementwiseOp)]
            if [op.fn for op in epi] != ["convert", "convert", "silu"]:
                raise ValueError("fragmented tail requires the register gate epilogue")
            fragments = width // epilogue_n_tile
            groups = min(reserve, fragments, 7)
            issue.body.ops = body[:-reserve]
            locks = [AllocOp.build_with_fresh_id(LockType.build("wgmma"))
                     for _ in range(groups)]
            interleaved: list[Op] = []
            next_group = 0
            for fragment in range(fragments):
                if next_group < groups and fragment == next_group * fragments // groups:
                    start = len(body) - reserve + next_group * reserve // groups
                    stop = len(body) - reserve + (next_group + 1) * reserve // groups
                    interleaved.append(AsyncOp.build(
                        locks[next_group].result(), collective=issue.collective,
                        body=Block(ops=body[start:stop]),
                        attrs={"label": "up_tail_continuation"}))
                    next_group += 1
                views: dict[Value, Value] = {}
                for op in epi:
                    for value in op.operands:
                        if value not in views:
                            typ = cast(WgrmemTensorType, value.type)
                            view = TileOp.build(value, [0, fragment],
                                WgrmemTensorType.build(
                                    [typ.static_shape[0], epilogue_n_tile], typ.dtype,
                                    warps=typ.warps, layout=typ.layout))
                            interleaved.append(view)
                            views[value] = view.result()
                    interleaved.append(ElementwiseOp.build(
                        views[op.dst], [views[op.src(i)] for i in range(op.num_srcs)],
                        op.fn, attrs=dict(op.attrs)))
            position = ops.index(epi[0])
            ops = [op for op in ops if op not in epi]
            ops[position:position] = interleaved
            wait.attrs["pending"] = groups
            position = ops.index(wait)
            ops[position + 1:position + 1] = [
                AwaitOp.build(lock.result(), attrs={"pending": groups - i - 1})
                for i, lock in enumerate(locks)]
            ops[0:0] = locks
            continue
        issue.body.ops = body[:-reserve]
        lock = AllocOp.build_with_fresh_id(LockType.build("wgmma"))
        continuation = AsyncOp.build(
            lock.result(), collective=issue.collective,
            body=Block(ops=body[-reserve:]),
            attrs={"label": "up_tail_continuation"},
        )
        wait.attrs["pending"] = 1
        position = ops.index(wait)
        ops[position:position] = [continuation]
        ops.insert(position + 2, AwaitOp.build(lock.result(), attrs={"pending": 0}))
        ops.insert(0, lock)
    return ops


def _overlap_projection_tail(
    module: ModuleOp, *, continuation_mmas: int | None = None,
    epilogue_n_tile: int | None = None,
) -> None:
    """Issue the retained-A tail, wait for gate, and leave up in flight."""

    def in_tail(value: Value) -> bool:
        roots = root_buffers(value)
        return bool(roots) and all(
            isinstance(root.definition, AllocOp)
            and root.definition.attrs.get("reuse_group") == "projection_tail_compute"
            for root in roots
        )

    def visit(block: Block) -> None:
        for op in block.ops:
            for nested in op.blocks:
                visit(nested)
        ops = list(block.ops)
        loads = [
            op
            for op in ops
            if isinstance(op, AsyncOp)
            and op.body.ops
            and all(
                isinstance(inner, MoveOp)
                and isinstance(inner.dst.type, SmemTensorType)
                and in_tail(inner.dst)
                for inner in op.body.ops
            )
        ]
        if not loads:
            return
        locks = {op.lock for op in loads}
        last_load = max(ops.index(op) for op in loads)
        first_compute = next(
            (
                index
                for index, op in enumerate(ops)
                if isinstance(op, AsyncOp)
                and any(
                    isinstance(inner, MatmulOp) and in_tail(inner.lhs)
                    for inner in op.body.ops
                )
            ),
            None,
        )
        if first_compute is None or first_compute <= last_load:
            raise ValueError("tail loads must precede the first tail compute")
        # Scheduling has hoisted all immutable tail loads ahead of compute.
        # Fuse their completion before regrouping WGMMA: intermediate TMA waits
        # would otherwise force gate and up to drain each K tile separately.
        collective = loads[0].collective
        for load in loads[1:]:
            collective = join_collectives(collective, load.collective)
        group = AsyncOp.build(
            loads[0].lock,
            collective=collective,
            body=Block(ops=[inner for load in loads for inner in load.body.ops]),
            attrs={"label": "projection_tail_load_async"},
        )
        rebuilt: list[Op] = []
        for index, op in enumerate(ops):
            if index == last_load:
                rebuilt.extend([group, AwaitOp.build(group.lock)])
            if op not in loads and not (isinstance(op, AwaitOp) and op.lock in locks):
                rebuilt.append(op)
        tail_issues = [
            op
            for op in rebuilt
            if isinstance(op, AsyncOp)
            and op.body.ops
            and all(
                isinstance(inner, MatmulOp) and in_tail(inner.lhs)
                for inner in op.body.ops
            )
        ]
        tail_locks = {issue.lock for issue in tail_issues}
        waits = {
            op.lock: op
            for op in rebuilt
            if isinstance(op, AwaitOp) and op.lock in tail_locks
        }
        normalized: list[Op] = []
        for op in rebuilt:
            if isinstance(op, AwaitOp) and op.lock in tail_locks:
                continue
            normalized.append(op)
            if op in tail_issues:
                assert isinstance(op, AsyncOp)
                normalized.append(waits[op.lock])
        block.ops = normalized
        FuseAsync(fuse_wgmma_groups=True).run_block(block)
        ScheduleAsync().run_block(block)
        ops = list(block.ops)
        issues = [
            op
            for op in ops
            if isinstance(op, AsyncOp)
            and op.body.ops
            and all(
                isinstance(inner, MatmulOp) and in_tail(inner.lhs)
                for inner in op.body.ops
            )
        ]
        for issue in issues:
            waits = [
                op for op in ops if isinstance(op, AwaitOp) and op.lock is issue.lock
            ]
            if len(waits) != 1:
                raise ValueError("projection tail issue requires one completion wait")
            wait = waits[0]
            newer = [
                other
                for other in issues
                if ops.index(issue) < ops.index(other) < ops.index(wait)
            ]
            # A WGMMA wait is a hardware queue operation, not a per-lock drain.
            # Match the IR's per-lock completion by retaining newer groups.
            if len(newer) > 7:
                raise ValueError("projection tail exceeds the WGMMA queue depth")
            wait.attrs["pending"] = len(newer)

        block.ops = _schedule_up_tail_continuations(
            ops, issues, continuation_mmas=continuation_mmas,
            epilogue_n_tile=epilogue_n_tile)

    visit(module.body)


def _overlap_projection_ring_tail(
    module: ModuleOp, *, tiles: int, continuation_mmas: int | None,
    epilogue_n_tile: int | None,
) -> None:
    """Regroup the already-loaded pipeline drain without rebuilding its loop.

    The final ring-depth K tiles have distinct retained slots. All their loads
    must precede the first selected issue; only their completion waits may
    remain inside the drain. No allocation or operand address is changed.
    """
    matched = 0

    def visit(block: Block) -> None:
        nonlocal matched
        for op in list(block.ops):
            for nested in op.blocks:
                visit(nested)
        ops = list(block.ops)
        epi = next((op for op in ops if isinstance(op, ElementwiseOp)
                    and op.attrs.get("label") == "gate_epi"), None)
        if epi is None:
            return
        selected: dict[str, list[AsyncOp]] = {}
        for label in ("gate", "up"):
            candidates = [op for op in ops[:ops.index(epi)]
                          if isinstance(op, AsyncOp) and op.body.ops
                          and all(isinstance(mm, MatmulOp)
                                  and str(mm.attrs.get("label", "")).startswith(label)
                                  for mm in op.body.ops)]
            if len(candidates) < tiles:
                raise ValueError("operand ring drain has too few straight-line K tiles")
            selected[label] = candidates[-tiles:]
        issues = selected["gate"] + selected["up"]
        first, last = min(map(ops.index, issues)), max(map(ops.index, issues))
        locks = {op.lock for op in issues}
        for op in ops[first:last + 1]:
            if op in issues or isinstance(op, AwaitOp) or is_side_effect_free(op):
                continue
            raise ValueError("operand ring tail contains a refill or other effect")
        # Require distinct operand views from the retained pipeline stages.
        # The no-refill check above preserves their original read lifetimes.
        for label, group in selected.items():
            slots = [cast(MatmulOp, op.body.ops[0]).lhs for op in group]
            if len(set(slots)) != tiles:
                raise ValueError(f"{label} ring tail must retain distinct operand slots")
        new_locks = [AllocOp.build_with_fresh_id(LockType.build("wgmma"))
                     for _ in range(2)]
        regrouped = [AsyncOp.build(
            lock.result(), collective=group[0].collective,
            body=Block(ops=[mm for issue in group for mm in issue.body.ops]),
            attrs={"label": f"{label}_ring_tail_async"})
            for (label, group), lock in zip(selected.items(), new_locks, strict=True)]
        # Hoist only completion of copies already issued into retained slots.
        completions = [op for op in ops[first:last + 1]
                       if isinstance(op, AwaitOp) and op.lock not in locks]
        pure = [op for op in ops[first:last + 1] if is_side_effect_free(op)]
        rebuilt = []
        for i, op in enumerate(ops):
            if i == first:
                rebuilt.extend([*pure, *completions, *regrouped,
                                AwaitOp.build(new_locks[0].result(), attrs={"pending": 1})])
            if first <= i <= last or (isinstance(op, AwaitOp) and op.lock in locks and i >= first):
                continue
            rebuilt.append(op)
            if op is epi:
                rebuilt.append(AwaitOp.build(new_locks[1].result(), attrs={"pending": 0}))
        rebuilt[0:0] = new_locks
        block.ops = _schedule_up_tail_continuations(
            rebuilt, regrouped, continuation_mmas=continuation_mmas,
            epilogue_n_tile=epilogue_n_tile)
        matched += 1

    visit(module.body)
    if matched != 1:
        raise ValueError("operand ring tail requires exactly one register gate epilogue")


def _schedule_projection(
    module: ModuleOp,
    *,
    operand_stages: int = 3,
    operand_stage_overrides: dict[str, int] | None = None,
    operand_k_tile: int = 64,
    cyclic_operand_pipeline: bool = False,
    operand_move_attrs: dict[str, dict[str, object]] | None = None,
    output_layout: dict[str, object] | None = None,
    k_tail_tiles: int | None = None,
    tail_continuation_mmas: int | None = None,
    tail_epilogue_n_tile: int | None = None,
    tail_from_operand_ring: bool = False,
    rhs_tma_load_slices: int = 1,
) -> None:
    move_attrs: dict[str, dict[str, object]] = {
        "lhs": {
            "l2_cache_hint": "evict_last",
            "l2_promotion": "128B",
        },
        "rhs": {
            "l2_cache_hint": "evict_first",
            "l2_promotion": "256B",
        },
    }
    if operand_move_attrs is not None:
        move_attrs = operand_move_attrs
    labels = ("gate", "up") if k_tail_tiles is None else ("gate|up",)
    for label in labels:
        overrides = {name: depth for name, depth in (operand_stage_overrides or {}).items()
                     if label == "gate|up" or name.startswith(f"{label}_")}
        schedule_register_matmul(
            label=label,
            operand_stages=operand_stages,
            operand_stage_overrides=overrides,
            operand_k_tile=operand_k_tile,
            rhs_tma_load_slices=rhs_tma_load_slices,
            cyclic_operand_pipeline=cyclic_operand_pipeline,
            unsafe_reuse_smem=True,
            operand_move_attrs=move_attrs,
            loop_label_prefix=label if k_tail_tiles is None else "projection",
            k_tail_tiles=0 if tail_from_operand_ring else k_tail_tiles,
        ).run(module)
    if k_tail_tiles is not None:
        # Gate/up execute together. Pack concurrent operands together; only
        # the drained common loop and its peeled tail may share their arena.
        for allocation in find_ops_of_type(module, AllocOp):
            group = allocation.attrs.get("reuse_group")
            if (
                isinstance(group, str)
                and group.startswith(("gate", "up"))
                and group.endswith("_compute")
            ):
                allocation.attrs["reuse_group"] = (
                    "projection_tail_compute" if allocation.attrs.get("peeled") is True
                    else "projection_compute"
                )
        FuseAsync(fuse_wgmma_groups=True).run(module)

    PromoteBuffers(
        target_type="wgrmem",
        operands=["dst"],
        physical_layout=output_layout,
        tma=True,
        unsafe_reuse_smem=True,
        move_attrs={"dst": {"l2_cache_hint": "evict_last"}},
        label_filter="up_dst_wrmem",
    ).run(module)
    LowerToAsync().run(module)
    FuseAsync().run(module)
    ScheduleAsync().run(module)
    Canonicalize().run(module)
    if k_tail_tiles:
        if tail_from_operand_ring:
            _overlap_projection_ring_tail(
                module, tiles=k_tail_tiles, continuation_mmas=tail_continuation_mmas,
                epilogue_n_tile=tail_epilogue_n_tile)
        else:
            _overlap_projection_tail(module, continuation_mmas=tail_continuation_mmas,
                                     epilogue_n_tile=tail_epilogue_n_tile)
        Canonicalize().run(module)


def _prefetch_projection_output(module: ModuleOp, tiles: int) -> None:
    """Load the next output's first K tiles while activating this output.

    The three-buffer operand ring stays unchanged. Writeback occupies the
    retired gate buffers (one-tile lookahead), or gate/up buffer 2 (two tiles).
    Its existing completion wait retires those reads before the next output
    reloads them. A CTA barrier retires WGMMA reads before issuing lookahead.
    Only complete resident work waves enter this transformation.
    """
    kernel = cast(KernelOp, module.body.ops[0])
    [loop] = [op for op in kernel.body.ops if isinstance(op, ForOp)]
    if any(isinstance(op, IfOp) for op in loop.body.ops):
        raise ValueError("output prefetch requires complete resident work waves")
    gate = [op for op in kernel.body.ops if isinstance(op, AllocOp)
            and op.attrs.get("label") == "gate_rhs_smem"]
    up = [op for op in kernel.body.ops if isinstance(op, AllocOp)
          and op.attrs.get("label") == "up_rhs_smem"]
    if len(gate) != 3 or len(up) != 3:
        raise ValueError("output prefetch requires three gate/up buffers")
    # ReuseSmemAllocs places a group's allocations in IR order. In the
    # two-tile case the contiguous gate-2/up-2 range holds the entire output.
    ordered = gate if tiles == 1 else [*gate, up[2], *up[:2]]
    for allocation in ordered:
        kernel.body.ops.remove(allocation)
    kernel.body.ops[0:0] = ordered
    writeback = next(op for op in loop.body.ops if isinstance(op, AllocOp)
                     and isinstance(op.result().type, SmemTensorType))
    typ = cast(SmemTensorType, writeback.result().type)
    n = typ.static_shape[1]
    stage_bytes = n * 64 * cast(BF16Type, typ.dtype).byte_size
    storage = AllocOp.build_with_fresh_id(
        SmemTensorType.build([(tiles + 2) * n * 64], typ.dtype)
    )
    storage.attrs["reuse_group"] = writeback.attrs["reuse_group"]
    view = ReinterpretOp.build(storage.result(), typ, byte_offset=tiles * stage_bytes)
    for user in list(writeback.result().users()):
        for index, value in enumerate(user.operands):
            if value is writeback.result():
                user.operands[index] = view.result()
    position = loop.body.ops.index(writeback)
    loop.body.ops[position:position + 1] = [storage, view]

    ops = list(loop.body.ops)
    first_compute = next(i for i, op in enumerate(ops) if isinstance(op, AsyncOp)
                         and cast(LockType, op.lock.type).kind == "wgmma")
    loads = [op for op in ops[:first_compute] if isinstance(op, AsyncOp)
             and (str(op.attrs.get("label", "")).endswith("_smem_front0_async")
                  or (tiles == 2 and op.attrs.get("label") in
                      ("up_rhs_smem_iter0_async", "gate_lhs_smem_iter0_async")))]
    if len(loads) != 2 * tiles:
        raise ValueError("output prefetch could not identify initial operand loads")
    # Duplicate address calculations only. Locks and shared buffers remain
    # outside the loop, so their parity and contents carry across outputs.
    definitions = {value: op for op in ops for value in op.results}
    dependencies: set[Op] = set()

    def require(value: Value) -> None:
        defining = definitions.get(value)
        if defining is None or defining in dependencies:
            return
        if not is_side_effect_free(defining):
            raise ValueError("output prefetch address recipes must be pure")
        dependencies.add(defining)
        for operand in effective_operands(defining):
            require(operand)

    for load in loads:
        for operand in effective_operands(load):
            require(operand)
    recipes = [op for op in ops if op in dependencies]
    first_limit = IndexConstantOp.build(loop.lb + loop.step)
    first = IndexLtOp.build(loop.iv, first_limit.result())
    rebuilt: list[Op] = [first_limit, first]
    for op in ops:
        rebuilt.append(IfOp.build(first.result(), Block(ops=[op])) if op in loads else op)
    epilogue = next(i for i, op in enumerate(rebuilt) if isinstance(op, ElementwiseOp))
    waits = [op for op in rebuilt if isinstance(op, AwaitOp)
             and cast(LockType, op.lock.type).kind == "wgmma"]
    # Complete up before the cross-warpgroup reuse barrier. No next-output
    # accumulators are introduced; gate/up arithmetic and rounding stay intact.
    last_wait = waits[-1]
    if rebuilt.index(last_wait) > epilogue:
        rebuilt.remove(last_wait)
        rebuilt.insert(epilogue, last_wait)
        epilogue += 1
    barrier = next(op for op in rebuilt[epilogue:] if isinstance(op, BarrierOp)
                   and op.scope == "block" and not op.attrs.get("drain_bulk"))
    rebuilt.remove(barrier)
    step = IndexConstantOp.build(loop.step)
    ahead = IndexAddOp.build(loop.iv, step.result())
    limit = IndexConstantOp.build(loop.ub)
    valid = IndexLtOp.build(ahead.result(), limit.result())
    mapping = {loop.iv: ahead.result()}
    future = [clone_op(op, mapping) for op in [*recipes, *loads]]
    for op in future:
        if isinstance(op, AsyncOp):
            op.attrs["label"] = f"output_prefetch_{op.attrs.get('label')}"
    rebuilt[epilogue:epilogue] = [barrier, step, ahead, limit, valid,
                                IfOp.build(valid.result(), Block(ops=future))]
    loop.body.ops = rebuilt


def schedule_swiglu_projection(
    module: ModuleOp,
    lhs: Value,
    packed_weight: Value,
    hidden: Value,
    *,
    blocks: int,
    threads: int = 256,
    warpgroup_m_tile: int = 64,
    warpgroup_n_tile: int | None = None,
    stage_gate: bool = False,
    gate_padding: int = 8,
    block_store: bool = False,
    k_tail_tiles: int | None = None,
    tail_continuation_mmas: int | None = None,
    tail_epilogue_n_tile: int | None = None,
    tail_from_operand_ring: bool = False,
    n_tile: int = 192,
    operand_stages: int = 3,
    operand_stage_overrides: dict[str, int] | None = None,
    operand_k_tile: int = 64,
    cyclic_operand_pipeline: bool = False,
    operand_move_attrs: dict[str, dict[str, object]] | None = None,
    output_tile_range: tuple[int, int] | None = None,
    output_prefetch_tiles: int = 0,
) -> ModuleOp:
    """Lower packed SwiGLU on a resident grid of two to four warpgroups.

    Reuse the decode projection's register arithmetic and operand staging.
    Each warpgroup owns 64 or 128 rows; warpgroups divide CTA rows and columns.
    CTAs visit tiles in M-fast order. Gate/up operands and writeback retain their
    disjoint lifetime groups so the existing shared-memory arena can overlay
    their storage. Optional gate staging reduces register pressure; padded
    rows spread bank accesses, and block-major writeback uses STMatrix.
    Peeled tails defer up MMAs after an initial commit of at most four K16s;
    ``tail_continuation_mmas`` overrides that count (zero disables it).
    ``tail_epilogue_n_tile`` interleaves up commits with static column
    fragments of gate conversion and activation; the default retains one epilogue.
    ``tail_from_operand_ring`` regroups already-loaded final K tiles after
    pipelining, preserving the common reduction loop and its operand storage.
    """
    [kernel] = module.body.ops
    assert isinstance(kernel, KernelOp)
    if tail_continuation_mmas is not None and (
        type(tail_continuation_mmas) is not int or tail_continuation_mmas < 0
        or (tail_continuation_mmas > 0 and not k_tail_tiles)
    ):
        raise ValueError("tail continuation requires a nonnegative MMA count and a peeled tail")
    if tail_epilogue_n_tile is not None and (
        type(tail_epilogue_n_tile) is not int or tail_epilogue_n_tile < 16
        or tail_epilogue_n_tile % 16 or not k_tail_tiles
        or tail_continuation_mmas == 0 or stage_gate
    ):
        raise ValueError("tail epilogue tile requires a peeled register tail and a multiple of 16")
    if type(tail_from_operand_ring) is not bool or (tail_from_operand_ring and (
        type(k_tail_tiles) is not int or k_tail_tiles < 1 or k_tail_tiles > operand_stages
        or operand_stage_overrides or cyclic_operand_pipeline or stage_gate
    )):
        raise ValueError("operand ring tail requires 1..stages tiles and equal noncyclic rings")
    if blocks < 1 or operand_stages < 2:
        raise ValueError(
            "resident SwiGLU requires positive blocks and at least two stages"
        )
    allowed_overrides = {f"{label}_{operand}_smem"
                         for label in ("gate", "up") for operand in ("lhs", "rhs")}
    if set(operand_stage_overrides or {}) - allowed_overrides:
        raise ValueError("SwiGLU stage overrides must name gate/up operand allocations")
    if type(threads) is not int or threads not in (256, 384, 512):
        raise ValueError("resident SwiGLU requires two, three or four warpgroups")
    if type(warpgroup_m_tile) is not int or warpgroup_m_tile not in (64, 128):
        raise ValueError("resident SwiGLU warpgroup rows must be 64 or 128")
    if warpgroup_n_tile is None:
        warpgroup_n_tile = n_tile
    if (
        type(n_tile) is not int
        or n_tile < 1
        or warpgroup_n_tile not in (96, 104, 128, 192, 208, 256)
        or n_tile % warpgroup_n_tile
    ):
        raise ValueError("unsupported resident SwiGLU warpgroup tile")
    n_groups = n_tile // warpgroup_n_tile
    if n_groups < 1 or (threads // 128) % n_groups:
        raise ValueError("warpgroup column tiles must divide the CTA warpgroup count")
    if type(gate_padding) is not int or gate_padding < 0 or gate_padding % 8:
        raise ValueError("gate padding must be a nonnegative multiple of eight")
    cta_m = warpgroup_m_tile * ((threads // 128) // n_groups)
    m, k = _static_pair(lhs)
    rows, intermediate = _static_pair(hidden)
    if rows != m or _static_pair(packed_weight) != (2 * intermediate, k):
        raise ValueError("inconsistent packed projection dimensions")
    if type(operand_k_tile) is not int or operand_k_tile not in (64, 128):
        raise ValueError("resident SwiGLU K tile must be 64 or 128")
    if intermediate % n_tile or k % operand_k_tile:
        raise ValueError("resident SwiGLU projection requires complete tiles")
    tile_count = ((m + cta_m - 1) // cta_m) * (intermediate // n_tile)
    start, stop = output_tile_range or (0, tile_count)
    if type(start) is not int or type(stop) is not int or not 0 <= start < stop <= tile_count:
        raise ValueError("resident SwiGLU output range must lie within the tiled projection")
    if type(output_prefetch_tiles) is not int or output_prefetch_tiles not in (0, 1, 2):
        raise ValueError("output prefetch tiles must be zero, one or two")
    if output_prefetch_tiles and (
        threads != 256 or warpgroup_m_tile != 64 or warpgroup_n_tile != n_tile
        or operand_stages != 3 or operand_k_tile != 64 or cyclic_operand_pipeline
        or operand_stage_overrides
        or k_tail_tiles != 0 or not block_store or stage_gate
        or start % blocks or stop % blocks or k < 256
    ):
        raise ValueError(
            "output prefetch requires complete work waves, two row warpgroups, "
            "three noncyclic K64 stages, a joint loop and block-major writeback"
        )
    if any(
        not isinstance(_gmem(value).dtype, BF16Type)
        for value in (lhs, packed_weight, hidden)
    ):
        raise ValueError("resident SwiGLU projection requires BF16 tensors")
    structural: list[Op] = [
        op for op in kernel.body.ops if isinstance(op, (ConstantOp, TileOp, ReshapeOp))
    ]
    ops, writeback = _projection_ops(
        lhs,
        packed_weight,
        hidden,
        m=m,
        k=k,
        intermediate=intermediate,
        m_tile=warpgroup_m_tile,
        n_tile=warpgroup_n_tile,
    )
    bx, by = ops[:2]
    assert isinstance(bx, BlockIdOp) and isinstance(by, BlockIdOp)
    block = BlockIdOp.build(CollectiveAxis.X)
    loop = ForOp.build(start, stop, blocks, attrs={"tile_output_loop": True, "tile_task_count": stop})
    flat = IndexAddOp.build(loop.iv, block.value)
    extent = IndexConstantOp.build((m + cta_m - 1) // cta_m)
    mi = IndexModOp.build(flat.result(), extent.result())
    ni = IndexDivOp.build(flat.result(), extent.result())
    limit = IndexConstantOp.build(stop)
    condition = IndexLtOp.build(flat.result(), limit.result())
    wg = WarpGroupIdOp.build()
    mapping: dict[Value, Value] = {bx.value: ni.result(), by.value: mi.result()}
    group_columns = IndexConstantOp.build(n_groups)
    group_m = IndexDivOp.build(wg.result(), group_columns.result())
    group_n = IndexModOp.build(wg.result(), group_columns.result())
    body: list[Op] = [wg, group_columns, group_m, group_n]
    for op in ops[2:]:
        if isinstance(op, TileOp) and op.arg in (lhs, hidden):
            typ = _gmem(op.result())
            wide = TileOp.build(
                op.arg,
                [mi.result(), ni.result() if op.arg is hidden else 0],
                clone_type(
                    typ,
                    {
                        "sizes": [
                            cta_m,
                            n_tile if op.arg is hidden else typ.static_shape[1],
                        ]
                    },
                ),
                maybe_partial=m % cta_m != 0,
            )
            narrow = TileOp.build(
                wide.result(),
                [group_m.result(), group_n.result() if op.arg is hidden else 0],
                clone_type(typ),
                maybe_partial=m % cta_m != 0,
            )
            body.extend([wide, narrow])
            mapping[op.result()] = narrow.result()
        elif isinstance(op, TileOp) and op.indices[0] is bx.value:
            typ = _gmem(op.result())
            wide = TileOp.build(
                mapping.get(op.arg, op.arg),
                [ni.result(), 0],
                clone_type(typ, {"sizes": [n_tile, typ.static_shape[1]]}),
            )
            narrow = TileOp.build(wide.result(), [group_n.result(), 0], clone_type(typ))
            body.extend([wide, narrow])
            mapping[op.result()] = narrow.result()
        else:
            cloned = clone_op(op, mapping)
            if op is writeback:
                cloned.attrs["dst_tiling_scheme"] = (
                    MemoryEffect.fresh_tiling_scheme_id()
                )
            body.append(cloned)
    if k_tail_tiles is not None:
        if type(k_tail_tiles) is not int or k_tail_tiles < 0 or k_tail_tiles >= k // operand_k_tile:
            raise ValueError("joint projection tail must leave at least one common K tile")
        if stage_gate:
            raise ValueError("joint projection retains both accumulators; gate staging is separate")
        body = [op for op in body if not isinstance(op, BarrierOp)]
        up_matmul = next(
            op for op in body
            if isinstance(op, MatmulOp) and op.attrs.get("label") == "up"
        )
        up_fill = next(
            op for op in body if isinstance(op, ElementwiseOp)
            and op.dst is up_matmul.dst and op.fn == "fill"
        )
        body.remove(up_fill)
        gate_matmul = next(
            op for op in body
            if isinstance(op, MatmulOp) and op.attrs.get("label") == "gate"
        )
        body.insert(body.index(gate_matmul), up_fill)
    if stage_gate:
        # Keep the BF16 gate out of registers during up accumulation. Its
        # lifetime overlaps up operands, so pack it into the SAME reuse group.
        # Reload after round_up, when the F32 accumulator is dead. Writeback
        # subsequently reuses this arena after the shared gate read completes.
        stash = AllocOp.build_with_fresh_id(
            SmemTensorType.build([cta_m, n_tile + gate_padding], BF16Type())
        )
        stash.attrs["reuse_group"] = "up_compute"
        structural.append(stash)
        view = TileOp.build(
            stash.result(),
            [group_m.result(), group_n.result()],
            SmemTensorType.build(
                [warpgroup_m_tile, warpgroup_n_tile], BF16Type(), strides=[n_tile + gate_padding, 1]
            ),
        )
        body.insert(4, view)
        activation = next(
            op
            for op in body
            if isinstance(op, ElementwiseOp) and op.attrs.get("label") == "gate_epi"
        )
        multiply = next(
            op
            for op in body
            if isinstance(op, ElementwiseOp) and op.attrs.get("label") == "up_epi"
        )
        collective = {
            CollectiveAxis.X: CollectiveLevel.WG,
            CollectiveAxis.Y: CollectiveLevel.GRID,
            CollectiveAxis.Z: CollectiveLevel.GRID,
        }
        body.insert(body.index(activation) + 1, BarrierOp.build("block"))
        body.insert(
            body.index(activation) + 2,
            MoveOp.build(view.result(), activation.dst, collective=collective),
        )
        body.insert(
            body.index(multiply),
            MoveOp.build(activation.dst, view.result(), collective=collective),
        )
    # Distinct reuse groups alias physical SMEM. Every warpgroup must retire
    # operand reads before the first warpgroup writes the next group's arena.
    # Logical memory effects alone cannot infer this unsafe lifetime assertion.
    body = [
        BarrierOp.build("block")
        if isinstance(op, BarrierOp) and op.scope == "thread"
        else op
        for op in body
    ]
    store = next(
        op
        for op in body
        if isinstance(op, MoveOp) and op.attrs.get("label") == "up_dst_wrmem"
    )
    body.insert(body.index(store), BarrierOp.build("block"))
    body.append(BarrierOp.build("block", attrs={"drain_bulk": True}))
    loop.body.ops = [
        flat,
        extent,
        mi,
        ni,
        limit,
        condition,
        IfOp.build(condition.result(), Block(ops=body)),
    ]
    kernel.body.ops = [*structural, block, loop]
    kernel.block_dim = (threads, 1, 1)
    kernel.grid_dim = (blocks, 1, 1)
    kernel.num_warp_groups = threads // 128
    output_layout = None
    if block_store:
        # Each warpgroup must start on a physical block boundary, including
        # column partitions of a wider CTA tile. Eight BF16 columns need no
        # swizzle and use the existing narrow-block STMatrix/TMA transport.
        block_width = next(
            (width for width in (64, 32, 16, 8) if warpgroup_n_tile % width == 0), None
        )
        if block_width is None:
            raise ValueError("block-major writeback requires a multiple-of-eight warpgroup width")
        output_layout = {
            "swizzle": None if block_width == 8 else f"{block_width * 2}B",
            "block_width": block_width,
            "order": "block_major",
        }
    _schedule_projection(
        module,
        operand_stages=operand_stages,
        operand_stage_overrides=operand_stage_overrides,
        operand_k_tile=operand_k_tile,
        cyclic_operand_pipeline=cyclic_operand_pipeline,
        operand_move_attrs=operand_move_attrs,
        output_layout=output_layout,
        rhs_tma_load_slices=n_groups if n_tile > 256 else 1,
        k_tail_tiles=k_tail_tiles,
        tail_continuation_mmas=tail_continuation_mmas,
        tail_epilogue_n_tile=tail_epilogue_n_tile,
        tail_from_operand_ring=tail_from_operand_ring,
    )
    if output_prefetch_tiles and stop - start > blocks:
        _prefetch_projection_output(module, output_prefetch_tiles)
    if k_tail_tiles is not None:
        for barrier in find_ops_of_type(module, BarrierOp):
            if barrier.scope == "thread":
                barrier.attrs["scope"] = "block"
    return module


def schedule_swiglu_projection_wave_tail(
    module: ModuleOp, lhs: Value, packed_weight: Value, hidden: Value,
    *, blocks: int, grid_sync: Value, n_tile: int, tail_n_tile: int,
    operand_stages: int, tail_operand_stages: int,
    threads: int = 256, cyclic_operand_pipeline: bool = False,
    k_tail_tiles: int | None = None, block_store: bool = False,
    tail_continuation_mmas: int | None = None,
    tail_epilogue_n_tile: int | None = None,
    tail_from_operand_ring: bool = False,
    output_prefetch_tiles: int = 0,
    operand_stage_overrides: dict[str, int] | None = None,
    operand_move_attrs: dict[str, dict[str, object]] | None = None,
) -> ModuleOp:
    """Use a narrower output tile for the final resident projection wave.

    The main range ends on a complete row strip. The tail starts at its own
    column boundary and may recompute a small overlap. A cooperative barrier
    orders overlapping output stores and retires the main shared arena.
    """
    kernel = cast(KernelOp, module.body.ops[0])
    if type(output_prefetch_tiles) is not int or output_prefetch_tiles not in (0, 1, 2):
        raise ValueError("output prefetch tiles must be zero, one or two")
    if type(threads) is not int or threads not in (256, 384, 512):
        raise ValueError("resident SwiGLU requires two, three or four warpgroups")
    m, _ = _static_pair(lhs)
    _, intermediate = _static_pair(hidden)
    rows = (m + 64 * (threads // 128) - 1) // (64 * (threads // 128))
    if blocks < 1 or n_tile < 1 or tail_n_tile < 1 or intermediate % n_tile or intermediate % tail_n_tile:
        raise ValueError("wave-tail scheduling requires complete projection columns")
    total = rows * (intermediate // n_tile)
    main_stop = (total // blocks) * blocks
    if main_stop in (0, total):
        return schedule_swiglu_projection(module, lhs, packed_weight, hidden,
                                          n_tile=n_tile, operand_stages=operand_stages,
                                          operand_stage_overrides=operand_stage_overrides,
                                          operand_move_attrs=operand_move_attrs,
                                          blocks=blocks, threads=threads,
                                          cyclic_operand_pipeline=cyclic_operand_pipeline,
                                          k_tail_tiles=k_tail_tiles, block_store=block_store,
                                          tail_continuation_mmas=tail_continuation_mmas,
                                          tail_epilogue_n_tile=tail_epilogue_n_tile,
                                          tail_from_operand_ring=tail_from_operand_ring,
                                          output_prefetch_tiles=output_prefetch_tiles if main_stop == total else 0)
    if main_stop % rows or grid_sync not in kernel.body.arguments:
        raise ValueError("wave-tail scheduling requires a row-aligned boundary and grid sync argument")
    prefix = (main_stop // rows) * n_tile
    tail_start = (prefix // tail_n_tile) * rows
    pieces: list[list[Op]] = []
    for name, start, stop, width, stages in (
        ("main", 0, main_stop, n_tile, operand_stages),
        ("tail", tail_start, rows * (intermediate // tail_n_tile), tail_n_tile, tail_operand_stages),
    ):
        candidate = cast(ModuleOp, clone_op(module, {}))
        cloned = cast(KernelOp, candidate.body.ops[0])
        mapping = dict(zip(kernel.body.arguments, cloned.body.arguments, strict=True))
        for old, new in zip(find_ops_of_type(module, Op), find_ops_of_type(candidate, Op), strict=True):
            mapping.update(zip(old.results, new.results, strict=True))
        schedule_swiglu_projection(candidate, mapping[lhs], mapping[packed_weight], mapping[hidden],
                                   n_tile=width, operand_stages=stages,
                                   operand_stage_overrides=operand_stage_overrides if name == "main" else None,
                                   operand_move_attrs=operand_move_attrs,
                                   output_tile_range=(start, stop), blocks=blocks, threads=threads,
                                   cyclic_operand_pipeline=cyclic_operand_pipeline,
                                   k_tail_tiles=k_tail_tiles, block_store=block_store,
                                   tail_continuation_mmas=tail_continuation_mmas,
                                   tail_epilogue_n_tile=tail_epilogue_n_tile,
                                   tail_from_operand_ring=tail_from_operand_ring,
                                   output_prefetch_tiles=output_prefetch_tiles if name == "main" else 0)
        for allocation in find_ops_of_type(cloned, AllocOp):
            if isinstance(allocation.result().type, SmemTensorType):
                allocation.attrs["reuse_group"] = f"wave_{name}_{allocation.attrs.get('reuse_group', 'phase')}"
        binding = dict(zip(cloned.body.arguments, kernel.body.arguments, strict=True))
        pieces.append([clone_op(op, binding) for op in cloned.body.ops])
    kernel.body.ops = [*pieces[0], BarrierOp.build("grid", sync=grid_sync, attrs={"drain_bulk": True}), *pieces[1]]
    kernel.block_dim, kernel.grid_dim = (threads, 1, 1), (blocks, 1, 1)
    kernel.num_warp_groups = threads // 128
    kernel.attrs["cooperative"] = True
    return module


def _schedule_down(
    module: ModuleOp,
    *,
    m_tile: int,
    stages: int,
) -> None:
    TileMatmul(
        op_dims=["M", "N"],
        tile_sizes=[m_tile, 64],
        iteration_bindings=[None, (CollectiveLevel.BLOCK, CollectiveAxis.X)],
        for_loop_labels=["down_m", None],
        op_label_filter="down",
        allow_partial_tiles=True,
    ).run(module)
    schedule_tiled_matmul(
        label="down",
        output_m_tile=m_tile,
        output_n_tile=64,
        operand_stages=stages,
        operand_k_tile=64,
        output_tma_inner_size=None,
        unsafe_reuse_smem=True,
        operand_move_attrs={
            "lhs": {
                "l2_cache_hint": "evict_last",
                "l2_promotion": "128B",
            },
            "rhs": {
                "l2_cache_hint": "evict_first",
                "l2_promotion": "256B",
            },
            "dst": {"l2_cache_hint": "evict_first"},
        },
        loop_label_prefix="down",
    ).run(module)


def schedule_swiglu(
    module: ModuleOp,
    *,
    output_m_tile: int = 64,
    projection_n_tile: int = 192,
) -> ModuleOp:
    """Schedule a one-wave packed SwiGLU GEMM chain as one kernel."""
    source_kernel, projection, down = _match_swiglu_chain(module)
    lhs_type = _gmem(projection.lhs)
    packed_type = _gmem(projection.rhs)
    down_type = _gmem(down.rhs)
    output_type = _gmem(down.dst)
    if not all(
        isinstance(typ.dtype, BF16Type)
        for typ in (lhs_type, packed_type, down_type, output_type)
    ):
        raise ValueError("SwiGLU schedule currently requires BF16 tensors")
    m, k = _static_pair(projection.lhs)
    packed_n, packed_k = _static_pair(projection.rhs)
    output_n, intermediate = _static_pair(down.rhs)
    if (
        packed_k != k
        or packed_n != 2 * intermediate
        or _static_pair(down.dst) != (m, output_n)
    ):
        raise ValueError("inconsistent packed SwiGLU dimensions")
    if m > output_m_tile:
        raise ValueError(f"SwiGLU schedule supports M <= {output_m_tile}, got {m}")
    if intermediate % projection_n_tile or output_n % 64:
        raise ValueError("SwiGLU output dimensions must divide schedule tiles")
    projection_blocks = intermediate // projection_n_tile
    down_blocks = output_n // 64
    if projection_blocks != down_blocks:
        raise ValueError(
            "SwiGLU schedule requires equal projection and down block counts"
        )

    lhs = Value("lhs", clone_type(lhs_type))
    packed_weight = Value("packed_weight", clone_type(packed_type))
    down_weight = Value("down_weight", clone_type(down_type))
    output = Value("output", clone_type(output_type))
    hidden = Value(
        "hidden",
        GmemTensorType.build([m, intermediate], lhs_type.dtype),
    )
    sync = Value("sync", GmemTensorType.build([2], F32Type()))
    projection_ops, _writeback = _projection_ops(
        lhs,
        packed_weight,
        hidden,
        m=m,
        k=k,
        intermediate=intermediate,
        m_tile=output_m_tile,
        n_tile=projection_n_tile,
    )
    down_zero = ConstantOp.build(0.0, RmemTensorType.build([], F32Type()))
    down_op = MatmulOp.build(
        hidden,
        down_weight,
        output,
        collective={axis: CollectiveLevel.GRID for axis in CollectiveAxis},
        init=down_zero.result(),
        attrs={"label": "down"},
    )
    kernel = KernelOp.build(
        source_kernel.name,
        Block(
            arguments=[
                lhs,
                packed_weight,
                down_weight,
                hidden,
                output,
                sync,
            ],
            ops=[
                *projection_ops,
                BarrierOp.build("grid", sync=sync),
                down_zero,
                down_op,
            ],
        ),
        grid_dim=(projection_blocks, 1, 1),
        block_dim=(128, 1, 1),
        num_warp_groups=1,
        ptx_optimization_level=4,
    )
    scheduled = ModuleOp.build(Block(ops=[kernel]))
    _schedule_projection(scheduled)
    _schedule_down(
        scheduled,
        m_tile=output_m_tile,
        stages=5 if m <= 32 else 6,
    )
    ScheduleAsync().run(scheduled)
    return Canonicalize().run(scheduled)


def compile_swiglu(module: ModuleOp) -> PtxRoutePlan:
    """Compile an imported packed SwiGLU region."""
    return PtxRoutePlan(
        make_ptx_kernel(schedule_swiglu(module)),
        bindings=[
            0,
            1,
            2,
            ScratchInit.EMPTY,
            3,
            ScratchInit.ZERO,
        ],
    )
