"""Balance contiguous output-tile/K work across CTAs using ordinary GPU IR."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

from ..base.ir import Block, Op, Value, clone_op, clone_type
from ..gpu_dialect import (
    BlockIdOp, CollectiveAxis, CollectiveLevel, ConstantOp, F32Type, ForOp,
    GmemTensorType, IfOp, IndexAddOp, IndexDivOp, IndexEqOp, IndexLtOp,
    IndexModOp, IndexMulOp, IndexConstantOp, KernelOp, MatmulOp,
    MemoryEffect, ModuleOp, MoveOp, ReshapeOp, RmemTensorType, ThreadIdOp, TileOp,
    root_buffer,
)
from .tile_reduce import ReductionModules, TileReduce, _replay_address_view


@dataclass(frozen=True)
class StreamKPlan:
    """Shape-specialized partition of a flattened tile-major K iteration space.

    Worker i owns [floor(i*total/workers), floor((i+1)*total/workers)).
    Worker counts above the work size are capped, so no CTA has an empty range.
    """

    tiles: int
    k_tiles: int
    workers: int

    def __post_init__(self):
        if any(type(v) is not int or v <= 0 for v in (self.tiles, self.k_tiles, self.workers)):
            raise ValueError("Stream-K requires positive integer tiles, K tiles and workers")
        object.__setattr__(self, "workers", min(self.workers, self.total))

    @property
    def total(self) -> int:
        return self.tiles * self.k_tiles

    def interval(self, worker: int) -> tuple[int, int]:
        if not 0 <= worker < self.workers:
            raise ValueError("worker is outside the Stream-K grid")
        return worker * self.total // self.workers, (worker + 1) * self.total // self.workers

    def owner(self, iteration: int) -> int:
        return ((iteration + 1) * self.workers - 1) // self.total

    @property
    def uniform_contributors(self) -> int | None:
        """Exact contributors when worker boundaries align within every tile."""
        if self.total % self.workers:
            return None
        work = self.total // self.workers
        if work % self.k_tiles == 0:
            return 1
        if self.k_tiles % work == 0:
            return self.k_tiles // work
        return None

    @property
    def slots(self) -> int:
        if self.uniform_contributors is not None:
            return self.uniform_contributors
        return min(self.workers, (self.k_tiles * self.workers + self.total - 1) // self.total + 1)


class _Builder:
    def __init__(self, ops: list[Op]):
        self.ops = ops

    def value(self, op: Op) -> Value:
        self.ops.append(op)
        return op.result()

    def index(self, v: int | Value) -> Value:
        if not isinstance(v, int):
            return v
        op = IndexConstantOp.build(v)
        if v < 0:
            op.result().name = f'cneg{abs(v)}'
        return self.value(op)

    def binary(self, cls, a: int | Value, b: int | Value) -> Value:
        return self.value(cls.build(self.index(a), self.index(b)))

    def subtract(self, a: int | Value, b: int | Value) -> Value:
        return self.binary(IndexAddOp, a, self.binary(IndexMulOp, b, -1))


def partition_stream_k(
    module: ModuleOp, *, workers: int, output_m_tile: int, output_n_tile: int,
    operand_k_tile: int = 64, hero_label_filter: str | None = "mm",
    output_grid_fast_axis: str = "n",
    initialization_threads: int = 32,
) -> ReductionModules:
    """Partition one static GEMM, retaining initialization and consumers in phase two.

    The returned producer is block-tiled GPU IR. Schedule it with
    ``schedule_tiled_matmul``; its K ranges are consumed by the register GEMM
    pipeline. The returned final phase uses the existing TileReduce contract.
    Both phases receive the original arguments followed by one F32 workspace.
    Batches already mapped to grid Z retain independent Stream-K grids.
    """
    from ..base.ir import find_ops_of_type

    heroes = find_ops_of_type(module, MatmulOp)
    if len(heroes) != 1:
        raise ValueError("Stream-K requires one matmul")
    hero = heroes[0]
    lhs = cast(GmemTensorType, hero.lhs.type)
    rhs = cast(GmemTensorType, hero.rhs.type)
    if not isinstance(lhs, GmemTensorType) or not isinstance(rhs, GmemTensorType) or lhs.rank != 2 or rhs.rank != 2:
        raise ValueError("Stream-K requires static rank-two global-memory contractions")
    m, k = lhs.static_shape
    n, rhs_k = rhs.static_shape
    if any(type(v) is not int or v <= 0 for v in (output_m_tile, output_n_tile, operand_k_tile)):
        raise ValueError("Stream-K tile sizes must be positive integers")
    if k != rhs_k or k % operand_k_tile:
        raise ValueError("Stream-K requires matching K divisible by operand_k_tile")
    if type(initialization_threads) is not int or initialization_threads < 32 or initialization_threads % 32:
        raise ValueError('initialization_threads must be a positive multiple of 32')
    if output_grid_fast_axis not in ('m', 'n'):
        raise ValueError("output_grid_fast_axis must be 'm' or 'n'")
    kernel = cast(KernelOp, module.body.ops[0])
    if any((kernel.grid_dim[a] or 1) != 1 or (kernel.cluster_dim[a] or 1) != 1
           for a in (CollectiveAxis.X, CollectiveAxis.Y)):
        raise ValueError("Stream-K owns an unclustered grid X/Y")
    mt, nt = math.ceil(m / output_m_tile), math.ceil(n / output_n_tile)
    plan = StreamKPlan(mt * nt, k // operand_k_tile, workers)
    if plan.total * plan.workers >= 2**32:
        raise ValueError("Stream-K ownership arithmetic exceeds the library's 32-bit index range")
    stages = TileReduce({1: k}, (CollectiveLevel.BLOCK, CollectiveAxis.X),
                        hero_label_filter=hero_label_filter).split(
                            module, num_partials=plan.slots, workspace_alignment=32)
    stages.partials = cast(ModuleOp, clone_op(module, {}))
    pkernel = cast(KernelOp, stages.partials.body.ops[0])
    original = cast(MatmulOp, pkernel.body.ops[kernel.body.ops.index(hero)])
    prologue = list(pkernel.body.ops[:kernel.body.ops.index(hero)])
    # TileReduce inserted partition arithmetic after the original prologue.
    pkernel.body.ops = prologue
    pkernel.grid_dim = (plan.workers, 1, kernel.grid_dim[2] or 1)
    pkernel.attrs['stream_k'] = dict(workers=plan.workers, tiles=plan.tiles,
                                    k_tiles=plan.k_tiles, slots=plan.slots)
    scratch = Value('stream_k_partials', GmemTensorType.build(stages.workspace_shape, F32Type()))
    pkernel.body.append_argument(scratch)
    root = root_buffer(original.dst)
    assert root is not None
    aligned_shape = stages.workspace_shape[1:]
    b = _Builder(pkernel.body.ops)
    worker = b.value(BlockIdOp.build(CollectiveAxis.X))
    start = b.binary(IndexDivOp, b.binary(IndexMulOp, worker, plan.total), plan.workers)
    end = b.binary(IndexDivOp, b.binary(IndexMulOp, b.binary(IndexAddOp, worker, 1), plan.total), plan.workers)
    offset = b.binary(IndexModOp, start, plan.k_tiles)
    length = b.subtract(end, start)
    first_tile = b.binary(IndexDivOp, start, plan.k_tiles)
    categories = sorted({(s % plan.k_tiles, e - s) for s, e in
                         (plan.interval(w) for w in range(plan.workers))})
    workspace_scheme = MemoryEffect.fresh_tiling_scheme_id()
    segment_id = 0
    for first_k, count in categories:
        branch = Block()
        cb = _Builder(branch.ops)
        matches_length = cb.binary(IndexEqOp, length, count)
        body = Block()
        branch.ops.append(IfOp.build(matches_length, body))
        match_offset = b.binary(IndexEqOp, offset, first_k)
        phase = IfOp.build(match_offset, branch)
        pkernel.body.ops.append(phase)
        # Consecutive complete output tiles share one rolled loop and one GEMM.
        groups = []
        remaining, tile_offset = count, 0
        if first_k:
            size = min(remaining, plan.k_tiles - first_k)
            groups.append((0, 1, first_k, first_k + size))
            remaining -= size
            tile_offset = 1
        full, tail = divmod(remaining, plan.k_tiles)
        if full:
            groups.append((tile_offset, tile_offset + full, 0, plan.k_tiles))
            tile_offset += full
        if tail:
            groups.append((tile_offset, tile_offset + 1, 0, tail))
        for begin, stop, ks, ke in groups:
            loop = ForOp.build(begin, stop, attrs={'label': 'stream_k_tiles', 'barrier_phase': True})
            body.ops.append(loop)
            sb = _Builder(loop.body.ops)
            tile = sb.binary(IndexAddOp, first_tile, loop.iv)
            mi = sb.binary(IndexModOp, tile, mt) if output_grid_fast_axis == 'm' else sb.binary(IndexDivOp, tile, nt)
            ni = sb.binary(IndexDivOp, tile, mt) if output_grid_fast_axis == 'm' else sb.binary(IndexModOp, tile, nt)
            tile_start = sb.binary(IndexMulOp, tile, plan.k_tiles)
            first_owner = sb.binary(IndexDivOp, sb.subtract(sb.binary(IndexMulOp, sb.binary(IndexAddOp, tile_start, 1), plan.workers), 1), plan.total)
            slot = sb.subtract(worker, first_owner)
            plane = sb.value(TileOp.build(scratch, [slot, *([0] * len(aligned_shape))],
                             clone_type(scratch.type, {'sizes': [1, *aligned_shape]})))
            view = sb.value(ReshapeOp.build(plane, GmemTensorType.build(aligned_shape, F32Type())))
            logical = sb.value(TileOp.build(view, [0] * len(aligned_shape),
                              clone_type(view.type, {'sizes': list(cast(GmemTensorType, root.type).static_shape)})))
            replayed, output = _replay_address_view(original.dst, root, logical)
            loop.body.ops.extend(replayed)
            dst = sb.value(TileOp.build(output, [mi, ni], clone_type(output.type, {'sizes': [output_m_tile, output_n_tile]}), maybe_partial=m % output_m_tile != 0 or n % output_n_tile != 0))
            operands = []
            for source, rows, row_index, tile_rows in ((original.lhs, m, mi, output_m_tile), (original.rhs, n, ni, output_n_tile)):
                operands.append(sb.value(TileOp.build(source, [row_index, 0], clone_type(source.type, {'sizes': [tile_rows, k]}), maybe_partial=rows % tile_rows != 0)))
            zero = sb.value(ConstantOp.build(0., RmemTensorType.build([], F32Type())))
            collective = dict(original.collective)
            collective[CollectiveAxis.X] = collective[CollectiveAxis.Y] = CollectiveLevel.BLOCK
            attrs = {**original.extra_attrs, 'label': f'mm_s{segment_id}', 'reduction_range': [ks, ke],
                     'dst_tiling_scheme': workspace_scheme}
            segment_id += 1
            attrs.pop('accumulate', None)
            loop.body.ops.append(MatmulOp.build(operands[0], operands[1], dst, collective, init=zero, attrs=attrs))
            # Only the first contributor fills unused slots. Every workspace
            # element read by the final phase has exactly one writer.
            if plan.uniform_contributors is None:
                last_owner = sb.binary(IndexDivOp, sb.subtract(sb.binary(IndexMulOp, sb.binary(IndexAddOp, tile_start, plan.k_tiles), plan.workers), 1), plan.total)
                contributors = sb.binary(IndexAddOp, sb.subtract(last_owner, first_owner), 1)
                is_first = sb.binary(IndexEqOp, slot, 0)
                fill = Block()
                loop.body.ops.append(IfOp.build(is_first, fill))
                _fill_unused(fill, scratch, root, original.dst, aligned_shape, mi, ni,
                             m, n, output_m_tile, output_n_tile, plan.slots, contributors, workspace_scheme, initialization_threads)
    return stages


def _fill_unused(body, scratch, root, original_dst, aligned_shape, mi, ni,
                 m, n, m_tile, n_tile, slots, contributors, workspace_scheme, initialization_threads):
    # Distribute unused-slot stores across the configured CTA threads.
    b = _Builder(body.ops)
    thread = b.value(ThreadIdOp.build(CollectiveAxis.X))
    active = b.binary(IndexLtOp, thread, initialization_threads)
    fill = Block()
    body.ops.append(IfOp.build(active, fill))
    p = ForOp.build(0, slots, attrs={'label': 'stream_k_unused_slots'})
    fill.ops.append(p)
    pb = _Builder(p.body.ops)
    unused = pb.binary(IndexLtOp, contributors, pb.binary(IndexAddOp, p.iv, 1))
    tile_body = Block()
    p.body.ops.append(IfOp.build(unused, tile_body))
    tb = _Builder(tile_body.ops)
    plane = tb.value(TileOp.build(scratch, [p.iv, *([0] * len(aligned_shape))], clone_type(scratch.type, {'sizes': [1, *aligned_shape]})))
    view = tb.value(ReshapeOp.build(plane, GmemTensorType.build(aligned_shape, F32Type())))
    logical = tb.value(TileOp.build(view, [0] * len(aligned_shape), clone_type(view.type, {'sizes': list(cast(GmemTensorType, root.type).static_shape)})))
    replayed, output = _replay_address_view(original_dst, root, logical)
    tile_body.ops.extend(replayed)
    elements = ForOp.build(0, math.ceil(m_tile * n_tile / initialization_threads), attrs={'label': 'stream_k_zero'})
    tile_body.ops.append(elements)
    eb = _Builder(elements.body.ops)
    index = eb.binary(IndexAddOp, eb.binary(IndexMulOp, elements.iv, initialization_threads), thread)
    row = eb.binary(IndexAddOp, eb.binary(IndexMulOp, mi, m_tile), eb.binary(IndexDivOp, index, n_tile))
    col = eb.binary(IndexAddOp, eb.binary(IndexMulOp, ni, n_tile), eb.binary(IndexModOp, index, n_tile))
    row_ok = eb.binary(IndexLtOp, row, m)
    col_ok = eb.binary(IndexLtOp, col, n)
    valid = Block()
    elements.body.ops.append(IfOp.build(row_ok, Block(ops=[IfOp.build(col_ok, valid)])))
    vb = _Builder(valid.ops)
    dst = vb.value(TileOp.build(output, [row, col], clone_type(output.type, {'sizes': [1, 1]})))
    zero = vb.value(ConstantOp.build(0., RmemTensorType.build([1, 1], F32Type())))
    valid.ops.append(MoveOp.build(dst, zero, collective={a: CollectiveLevel.THREAD for a in CollectiveAxis}, attrs={'dst_tiling_scheme': workspace_scheme}))
