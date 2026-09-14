"""Cluster-local split-K construction for small-M fused GEMMs."""
from __future__ import annotations

import textwrap

from ..base.ir import Op, Value
from ..gpu_dialect import (
    AllocOp,
    BF16Type,
    F32Type,
    GmemTensorType,
    IndexConstantOp,
    IndexDivOp,
    IndexModOp,
    MatmulOp,
    MoveOp,
    PTXTemplateOp,
    SmemTensorType,
    WrmemTensorType,
)
from ..passes.tile_matmul import build_persistent_grid, build_tile_ops


def build_persistent_split_k_matmul(
    *,
    resident_blocks: int,
    lhs: Value,
    rhs: Value,
    dst: Value,
    m_tile: int,
    n_tile: int,
    split_k: int,
    label: str,
) -> list[Op]:
    """Build a cluster-local persistent split-K GEMM.

    This deliberately covers the small-M fused workload: one 64-row tile,
    an N tile divisible among one 128-thread warp group, and FP32 partial
    storage. Each cluster owns one output tile. Its CTAs compute disjoint K
    partitions into local shared memory, then all cluster threads gather
    partials through DSMEM and write disjoint pieces of the BF16 result.
    """
    values = (lhs, rhs, dst)
    if any(not isinstance(value.type, GmemTensorType) for value in values):
        raise ValueError("split-K GEMM requires global-memory tensors")
    if any(not isinstance(value.type.dtype, BF16Type) for value in values):
        raise ValueError("split-K GEMM currently requires BF16 tensors")
    m, k = lhs.type.sizes
    n, rhs_k = rhs.type.sizes
    if None in (m, n, k, rhs_k):
        raise ValueError("split-K GEMM requires static dimensions")
    assert isinstance(m, int) and isinstance(n, int) and isinstance(k, int)
    if rhs_k != k or dst.type.sizes != [m, n]:
        raise ValueError("split-K GEMM operand dimensions do not match")
    if split_k < 2 or k % split_k:
        raise ValueError("split_k must evenly divide K and be at least two")
    if resident_blocks % split_k:
        raise ValueError("split_k must evenly divide resident_blocks")
    if not (0 < m <= m_tile) or m_tile != 64:
        raise ValueError(
            "split-K GEMM currently requires M <= m_tile == 64"
        )
    cluster_threads = split_k * 128
    if m * n_tile % cluster_threads:
        raise ValueError("output tile must divide evenly across cluster threads")
    thread_elements = m * n_tile // cluster_threads
    if n_tile % thread_elements:
        raise ValueError("each cluster thread must stay within one output row")

    tiles_n = (n + n_tile - 1) // n_tile
    partial_k = k // split_k

    def reduction_template() -> str:
        template = textwrap.dedent(f"""\
            bar.sync 0;
            barrier.cluster.arrive.release;
            barrier.cluster.wait.acquire;
            mov.u32 ${{b32:dsmem_rank}}, %cluster_ctarank;
            mov.u32 ${{b32:dsmem_tid}}, %tid.x;
            mad.lo.u32 ${{b32:dsmem_cluster_tid}}, ${{b32:dsmem_rank}}, 128, ${{b32:dsmem_tid}};
            mul.lo.u32 ${{b32:dsmem_flat}}, ${{b32:dsmem_cluster_tid}}, {thread_elements};
            div.u32 ${{b32:dsmem_row}}, ${{b32:dsmem_flat}}, {n_tile};
            rem.u32 ${{b32:dsmem_col}}, ${{b32:dsmem_flat}}, {n_tile};
            mad.lo.u32 ${{b32:dsmem_smem_off}}, ${{b32:dsmem_row}}, {n_tile}, ${{b32:dsmem_col}};
            shl.b32 ${{b32:dsmem_smem_off}}, ${{b32:dsmem_smem_off}}, 2;
            mad.lo.u32 ${{b32:dsmem_dst_off}}, ${{b32:dsmem_row}}, {dst.type.strides[0]}, ${{b32:dsmem_col}};
            shl.b32 ${{b32:dsmem_dst_off}}, ${{b32:dsmem_dst_off}}, 1;
            cvt.u64.u32 ${{b64:dsmem_dst_off64}}, ${{b32:dsmem_dst_off}};
            add.u64 ${{b64:dsmem_dst}}, ${{b64:operand.dst.ptr}}, ${{b64:dsmem_dst_off64}};
        """)
        if n % n_tile:
            template += textwrap.dedent(f"""\
                mad.lo.u32 ${{b32:dsmem_global_col}}, ${{b32:operand.tile.val}}, {n_tile}, ${{b32:dsmem_col}};
            """)
        for peer in range(split_k):
            template += textwrap.dedent(f"""\
                mov.u32 ${{b32:dsmem_peer_rank}}, {peer};
                mapa.shared::cluster.u32 ${{b32:dsmem_peer{peer}}}, ${{b32:operand.partial.ptr}}, ${{b32:dsmem_peer_rank}};
                add.u32 ${{b32:dsmem_peer{peer}}}, ${{b32:dsmem_peer{peer}}}, ${{b32:dsmem_smem_off}};
            """)
        for element in range(thread_elements):
            template += "mov.f32 ${f32:dsmem_sum}, 0f00000000;\n"
            for peer in range(split_k):
                template += textwrap.dedent(f"""\
                    ld.shared::cluster.f32 ${{f32:dsmem_value}}, [${{b32:dsmem_peer{peer}}}+{4 * element}];
                    add.f32 ${{f32:dsmem_sum}}, ${{f32:dsmem_sum}}, ${{f32:dsmem_value}};
                """)
            store_predicate = ""
            if n % n_tile:
                template += (
                    "setp.lt.u32 ${pred:dsmem_in_bounds}, "
                    f"${{b32:dsmem_global_col}}, {n};\n"
                )
                store_predicate = "@${pred:dsmem_in_bounds} "
            template += textwrap.dedent(f"""\
                cvt.rn.bf16.f32 ${{b16:dsmem_packed}}, ${{f32:dsmem_sum}};
                {store_predicate}st.global.b16 [${{b64:dsmem_dst}}+{2 * element}], ${{b16:dsmem_packed}};
            """)
            if n % n_tile:
                template += (
                    "add.u32 ${b32:dsmem_global_col}, "
                    "${b32:dsmem_global_col}, 1;\n"
                )
        return template + textwrap.dedent("""\
            barrier.cluster.arrive.release;
            barrier.cluster.wait.acquire;
        """)

    def build_partial(linear: Value) -> list[Op]:
        split_count = IndexConstantOp.build(split_k)
        tile_n = IndexDivOp.build(linear, split_count.result())
        split = IndexModOp.build(linear, split_count.result())
        zero = IndexConstantOp.build(0)
        lhs_ops, tiled_lhs = build_tile_ops(
            lhs, ["M", "K"], op_dims=["M", "K"],
            tile_idxs=[zero.result(), split.result()],
            tile_sizes=[m_tile, partial_k],
        )
        rhs_ops, tiled_rhs = build_tile_ops(
            rhs, ["N", "K"], op_dims=["N", "K"],
            tile_idxs=[tile_n.result(), split.result()],
            tile_sizes=[n_tile, partial_k],
        )
        dst_ops, tiled_dst = build_tile_ops(
            dst, ["M", "N"], op_dims=["M", "N"],
            tile_idxs=[zero.result(), tile_n.result()],
            tile_sizes=[m_tile, n_tile],
        )
        accumulator = AllocOp.build_with_fresh_id(
            WrmemTensorType.build(
                [m_tile, n_tile], F32Type(), warps=4,
                layout="wgmma_layout_0",
            ),
            init=0.0,
        )
        partial = AllocOp.build_with_fresh_id(
            SmemTensorType.build([m_tile, n_tile], F32Type()),
            attrs={"reuse_group": f"{label}_writeback"},
        )
        matmul = MatmulOp.build(
            tiled_lhs, tiled_rhs, accumulator.result(),
            attrs={"label": label, "tiled": ["block.x", "wg.x"]},
        )
        return [
            split_count, tile_n, split, zero,
            *lhs_ops, *rhs_ops, *dst_ops,
            accumulator, partial, matmul,
            MoveOp.build(
                partial.result(), accumulator.result(),
                attrs={"label": f"{label}_dst_smem"},
            ),
            PTXTemplateOp.build(
                operands=[partial.result(), tiled_dst, tile_n.result()],
                result_types=[],
                template=reduction_template(),
                operand_names=["partial", "dst", "tile"],
                result_names=[],
                description="cluster-local split-K reduction",
            ),
        ]

    return build_persistent_grid(
        logical_tiles=split_k * tiles_n,
        resident_blocks=resident_blocks,
        build_tile=build_partial,
    )
