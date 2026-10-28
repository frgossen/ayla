"""Tests for the cross-thread sync-barrier insertion pass."""
import textwrap

import pytest

from ayla import Block, Op, Value
from ayla.analysis.sync import (
    BarrierScope,
    BarrierTarget,
    SyncAnalysis,
    _barrier_covers,
    _barrier_scope,
)
from ayla.base.ir import find_ops_of_type
from ayla.gpu_dialect import (
    AllocOp,
    AsyncOp,
    BarrierOp,
    CollectiveScope,
    KernelOp,
    LockType,
    TmaMoveOp,
    gpu_dialect,
)
from ayla.passes.insert_syncs import (
    _insert_barriers,
    _materialize_cluster_mbars,
    insert_syncs,
)
from tests.testing_utils import assert_contains_in_order


def test_tma_descriptors_for_the_same_argument_alias():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @test(%h: gmem<2x2xf32>, %store: tma_desc<arg_idx=0 gmem<2x2xf32> tile=2x2>, %load: tma_desc<arg_idx=0 gmem<2x2xf32> tile=1x2>) {
            %c0: index = index_constant() attrs={value: 0}
            %s0: smem<2x2xf32> = alloc()
            %s1: smem<1x2xf32> = alloc()
            tma_move(%store: tma_desc<arg_idx=0 gmem<2x2xf32> tile=2x2>, %s0: smem<2x2xf32>, %c0: index, %c0: index)
            tma_move(%s1: smem<1x2xf32>, %load: tma_desc<arg_idx=0 gmem<2x2xf32> tile=1x2>, %c0: index, %c0: index)
          }
        }"""))  # noqa: E501
    analysis = SyncAnalysis()
    analysis.analyze(module)
    store, load = find_ops_of_type(module, TmaMoveOp)

    assert analysis.targets[load].scope == BarrierScope.GRID


def test_disjoint_tiled_writebacks_only_need_block_leader_barrier():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @test(%dst: tma_desc<arg_idx=0 gmem<128x64xbf16> tile=64x64>) grid_dim=[2, 1, 1] block_dim=[128, 1, 1] {
            %c0: index = index_constant() attrs={value: 0}
            %tile: smem<64x64xbf16> = alloc()
            %value: rmem<64x64xbf16> = alloc()
            %lock: lock<bulk> = alloc()
            move(%tile: smem<64x64xbf16>, %value: rmem<64x64xbf16>)
            async(%lock: lock<bulk>) attrs={label: "writeback0_async"} tma_move(%dst: tma_desc<arg_idx=0 gmem<128x64xbf16> tile=64x64>, %tile: smem<64x64xbf16>, %c0: index, %c0: index) attrs={dst_tiling_scheme: 0, label: "writeback0"}
            await(%lock: lock<bulk>)
            move(%tile: smem<64x64xbf16>, %value: rmem<64x64xbf16>)
            async(%lock: lock<bulk>) attrs={label: "writeback1_async"} tma_move(%dst: tma_desc<arg_idx=0 gmem<128x64xbf16> tile=64x64>, %tile: smem<64x64xbf16>, %c0: index, %c0: index) attrs={dst_tiling_scheme: 0, label: "writeback1"}
          }
        }"""))  # noqa: E501
    insert_syncs(module)

    assert repr(module) == textwrap.dedent("""\
        module {
          kernel @test(%dst: tma_desc<arg_idx=0 gmem<128x64xbf16> tile=64x64>) grid_dim=[2, 1, 1] block_dim=[128, 1, 1] {
            %c0: index = index_constant() attrs={value: 0}
            %tile: smem<64x64xbf16> = alloc()
            %value: rmem<64x64xbf16> = alloc()
            %lock: lock<bulk> = alloc()
            move(%tile: smem<64x64xbf16>, %value: rmem<64x64xbf16>)
            barrier() attrs={id: 1, num_threads: 128, scope: "block_leader"}
            async(%lock: lock<bulk>) attrs={label: "writeback0_async"} tma_move(%dst: tma_desc<arg_idx=0 gmem<128x64xbf16> tile=64x64>, %tile: smem<64x64xbf16>, %c0: index, %c0: index) attrs={dst_tiling_scheme: 0, label: "writeback0"}
            await(%lock: lock<bulk>)
            barrier() attrs={scope: "block"}
            move(%tile: smem<64x64xbf16>, %value: rmem<64x64xbf16>)
            barrier() attrs={id: 1, num_threads: 128, scope: "block_leader"}
            async(%lock: lock<bulk>) attrs={label: "writeback1_async"} tma_move(%dst: tma_desc<arg_idx=0 gmem<128x64xbf16> tile=64x64>, %tile: smem<64x64xbf16>, %c0: index, %c0: index) attrs={dst_tiling_scheme: 0, label: "writeback1"}
          }
        }""")  # noqa: E501


def test_block_leader_barrier_requires_whole_warps():
    op = Op(operands=[], results=[])
    block = Block(ops=[op])
    targets = {op: BarrierTarget(BarrierScope.BLOCK_TO_LEADER, named_id=1)}

    with pytest.raises(ValueError, match="whole number of warps"):
        _insert_barriers(block, targets, num_threads=33)


def test_cluster_barrier_scope_is_between_block_and_grid():
    assert (
        _barrier_scope(CollectiveScope.BLOCK, CollectiveScope.CLUSTER)
        == BarrierScope.CLUSTER
    )
    assert _barrier_covers(BarrierScope.CLUSTER, CollectiveScope.CLUSTER)
    assert not _barrier_covers(BarrierScope.CLUSTER, CollectiveScope.GRID)


def test_barrier_scopes_form_a_lattice():
    scopes = list(BarrierScope)
    assert all(a.join(b) == b.join(a) for a in scopes for b in scopes)
    assert all(a.join(a) == a for a in scopes)
    assert all(
        a.join(b).join(c) == a.join(b.join(c))
        for a in scopes
        for b in scopes
        for c in scopes
    )


@pytest.mark.parametrize(
    ("prior", "current", "expected"),
    [
        (
            CollectiveScope.CLUSTER_LEADER,
            CollectiveScope.CLUSTER_LEADER,
            BarrierScope.THREAD,
        ),
        (
            CollectiveScope.THREAD,
            CollectiveScope.CLUSTER_LEADER,
            BarrierScope.CLUSTER_TO_LEADER,
        ),
        (
            CollectiveScope.CLUSTER_LEADER,
            CollectiveScope.THREAD,
            BarrierScope.LEADER_TO_CLUSTER,
        ),
        (
            CollectiveScope.BLOCK_LEADER,
            CollectiveScope.CLUSTER_LEADER,
            BarrierScope.CLUSTER_TO_LEADER,
        ),
        (
            CollectiveScope.CLUSTER_LEADER,
            CollectiveScope.BLOCK_LEADER,
            BarrierScope.LEADER_TO_CLUSTER,
        ),
        (
            CollectiveScope.CLUSTER,
            CollectiveScope.CLUSTER_LEADER,
            BarrierScope.CLUSTER_TO_LEADER,
        ),
        (
            CollectiveScope.CLUSTER_LEADER,
            CollectiveScope.CLUSTER,
            BarrierScope.LEADER_TO_CLUSTER,
        ),
    ],
)
def test_cluster_leader_hazards_choose_a_sufficient_barrier(
    prior, current, expected,
):
    assert _barrier_scope(prior, current) == expected


def test_block_barrier_does_not_cover_a_cluster_leader_effect():
    assert not _barrier_covers(
        BarrierScope.BLOCK, CollectiveScope.CLUSTER_LEADER,
    )
    assert _barrier_covers(
        BarrierScope.CLUSTER, CollectiveScope.CLUSTER_LEADER,
    )


def test_cluster_leader_barrier_coverage_is_directional():
    assert not _barrier_covers(
        BarrierScope.CLUSTER_TO_LEADER, CollectiveScope.CLUSTER_LEADER,
    )
    assert _barrier_covers(
        BarrierScope.LEADER_TO_CLUSTER, CollectiveScope.CLUSTER_LEADER,
    )
    assert not _barrier_covers(
        BarrierScope.LEADER_TO_CLUSTER, CollectiveScope.BLOCK_LEADER,
    )
    assert not _barrier_covers(
        BarrierScope.LEADER_TO_CLUSTER, CollectiveScope.BLOCK,
    )
    assert _barrier_covers(
        BarrierScope.CLUSTER_TO_LEADER, CollectiveScope.BLOCK,
    )


def test_cluster_to_leader_join_retains_its_local_gather():
    assert (
        BarrierScope.CLUSTER_TO_LEADER.join(BarrierScope.BLOCK)
        == BarrierScope.CLUSTER_TO_LEADER
    )


@pytest.mark.parametrize(
    ("scope", "expected"),
    [
        (BarrierScope.CLUSTER_TO_LEADER, "cluster_to_leader"),
        (BarrierScope.LEADER_TO_CLUSTER, "leader_to_cluster"),
    ],
)
def test_insert_syncs_preserves_cluster_barrier_direction(scope, expected):
    op = Op(operands=[], results=[])
    block = Block(ops=[op])
    sync = (
        Value(name="sync", type=LockType.build(kind="cluster_mbar"))
        if scope == BarrierScope.CLUSTER_TO_LEADER
        else None
    )
    barrier_syncs = {op: sync} if sync is not None else None

    _insert_barriers(
        block, {op: BarrierTarget(scope)}, num_threads=32,
        barrier_syncs=barrier_syncs,
    )

    barrier = block.ops[0]
    assert isinstance(barrier, BarrierOp)
    assert barrier.scope == expected
    assert barrier.sync is sync


def test_cluster_mbar_materialization_depends_on_barrier_scope():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @test() grid_dim=[2, 1, 1] block_dim=[32, 1, 1] cluster_dim=[2, 1, 1] {
            %r: rmem<1xf32> = alloc() attrs={id: 0}
          }
        }"""))
    kernel = find_ops_of_type(module, KernelOp)[0]
    target_op = kernel.body.ops[0]

    syncs = _materialize_cluster_mbars(
        kernel,
        {target_op: BarrierTarget(BarrierScope.CLUSTER_TO_LEADER)},
    )

    sync = syncs[target_op]
    assert isinstance(sync.type, LockType)
    assert sync.type.kind == "cluster_mbar"
    assert isinstance(kernel.body.ops[0], AllocOp)
    assert kernel.body.ops[0].result() is sync


def test_batches_lock_initialization_publication():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @k() grid_dim=[2, 1, 1] block_dim=[128, 1, 1] cluster_dim=[2, 1, 1] {
            %local0: lock<mbar> = alloc()
            %buffer: smem<16xf32> = alloc()
            %local1: lock<mbar> = alloc()
            %cluster: lock<cluster_mbar> = alloc()
          }
        }"""))

    insert_syncs(module)

    assert [
        barrier.scope for barrier in find_ops_of_type(module, BarrierOp)
    ] == ["block", "cluster"]


def test_lock_initialization_is_published_before_its_first_use():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @k(%desc: tma_desc<arg_idx=0 gmem<16x16xf32> tile=16x16>) grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {
            %lock0: lock<mbar> = alloc()
            %c0: index = index_constant() attrs={value: 0}
            %lock1: lock<mbar> = alloc()
            %smem: smem<16x16xf32> = alloc()
            async(%lock0: lock<mbar>) tma_move(%smem: smem<16x16xf32>, %desc: tma_desc<arg_idx=0 gmem<16x16xf32> tile=16x16>, %c0: index, %c0: index)
            await(%lock0: lock<mbar>)
          }
        }"""))

    insert_syncs(module)

    ops = module.body.ops[0].body.ops
    [barrier] = [op for op in ops if isinstance(op, BarrierOp)]
    assert barrier.scope == "block"
    # The constant between the allocations does not split the publication.
    assert ops[ops.index(barrier) + 1] is find_ops_of_type(module, AsyncOp)[0]


def test_tma_multicast_reuse_waits_for_every_recipient_cta():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @test(%desc: tma_desc<arg_idx=0 gmem<16x16xf32> tile=16x16>) grid_dim=[2, 1, 1] block_dim=[32, 1, 1] cluster_dim=[2, 1, 1] {
            %c0: index = index_constant() attrs={value: 0}
            %smem: smem<16x16xf32> = alloc() attrs={id: 0}
            %lock: lock<mbar> = alloc() attrs={id: 1}
            async(%lock: lock<mbar>) collective=xyz.cluster tma_move(%smem: smem<16x16xf32>, %desc: tma_desc<arg_idx=0 gmem<16x16xf32> tile=16x16>, %c0: index, %c0: index) collective=xyz.cluster attrs={multicast: True}
            await(%lock: lock<mbar>)
            %rmem: rmem<16x16xf32> = alloc()
            move(%rmem: rmem<16x16xf32>, %smem: smem<16x16xf32>)
            async(%lock: lock<mbar>) collective=xyz.cluster tma_move(%smem: smem<16x16xf32>, %desc: tma_desc<arg_idx=0 gmem<16x16xf32> tile=16x16>, %c0: index, %c0: index) collective=xyz.cluster attrs={multicast: True}
            await(%lock: lock<mbar>)
          }
        }"""))

    insert_syncs(module)

    text = repr(module)
    # One cluster rendezvous publishes both locks' initialization. Reuse only
    # needs the separate one-way handoff to the next multicast issuer.
    assert text.count('barrier() attrs={scope: "cluster"}') == 1
    barriers = [
        op for op in find_ops_of_type(module, BarrierOp)
        if op.scope == "cluster_to_leader"
    ]
    assert len(barriers) == 1
    sync = barriers[0].sync
    assert sync is not None
    assert isinstance(sync.type, LockType)
    assert sync.type.kind == "cluster_mbar"


def test_preexisting_barrier_updates_dataflow_state():
    source = """\
        module {
          kernel @test(%g: gmem<1xf32>) grid_dim=[1, 1, 1] block_dim=[32, 1, 1] {
            %r: rmem<1xf32> = alloc()
            move(%g: gmem<1xf32>, %r: rmem<1xf32>)
            barrier() attrs={scope: \"__SCOPE__\"}
            move(%r: rmem<1xf32>, %g: gmem<1xf32>)
          }
        }"""

    # A block barrier cannot discharge a grid-scoped global-memory effect.
    with pytest.raises(NotImplementedError, match="barrier GRID"):
        insert_syncs(gpu_dialect.parse(textwrap.dedent(
            source.replace("__SCOPE__", "block"),
        )))

    # Scheduling passes may author structural barriers before InsertSyncs. The
    # grid barrier discharges the effect, so the pass must not insert another.
    module = gpu_dialect.parse(textwrap.dedent(
        source.replace("__SCOPE__", "grid"),
    ))
    insert_syncs(module)
    assert repr(module).count("barrier()") == 1


def test_war_on_shared_buffer_gets_barrier():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @test(%a: gmem<128x4096xbf16>, %b: gmem<128x4096xbf16>, %acc: wgrmem<128x128xf32 warps=4 layout="wgmma_layout_0">) grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {
            %sa: smem<128x64xbf16 swizzle="128B"> = alloc() attrs={id: 0}
            %sb: smem<128x64xbf16 swizzle="128B"> = alloc() attrs={id: 1}
            for (%k: index = 0 to 4) attrs={label: "k"} {
              %ak: gmem<128x64xbf16 strides=[4096, 1]> = tile(%a: gmem<128x4096xbf16>, 0, %k: index)
              %bk: gmem<128x64xbf16 strides=[4096, 1]> = tile(%b: gmem<128x4096xbf16>, 0, %k: index)
              %lm: lock<mbar> = alloc() attrs={id: 2}
              async(%lm: lock<mbar>) {
                move(%sa: smem<128x64xbf16 swizzle="128B">, %ak: gmem<128x64xbf16 strides=[4096, 1]>) attrs={tma: True}
                move(%sb: smem<128x64xbf16 swizzle="128B">, %bk: gmem<128x64xbf16 strides=[4096, 1]>) attrs={tma: True}
              }
              await(%lm: lock<mbar>)
              %lw: lock<wgmma> = alloc() attrs={id: 3}
              async(%lw: lock<wgmma>) matmul(%sa: smem<128x64xbf16 swizzle="128B">, %sb: smem<128x64xbf16 swizzle="128B">, %acc: wgrmem<128x128xf32 warps=4 layout="wgmma_layout_0">) collective=[xy.block, z.thread]
              await(%lw: lock<wgmma>)
            }
          }
        }"""))  # noqa: E501

    insert_syncs(module)

    # One named (block_leader) barrier, before the TMA loads: the WAR on
    # %sa/%sb vs. the prior iteration's matmul reads. It's block_leader,
    # not block,
    # because the TMA write is issued by a single leader thread (all-arrive /
    # t0-wait suffices). The matmul's own RAW on the freshly loaded %sa/%sb needs
    # no barrier — the load's mbar await already covers it.
    assert repr(module) == textwrap.dedent("""\
        module {
          kernel @test(%a: gmem<128x4096xbf16>, %b: gmem<128x4096xbf16>, %acc: wgrmem<128x128xf32 warps=4 layout="wgmma_layout_0">) grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {
            %sa: smem<128x64xbf16 swizzle="128B"> = alloc() attrs={id: 0}
            %sb: smem<128x64xbf16 swizzle="128B"> = alloc() attrs={id: 1}
            for (%k: index = 0 to 4) attrs={label: "k"} {
              %ak: gmem<128x64xbf16 strides=[4096, 1]> = tile(%a: gmem<128x4096xbf16>, 0, %k: index)
              %bk: gmem<128x64xbf16 strides=[4096, 1]> = tile(%b: gmem<128x4096xbf16>, 0, %k: index)
              %lm: lock<mbar> = alloc() attrs={id: 2}
              barrier() attrs={scope: "block"}
              async(%lm: lock<mbar>) {
                move(%sa: smem<128x64xbf16 swizzle="128B">, %ak: gmem<128x64xbf16 strides=[4096, 1]>) attrs={tma: True}
                move(%sb: smem<128x64xbf16 swizzle="128B">, %bk: gmem<128x64xbf16 strides=[4096, 1]>) attrs={tma: True}
              }
              await(%lm: lock<mbar>)
              %lw: lock<wgmma> = alloc() attrs={id: 3}
              async(%lw: lock<wgmma>) matmul(%sa: smem<128x64xbf16 swizzle="128B">, %sb: smem<128x64xbf16 swizzle="128B">, %acc: wgrmem<128x128xf32 warps=4 layout="wgmma_layout_0">) collective=[xy.block, z.thread]
              await(%lw: lock<wgmma>)
            }
          }
        }""")  # noqa: E501



def test_wgmma256_wg_pipeline_inserts_barriers():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @auto_matmul_4096_wgmma256_wg(%a: gmem<4096x4096xbf16>, %b: gmem<4096x4096xbf16>, %c: gmem<4096x4096xbf16>) grid_dim=[32, 16, 1] block_dim=[256, 1, 1] num_warp_groups=2 {
            %c0: index = index_constant() attrs={value: 0}
            %0: smem<256x64xbf16 swizzle="128B"> = alloc() attrs={id: 0}
            %1: lock<mbar> = alloc() attrs={id: 1}
            %2: smem<128x64xbf16 swizzle="128B"> = alloc() attrs={id: 2}
            %c3: index = index_constant() attrs={value: 3}
            %3: smem<256x64xbf16 swizzle="128B"> = alloc() attrs={id: 3}
            %4: lock<mbar> = alloc() attrs={id: 4}
            %5: smem<128x64xbf16 swizzle="128B"> = alloc() attrs={id: 5}
            %c2: index = index_constant() attrs={value: 2}
            %6: smem<256x64xbf16 swizzle="128B"> = alloc() attrs={id: 6}
            %7: lock<mbar> = alloc() attrs={id: 7}
            %8: smem<128x64xbf16 swizzle="128B"> = alloc() attrs={id: 8}
            %c1: index = index_constant() attrs={value: 1}
            %9: smem<256x64xbf16 swizzle="128B"> = alloc() attrs={id: 9}
            %10: lock<mbar> = alloc() attrs={id: 10}
            %11: smem<128x64xbf16 swizzle="128B"> = alloc() attrs={id: 11}
            %bx: index = block_id.x
            %by: index = block_id.y
            %12: gmem<256x4096xbf16> = tile(%b: gmem<4096x4096xbf16>, %by: index, 0)
            %13: gmem<256x64xbf16 strides=[4096, 1]> = tile(%12: gmem<256x4096xbf16>, 0, %c3: index)
            %14: gmem<256x64xbf16 strides=[4096, 1]> = tile(%12: gmem<256x4096xbf16>, 0, %c2: index)
            %15: gmem<256x64xbf16 strides=[4096, 1]> = tile(%12: gmem<256x4096xbf16>, 0, %c1: index)
            %16: gmem<256x64xbf16 strides=[4096, 1]> = tile(%12: gmem<256x4096xbf16>, 0, %c0: index)
            %17: gmem<128x4096xbf16> = tile(%a: gmem<4096x4096xbf16>, %bx: index, 0)
            %18: gmem<128x64xbf16 strides=[4096, 1]> = tile(%17: gmem<128x4096xbf16>, 0, %c3: index)
            async(%1: lock<mbar>) attrs={label: "mm_lhs_smem_iter3_async"} {
              move(%2: smem<128x64xbf16 swizzle="128B">, %18: gmem<128x64xbf16 strides=[4096, 1]>) attrs={label: "mm_lhs_smem_iter3", nc: True, tma: True}
              move(%0: smem<256x64xbf16 swizzle="128B">, %13: gmem<256x64xbf16 strides=[4096, 1]>) attrs={label: "mm_rhs_smem_iter3", nc: True, tma: True}
            }
            %19: gmem<128x64xbf16 strides=[4096, 1]> = tile(%17: gmem<128x4096xbf16>, 0, %c2: index)
            async(%4: lock<mbar>) attrs={label: "mm_lhs_smem_iter2_async"} {
              move(%5: smem<128x64xbf16 swizzle="128B">, %19: gmem<128x64xbf16 strides=[4096, 1]>) attrs={label: "mm_lhs_smem_iter2", nc: True, tma: True}
              move(%3: smem<256x64xbf16 swizzle="128B">, %14: gmem<256x64xbf16 strides=[4096, 1]>) attrs={label: "mm_rhs_smem_iter2", nc: True, tma: True}
            }
            %20: gmem<128x64xbf16 strides=[4096, 1]> = tile(%17: gmem<128x4096xbf16>, 0, %c1: index)
            async(%7: lock<mbar>) attrs={label: "mm_lhs_smem_iter1_async"} {
              move(%8: smem<128x64xbf16 swizzle="128B">, %20: gmem<128x64xbf16 strides=[4096, 1]>) attrs={label: "mm_lhs_smem_iter1", nc: True, tma: True}
              move(%6: smem<256x64xbf16 swizzle="128B">, %15: gmem<256x64xbf16 strides=[4096, 1]>) attrs={label: "mm_rhs_smem_iter1", nc: True, tma: True}
            }
            %21: gmem<128x64xbf16 strides=[4096, 1]> = tile(%17: gmem<128x4096xbf16>, 0, %c0: index)
            async(%10: lock<mbar>) attrs={label: "mm_lhs_smem_iter0_async"} {
              move(%11: smem<128x64xbf16 swizzle="128B">, %21: gmem<128x64xbf16 strides=[4096, 1]>) attrs={label: "mm_lhs_smem_iter0", nc: True, tma: True}
              move(%9: smem<256x64xbf16 swizzle="128B">, %16: gmem<256x64xbf16 strides=[4096, 1]>) attrs={label: "mm_rhs_smem_iter0", nc: True, tma: True}
            }
            %22: gmem<128x4096xbf16> = tile(%c: gmem<4096x4096xbf16>, %bx: index, 0)
            %23: gmem<128x256xbf16 strides=[4096, 1]> = tile(%22: gmem<128x4096xbf16>, 0, %by: index)
            %wgx: index, %wgy: index, %wgz: index = warp_group_id()
            %24: gmem<64x256xbf16 strides=[4096, 1]> = tile(%23: gmem<128x256xbf16 strides=[4096, 1]>, %wgx: index, 0)
            %25: wgrmem<64x256xf32 warps=4 layout="wgmma_layout_0"> = alloc() attrs={id: 12}
            %25_init: rmem<f32> = constant() attrs={value: 0.0}
            elementwise fill(%25: wgrmem<64x256xf32 warps=4 layout="wgmma_layout_0">, %25_init: rmem<f32>)
            %26: smem<64x64xbf16 swizzle="128B"> = tile(%11: smem<128x64xbf16 swizzle="128B">, %wgx: index, 0)
            %27: smem<64x64xbf16 swizzle="128B"> = tile(%8: smem<128x64xbf16 swizzle="128B">, %wgx: index, 0)
            %28: smem<64x64xbf16 swizzle="128B"> = tile(%5: smem<128x64xbf16 swizzle="128B">, %wgx: index, 0)
            %29: smem<64x64xbf16 swizzle="128B"> = tile(%2: smem<128x64xbf16 swizzle="128B">, %wgx: index, 0)
            %30: lock<wgmma> = alloc() attrs={id: 13}
            %31: lock<wgmma> = alloc() attrs={id: 14}
            %32: lock<wgmma> = alloc() attrs={id: 15}
            %33: lock<wgmma> = alloc() attrs={id: 16}
            %c4: index = index_constant() attrs={value: 4}
            %34: smem<64x16xbf16 strides=[64, 1] swizzle="128B"> = tile(%26: smem<64x64xbf16 swizzle="128B">, 0, %c0: index)
            %35: smem<256x16xbf16 strides=[64, 1] swizzle="128B"> = tile(%9: smem<256x64xbf16 swizzle="128B">, 0, %c0: index)
            %36: smem<64x16xbf16 strides=[64, 1] swizzle="128B"> = tile(%26: smem<64x64xbf16 swizzle="128B">, 0, %c1: index)
            %37: smem<256x16xbf16 strides=[64, 1] swizzle="128B"> = tile(%9: smem<256x64xbf16 swizzle="128B">, 0, %c1: index)
            %38: smem<64x16xbf16 strides=[64, 1] swizzle="128B"> = tile(%26: smem<64x64xbf16 swizzle="128B">, 0, %c2: index)
            %39: smem<256x16xbf16 strides=[64, 1] swizzle="128B"> = tile(%9: smem<256x64xbf16 swizzle="128B">, 0, %c2: index)
            %40: smem<64x16xbf16 strides=[64, 1] swizzle="128B"> = tile(%26: smem<64x64xbf16 swizzle="128B">, 0, %c3: index)
            %41: smem<256x16xbf16 strides=[64, 1] swizzle="128B"> = tile(%9: smem<256x64xbf16 swizzle="128B">, 0, %c3: index)
            %42: smem<64x16xbf16 strides=[64, 1] swizzle="128B"> = tile(%27: smem<64x64xbf16 swizzle="128B">, 0, %c0: index)
            %43: smem<256x16xbf16 strides=[64, 1] swizzle="128B"> = tile(%6: smem<256x64xbf16 swizzle="128B">, 0, %c0: index)
            %44: smem<64x16xbf16 strides=[64, 1] swizzle="128B"> = tile(%27: smem<64x64xbf16 swizzle="128B">, 0, %c1: index)
            %45: smem<256x16xbf16 strides=[64, 1] swizzle="128B"> = tile(%6: smem<256x64xbf16 swizzle="128B">, 0, %c1: index)
            %46: smem<64x16xbf16 strides=[64, 1] swizzle="128B"> = tile(%27: smem<64x64xbf16 swizzle="128B">, 0, %c2: index)
            %47: smem<256x16xbf16 strides=[64, 1] swizzle="128B"> = tile(%6: smem<256x64xbf16 swizzle="128B">, 0, %c2: index)
            %48: smem<64x16xbf16 strides=[64, 1] swizzle="128B"> = tile(%27: smem<64x64xbf16 swizzle="128B">, 0, %c3: index)
            %49: smem<256x16xbf16 strides=[64, 1] swizzle="128B"> = tile(%6: smem<256x64xbf16 swizzle="128B">, 0, %c3: index)
            %50: smem<64x16xbf16 strides=[64, 1] swizzle="128B"> = tile(%28: smem<64x64xbf16 swizzle="128B">, 0, %c0: index)
            %51: smem<256x16xbf16 strides=[64, 1] swizzle="128B"> = tile(%3: smem<256x64xbf16 swizzle="128B">, 0, %c0: index)
            %52: smem<64x16xbf16 strides=[64, 1] swizzle="128B"> = tile(%28: smem<64x64xbf16 swizzle="128B">, 0, %c1: index)
            %53: smem<256x16xbf16 strides=[64, 1] swizzle="128B"> = tile(%3: smem<256x64xbf16 swizzle="128B">, 0, %c1: index)
            %54: smem<64x16xbf16 strides=[64, 1] swizzle="128B"> = tile(%28: smem<64x64xbf16 swizzle="128B">, 0, %c2: index)
            %55: smem<256x16xbf16 strides=[64, 1] swizzle="128B"> = tile(%3: smem<256x64xbf16 swizzle="128B">, 0, %c2: index)
            %56: smem<64x16xbf16 strides=[64, 1] swizzle="128B"> = tile(%28: smem<64x64xbf16 swizzle="128B">, 0, %c3: index)
            %57: smem<256x16xbf16 strides=[64, 1] swizzle="128B"> = tile(%3: smem<256x64xbf16 swizzle="128B">, 0, %c3: index)
            %58: smem<64x16xbf16 strides=[64, 1] swizzle="128B"> = tile(%29: smem<64x64xbf16 swizzle="128B">, 0, %c0: index)
            %59: smem<256x16xbf16 strides=[64, 1] swizzle="128B"> = tile(%0: smem<256x64xbf16 swizzle="128B">, 0, %c0: index)
            %60: smem<64x16xbf16 strides=[64, 1] swizzle="128B"> = tile(%29: smem<64x64xbf16 swizzle="128B">, 0, %c1: index)
            %61: smem<256x16xbf16 strides=[64, 1] swizzle="128B"> = tile(%0: smem<256x64xbf16 swizzle="128B">, 0, %c1: index)
            %62: smem<64x16xbf16 strides=[64, 1] swizzle="128B"> = tile(%29: smem<64x64xbf16 swizzle="128B">, 0, %c2: index)
            %63: smem<256x16xbf16 strides=[64, 1] swizzle="128B"> = tile(%0: smem<256x64xbf16 swizzle="128B">, 0, %c2: index)
            %64: smem<64x16xbf16 strides=[64, 1] swizzle="128B"> = tile(%29: smem<64x64xbf16 swizzle="128B">, 0, %c3: index)
            %65: smem<256x16xbf16 strides=[64, 1] swizzle="128B"> = tile(%0: smem<256x64xbf16 swizzle="128B">, 0, %c3: index)
            %c5: index = index_constant() attrs={value: 5}
            %c6: index = index_constant() attrs={value: 6}
            %c7: index = index_constant() attrs={value: 7}
            for (%k: index = 0 to 60 step 4) attrs={label: "k"} {
              %66: index = index_add(%k: index, %c7: index)
              %67: gmem<128x64xbf16 strides=[4096, 1]> = tile(%17: gmem<128x4096xbf16>, 0, %66: index)
              %68: gmem<256x64xbf16 strides=[4096, 1]> = tile(%12: gmem<256x4096xbf16>, 0, %66: index)
              %69: index = index_add(%k: index, %c6: index)
              %70: gmem<128x64xbf16 strides=[4096, 1]> = tile(%17: gmem<128x4096xbf16>, 0, %69: index)
              %71: gmem<256x64xbf16 strides=[4096, 1]> = tile(%12: gmem<256x4096xbf16>, 0, %69: index)
              %72: index = index_add(%k: index, %c5: index)
              %73: gmem<128x64xbf16 strides=[4096, 1]> = tile(%17: gmem<128x4096xbf16>, 0, %72: index)
              %74: gmem<256x64xbf16 strides=[4096, 1]> = tile(%12: gmem<256x4096xbf16>, 0, %72: index)
              %75: index = index_add(%k: index, %c4: index)
              %76: gmem<128x64xbf16 strides=[4096, 1]> = tile(%17: gmem<128x4096xbf16>, 0, %75: index)
              %77: gmem<256x64xbf16 strides=[4096, 1]> = tile(%12: gmem<256x4096xbf16>, 0, %75: index)
              await(%10: lock<mbar>)
              async(%30: lock<wgmma>) attrs={label: "mm_iter0_async_iter0"} {
                matmul(%34: smem<64x16xbf16 strides=[64, 1] swizzle="128B">, %35: smem<256x16xbf16 strides=[64, 1] swizzle="128B">, %25: wgrmem<64x256xf32 warps=4 layout="wgmma_layout_0">) collective=[xy.block, z.thread] attrs={label: "mm_iter0_iter0"}
                matmul(%36: smem<64x16xbf16 strides=[64, 1] swizzle="128B">, %37: smem<256x16xbf16 strides=[64, 1] swizzle="128B">, %25: wgrmem<64x256xf32 warps=4 layout="wgmma_layout_0">) collective=[xy.block, z.thread] attrs={label: "mm_iter0_iter1"}
                matmul(%38: smem<64x16xbf16 strides=[64, 1] swizzle="128B">, %39: smem<256x16xbf16 strides=[64, 1] swizzle="128B">, %25: wgrmem<64x256xf32 warps=4 layout="wgmma_layout_0">) collective=[xy.block, z.thread] attrs={label: "mm_iter0_iter2"}
                matmul(%40: smem<64x16xbf16 strides=[64, 1] swizzle="128B">, %41: smem<256x16xbf16 strides=[64, 1] swizzle="128B">, %25: wgrmem<64x256xf32 warps=4 layout="wgmma_layout_0">) collective=[xy.block, z.thread] attrs={label: "mm_iter0_iter3"}
              }
              await(%30: lock<wgmma>)
              async(%10: lock<mbar>) attrs={label: "mm_lhs_smem_iter0_async"} {
                move(%11: smem<128x64xbf16 swizzle="128B">, %76: gmem<128x64xbf16 strides=[4096, 1]>) attrs={label: "mm_lhs_smem_iter0", nc: True, tma: True}
                move(%9: smem<256x64xbf16 swizzle="128B">, %77: gmem<256x64xbf16 strides=[4096, 1]>) attrs={label: "mm_rhs_smem_iter0", nc: True, tma: True}
              }
              await(%7: lock<mbar>)
              async(%31: lock<wgmma>) attrs={label: "mm_iter1_async_iter0"} {
                matmul(%42: smem<64x16xbf16 strides=[64, 1] swizzle="128B">, %43: smem<256x16xbf16 strides=[64, 1] swizzle="128B">, %25: wgrmem<64x256xf32 warps=4 layout="wgmma_layout_0">) collective=[xy.block, z.thread] attrs={label: "mm_iter1_iter0"}
                matmul(%44: smem<64x16xbf16 strides=[64, 1] swizzle="128B">, %45: smem<256x16xbf16 strides=[64, 1] swizzle="128B">, %25: wgrmem<64x256xf32 warps=4 layout="wgmma_layout_0">) collective=[xy.block, z.thread] attrs={label: "mm_iter1_iter1"}
                matmul(%46: smem<64x16xbf16 strides=[64, 1] swizzle="128B">, %47: smem<256x16xbf16 strides=[64, 1] swizzle="128B">, %25: wgrmem<64x256xf32 warps=4 layout="wgmma_layout_0">) collective=[xy.block, z.thread] attrs={label: "mm_iter1_iter2"}
                matmul(%48: smem<64x16xbf16 strides=[64, 1] swizzle="128B">, %49: smem<256x16xbf16 strides=[64, 1] swizzle="128B">, %25: wgrmem<64x256xf32 warps=4 layout="wgmma_layout_0">) collective=[xy.block, z.thread] attrs={label: "mm_iter1_iter3"}
              }
              await(%31: lock<wgmma>)
              async(%7: lock<mbar>) attrs={label: "mm_lhs_smem_iter1_async"} {
                move(%8: smem<128x64xbf16 swizzle="128B">, %73: gmem<128x64xbf16 strides=[4096, 1]>) attrs={label: "mm_lhs_smem_iter1", nc: True, tma: True}
                move(%6: smem<256x64xbf16 swizzle="128B">, %74: gmem<256x64xbf16 strides=[4096, 1]>) attrs={label: "mm_rhs_smem_iter1", nc: True, tma: True}
              }
              await(%4: lock<mbar>)
              async(%32: lock<wgmma>) attrs={label: "mm_iter2_async_iter0"} {
                matmul(%50: smem<64x16xbf16 strides=[64, 1] swizzle="128B">, %51: smem<256x16xbf16 strides=[64, 1] swizzle="128B">, %25: wgrmem<64x256xf32 warps=4 layout="wgmma_layout_0">) collective=[xy.block, z.thread] attrs={label: "mm_iter2_iter0"}
                matmul(%52: smem<64x16xbf16 strides=[64, 1] swizzle="128B">, %53: smem<256x16xbf16 strides=[64, 1] swizzle="128B">, %25: wgrmem<64x256xf32 warps=4 layout="wgmma_layout_0">) collective=[xy.block, z.thread] attrs={label: "mm_iter2_iter1"}
                matmul(%54: smem<64x16xbf16 strides=[64, 1] swizzle="128B">, %55: smem<256x16xbf16 strides=[64, 1] swizzle="128B">, %25: wgrmem<64x256xf32 warps=4 layout="wgmma_layout_0">) collective=[xy.block, z.thread] attrs={label: "mm_iter2_iter2"}
                matmul(%56: smem<64x16xbf16 strides=[64, 1] swizzle="128B">, %57: smem<256x16xbf16 strides=[64, 1] swizzle="128B">, %25: wgrmem<64x256xf32 warps=4 layout="wgmma_layout_0">) collective=[xy.block, z.thread] attrs={label: "mm_iter2_iter3"}
              }
              await(%32: lock<wgmma>)
              async(%4: lock<mbar>) attrs={label: "mm_lhs_smem_iter2_async"} {
                move(%5: smem<128x64xbf16 swizzle="128B">, %70: gmem<128x64xbf16 strides=[4096, 1]>) attrs={label: "mm_lhs_smem_iter2", nc: True, tma: True}
                move(%3: smem<256x64xbf16 swizzle="128B">, %71: gmem<256x64xbf16 strides=[4096, 1]>) attrs={label: "mm_rhs_smem_iter2", nc: True, tma: True}
              }
              await(%1: lock<mbar>)
              async(%33: lock<wgmma>) attrs={label: "mm_iter3_async_iter0"} {
                matmul(%58: smem<64x16xbf16 strides=[64, 1] swizzle="128B">, %59: smem<256x16xbf16 strides=[64, 1] swizzle="128B">, %25: wgrmem<64x256xf32 warps=4 layout="wgmma_layout_0">) collective=[xy.block, z.thread] attrs={label: "mm_iter3_iter0"}
                matmul(%60: smem<64x16xbf16 strides=[64, 1] swizzle="128B">, %61: smem<256x16xbf16 strides=[64, 1] swizzle="128B">, %25: wgrmem<64x256xf32 warps=4 layout="wgmma_layout_0">) collective=[xy.block, z.thread] attrs={label: "mm_iter3_iter1"}
                matmul(%62: smem<64x16xbf16 strides=[64, 1] swizzle="128B">, %63: smem<256x16xbf16 strides=[64, 1] swizzle="128B">, %25: wgrmem<64x256xf32 warps=4 layout="wgmma_layout_0">) collective=[xy.block, z.thread] attrs={label: "mm_iter3_iter2"}
                matmul(%64: smem<64x16xbf16 strides=[64, 1] swizzle="128B">, %65: smem<256x16xbf16 strides=[64, 1] swizzle="128B">, %25: wgrmem<64x256xf32 warps=4 layout="wgmma_layout_0">) collective=[xy.block, z.thread] attrs={label: "mm_iter3_iter3"}
              }
              await(%33: lock<wgmma>)
              async(%1: lock<mbar>) attrs={label: "mm_lhs_smem_iter3_async"} {
                move(%2: smem<128x64xbf16 swizzle="128B">, %67: gmem<128x64xbf16 strides=[4096, 1]>) attrs={label: "mm_lhs_smem_iter3", nc: True, tma: True}
                move(%0: smem<256x64xbf16 swizzle="128B">, %68: gmem<256x64xbf16 strides=[4096, 1]>) attrs={label: "mm_rhs_smem_iter3", nc: True, tma: True}
              }
            }
            await(%10: lock<mbar>)
            async(%30: lock<wgmma>) attrs={label: "mm_iter0_async_iter0"} {
              matmul(%34: smem<64x16xbf16 strides=[64, 1] swizzle="128B">, %35: smem<256x16xbf16 strides=[64, 1] swizzle="128B">, %25: wgrmem<64x256xf32 warps=4 layout="wgmma_layout_0">) collective=[xy.block, z.thread] attrs={label: "mm_iter0_iter0"}
              matmul(%36: smem<64x16xbf16 strides=[64, 1] swizzle="128B">, %37: smem<256x16xbf16 strides=[64, 1] swizzle="128B">, %25: wgrmem<64x256xf32 warps=4 layout="wgmma_layout_0">) collective=[xy.block, z.thread] attrs={label: "mm_iter0_iter1"}
              matmul(%38: smem<64x16xbf16 strides=[64, 1] swizzle="128B">, %39: smem<256x16xbf16 strides=[64, 1] swizzle="128B">, %25: wgrmem<64x256xf32 warps=4 layout="wgmma_layout_0">) collective=[xy.block, z.thread] attrs={label: "mm_iter0_iter2"}
              matmul(%40: smem<64x16xbf16 strides=[64, 1] swizzle="128B">, %41: smem<256x16xbf16 strides=[64, 1] swizzle="128B">, %25: wgrmem<64x256xf32 warps=4 layout="wgmma_layout_0">) collective=[xy.block, z.thread] attrs={label: "mm_iter0_iter3"}
            }
            await(%30: lock<wgmma>)
            await(%7: lock<mbar>)
            async(%31: lock<wgmma>) attrs={label: "mm_iter1_async_iter0"} {
              matmul(%42: smem<64x16xbf16 strides=[64, 1] swizzle="128B">, %43: smem<256x16xbf16 strides=[64, 1] swizzle="128B">, %25: wgrmem<64x256xf32 warps=4 layout="wgmma_layout_0">) collective=[xy.block, z.thread] attrs={label: "mm_iter1_iter0"}
              matmul(%44: smem<64x16xbf16 strides=[64, 1] swizzle="128B">, %45: smem<256x16xbf16 strides=[64, 1] swizzle="128B">, %25: wgrmem<64x256xf32 warps=4 layout="wgmma_layout_0">) collective=[xy.block, z.thread] attrs={label: "mm_iter1_iter1"}
              matmul(%46: smem<64x16xbf16 strides=[64, 1] swizzle="128B">, %47: smem<256x16xbf16 strides=[64, 1] swizzle="128B">, %25: wgrmem<64x256xf32 warps=4 layout="wgmma_layout_0">) collective=[xy.block, z.thread] attrs={label: "mm_iter1_iter2"}
              matmul(%48: smem<64x16xbf16 strides=[64, 1] swizzle="128B">, %49: smem<256x16xbf16 strides=[64, 1] swizzle="128B">, %25: wgrmem<64x256xf32 warps=4 layout="wgmma_layout_0">) collective=[xy.block, z.thread] attrs={label: "mm_iter1_iter3"}
            }
            await(%31: lock<wgmma>)
            await(%4: lock<mbar>)
            async(%32: lock<wgmma>) attrs={label: "mm_iter2_async_iter0"} {
              matmul(%50: smem<64x16xbf16 strides=[64, 1] swizzle="128B">, %51: smem<256x16xbf16 strides=[64, 1] swizzle="128B">, %25: wgrmem<64x256xf32 warps=4 layout="wgmma_layout_0">) collective=[xy.block, z.thread] attrs={label: "mm_iter2_iter0"}
              matmul(%52: smem<64x16xbf16 strides=[64, 1] swizzle="128B">, %53: smem<256x16xbf16 strides=[64, 1] swizzle="128B">, %25: wgrmem<64x256xf32 warps=4 layout="wgmma_layout_0">) collective=[xy.block, z.thread] attrs={label: "mm_iter2_iter1"}
              matmul(%54: smem<64x16xbf16 strides=[64, 1] swizzle="128B">, %55: smem<256x16xbf16 strides=[64, 1] swizzle="128B">, %25: wgrmem<64x256xf32 warps=4 layout="wgmma_layout_0">) collective=[xy.block, z.thread] attrs={label: "mm_iter2_iter2"}
              matmul(%56: smem<64x16xbf16 strides=[64, 1] swizzle="128B">, %57: smem<256x16xbf16 strides=[64, 1] swizzle="128B">, %25: wgrmem<64x256xf32 warps=4 layout="wgmma_layout_0">) collective=[xy.block, z.thread] attrs={label: "mm_iter2_iter3"}
            }
            await(%32: lock<wgmma>)
            await(%1: lock<mbar>)
            async(%33: lock<wgmma>) attrs={label: "mm_iter3_async_iter0"} {
              matmul(%58: smem<64x16xbf16 strides=[64, 1] swizzle="128B">, %59: smem<256x16xbf16 strides=[64, 1] swizzle="128B">, %25: wgrmem<64x256xf32 warps=4 layout="wgmma_layout_0">) collective=[xy.block, z.thread] attrs={label: "mm_iter3_iter0"}
              matmul(%60: smem<64x16xbf16 strides=[64, 1] swizzle="128B">, %61: smem<256x16xbf16 strides=[64, 1] swizzle="128B">, %25: wgrmem<64x256xf32 warps=4 layout="wgmma_layout_0">) collective=[xy.block, z.thread] attrs={label: "mm_iter3_iter1"}
              matmul(%62: smem<64x16xbf16 strides=[64, 1] swizzle="128B">, %63: smem<256x16xbf16 strides=[64, 1] swizzle="128B">, %25: wgrmem<64x256xf32 warps=4 layout="wgmma_layout_0">) collective=[xy.block, z.thread] attrs={label: "mm_iter3_iter2"}
              matmul(%64: smem<64x16xbf16 strides=[64, 1] swizzle="128B">, %65: smem<256x16xbf16 strides=[64, 1] swizzle="128B">, %25: wgrmem<64x256xf32 warps=4 layout="wgmma_layout_0">) collective=[xy.block, z.thread] attrs={label: "mm_iter3_iter3"}
            }
            %78: wgrmem<64x256xbf16 warps=4 layout="wgmma_layout_0"> = alloc() attrs={id: 17}
            await(%33: lock<wgmma>)
            elementwise convert(%78: wgrmem<64x256xbf16 warps=4 layout="wgmma_layout_0">, %25: wgrmem<64x256xf32 warps=4 layout="wgmma_layout_0">)
            move(%24: gmem<64x256xbf16 strides=[4096, 1]>, %78: wgrmem<64x256xbf16 warps=4 layout="wgmma_layout_0">) attrs={label: "mm_dst_wgrmem"}
          }
        }"""))  # noqa: E501
    insert_syncs(module)

    # One named (block_leader) barrier per stage reload (WAR: the next TMA load
    # overwrites the smem stage the just-awaited matmul read), in rotating-stage
    # order. It's block_leader, not block, because the TMA write is issued by a
    # single leader thread.
    text = repr(module)
    # One publication before the first load covers all four stage locks.
    assert text.count('barrier() attrs={scope: "block"}') == 1
    assert text.count('scope: "block_leader"') == 4
    assert_contains_in_order(
        text,
        textwrap.dedent("""\
            barrier() attrs={id: 1, num_threads: 256, scope: "block_leader"}
            async(%10: lock<mbar>) attrs={label: "mm_lhs_smem_iter0_async"} {
            barrier() attrs={id: 2, num_threads: 256, scope: "block_leader"}
            async(%7: lock<mbar>) attrs={label: "mm_lhs_smem_iter1_async"} {
            barrier() attrs={id: 3, num_threads: 256, scope: "block_leader"}
            async(%4: lock<mbar>) attrs={label: "mm_lhs_smem_iter2_async"} {
            barrier() attrs={id: 4, num_threads: 256, scope: "block_leader"}
            async(%1: lock<mbar>) attrs={label: "mm_lhs_smem_iter3_async"} {"""),
    )


def test_named_barrier_id_is_reused_after_mbar_await():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @test(%a: gmem<64x64xbf16>, %b: gmem<64x64xbf16>, %c: gmem<64x64xbf16>, %d: gmem<64x64xbf16>) grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {
            %sa: smem<64x64xbf16> = alloc()
            %sb: smem<64x64xbf16> = alloc()
            %sc: smem<64x64xbf16> = alloc()
            %sd: smem<64x64xbf16> = alloc()
            %acc: wgrmem<64x64xf32 warps=4 layout="wgmma_layout_0"> = alloc()
            %acc_init: rmem<f32> = constant() attrs={value: 0.0}
            elementwise fill(%acc: wgrmem<64x64xf32 warps=4 layout="wgmma_layout_0">, %acc_init: rmem<f32>)
            %l0: lock<mbar> = alloc()
            %l1: lock<mbar> = alloc()
            %l2: lock<mbar> = alloc()
            async(%l0: lock<mbar>) {
              move(%sa: smem<64x64xbf16>, %a: gmem<64x64xbf16>) attrs={tma: True}
              move(%sb: smem<64x64xbf16>, %b: gmem<64x64xbf16>) attrs={tma: True}
              move(%sc: smem<64x64xbf16>, %c: gmem<64x64xbf16>) attrs={tma: True}
              move(%sd: smem<64x64xbf16>, %d: gmem<64x64xbf16>) attrs={tma: True}
            }
            await(%l0: lock<mbar>)
            matmul(%sa: smem<64x64xbf16>, %sb: smem<64x64xbf16>, %acc: wgrmem<64x64xf32 warps=4 layout="wgmma_layout_0">) collective=[xy.block, z.thread]
            async(%l1: lock<mbar>) {
              move(%sa: smem<64x64xbf16>, %a: gmem<64x64xbf16>) attrs={tma: True}
              move(%sb: smem<64x64xbf16>, %b: gmem<64x64xbf16>) attrs={tma: True}
            }
            await(%l1: lock<mbar>)
            matmul(%sc: smem<64x64xbf16>, %sd: smem<64x64xbf16>, %acc: wgrmem<64x64xf32 warps=4 layout="wgmma_layout_0">) collective=[xy.block, z.thread]
            async(%l2: lock<mbar>) {
              move(%sc: smem<64x64xbf16>, %c: gmem<64x64xbf16>) attrs={tma: True}
              move(%sd: smem<64x64xbf16>, %d: gmem<64x64xbf16>) attrs={tma: True}
            }
          }
        }"""))

    insert_syncs(module)
    barriers = [
        line.strip() for line in repr(module).splitlines()
        if 'scope: "block_leader"' in line
    ]
    assert barriers == [
        'barrier() attrs={id: 1, num_threads: 128, scope: "block_leader"}',
        'barrier() attrs={id: 1, num_threads: 128, scope: "block_leader"}',
    ]


def test_bulk_await_does_not_release_named_barrier_id():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @test(%d0: tma_desc<arg_idx=0 gmem<64x64xf32> tile=64x64>, %d1: tma_desc<arg_idx=1 gmem<64x64xf32> tile=64x64>) grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {
            %c0: index = index_constant() attrs={value: 0}
            %s0: smem<64x64xf32> = alloc()
            %r0: rmem<64x64xf32> = alloc()
            move(%s0: smem<64x64xf32>, %r0: rmem<64x64xf32>)
            %l0: lock<bulk> = alloc()
            async(%l0: lock<bulk>) tma_move(%d0: tma_desc<arg_idx=0 gmem<64x64xf32> tile=64x64>, %s0: smem<64x64xf32>, %c0: index, %c0: index)
            await(%l0: lock<bulk>)
            %s1: smem<64x64xf32> = alloc()
            %r1: rmem<64x64xf32> = alloc()
            move(%s1: smem<64x64xf32>, %r1: rmem<64x64xf32>)
            %l1: lock<bulk> = alloc()
            async(%l1: lock<bulk>) tma_move(%d1: tma_desc<arg_idx=1 gmem<64x64xf32> tile=64x64>, %s1: smem<64x64xf32>, %c0: index, %c0: index)
          }
        }"""))

    insert_syncs(module)
    barriers = [
        line.strip() for line in repr(module).splitlines()
        if 'scope: "block_leader"' in line
    ]
    assert barriers == [
        'barrier() attrs={id: 1, num_threads: 128, scope: "block_leader"}',
        'barrier() attrs={id: 2, num_threads: 128, scope: "block_leader"}',
    ]


def test_loop_backedge_upgrades_live_named_barrier_to_block_sync():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @test(%d: tma_desc<arg_idx=0 gmem<64x64xf32> tile=64x64>) grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {
            %c0: index = index_constant() attrs={value: 0}
            %s: smem<64x64xf32> = alloc()
            %r: rmem<64x64xf32> = alloc()
            %l: lock<bulk> = alloc()
            move(%s: smem<64x64xf32>, %r: rmem<64x64xf32>)
            for (%i: index = 0 to 2) {
              async(%l: lock<bulk>) tma_move(%d: tma_desc<arg_idx=0 gmem<64x64xf32> tile=64x64>, %s: smem<64x64xf32>, %c0: index, %c0: index) attrs={dst_tiling_scheme: 0}
              await(%l: lock<bulk>)
            }
          }
        }"""))

    insert_syncs(module)

    barriers = find_ops_of_type(module, BarrierOp)
    assert len(barriers) == 1
    assert barriers[0].scope == "block"


def test_full_block_sync_releases_all_named_barrier_ids():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @test(%d0: tma_desc<arg_idx=0 gmem<64x64xf32> tile=64x64>, %d1: tma_desc<arg_idx=1 gmem<64x64xf32> tile=64x64>) grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {
            %c0: index = index_constant() attrs={value: 0}
            %s0: smem<64x64xf32> = alloc()
            %r0: rmem<64x64xf32> = alloc()
            move(%s0: smem<64x64xf32>, %r0: rmem<64x64xf32>)
            %l0: lock<bulk> = alloc()
            async(%l0: lock<bulk>) tma_move(%d0: tma_desc<arg_idx=0 gmem<64x64xf32> tile=64x64>, %s0: smem<64x64xf32>, %c0: index, %c0: index)
            await(%l0: lock<bulk>)
            %sync_s: smem<64x64xf32> = alloc()
            %sync_r: rmem<64x64xf32> = alloc()
            move(%sync_s: smem<64x64xf32>, %sync_r: rmem<64x64xf32>)
            move(%sync_r: rmem<64x64xf32>, %sync_s: smem<64x64xf32>)
            %s1: smem<64x64xf32> = alloc()
            %r1: rmem<64x64xf32> = alloc()
            move(%s1: smem<64x64xf32>, %r1: rmem<64x64xf32>)
            %l1: lock<bulk> = alloc()
            async(%l1: lock<bulk>) tma_move(%d1: tma_desc<arg_idx=1 gmem<64x64xf32> tile=64x64>, %s1: smem<64x64xf32>, %c0: index, %c0: index)
          }
        }"""))

    insert_syncs(module)
    barriers = [
        line.strip() for line in repr(module).splitlines()
        if "barrier()" in line
    ]
    assert barriers == [
        'barrier() attrs={id: 1, num_threads: 128, scope: "block_leader"}',
        'barrier() attrs={scope: "block"}',
        'barrier() attrs={id: 1, num_threads: 128, scope: "block_leader"}',
    ]


def test_bulk_await_orders_tma_store_read_only_for_the_leader():
    # wait_group completes in t0, the thread that committed the store, so the
    # other threads still need t0's word before overwriting its source. The
    # authored barrier retires the first write, leaving only the TMA read.
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @test(%d0: tma_desc<arg_idx=0 gmem<64x64xf32> tile=64x64>) grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {
            %c0: index = index_constant() attrs={value: 0}
            %s0: smem<64x64xf32> = alloc()
            %r0: rmem<64x64xf32> = alloc()
            move(%s0: smem<64x64xf32>, %r0: rmem<64x64xf32>)
            barrier() attrs={scope: "block"}
            %l0: lock<bulk> = alloc()
            async(%l0: lock<bulk>) tma_move(%d0: tma_desc<arg_idx=0 gmem<64x64xf32> tile=64x64>, %s0: smem<64x64xf32>, %c0: index, %c0: index)
            await(%l0: lock<bulk>)
            move(%s0: smem<64x64xf32>, %r0: rmem<64x64xf32>)
          }
        }"""))

    insert_syncs(module)
    order = [
        line.strip() for line in repr(module).splitlines()
        if line.strip().startswith(("barrier()", "await(", "move("))
    ]
    assert order == [
        "move(%s0: smem<64x64xf32>, %r0: rmem<64x64xf32>)",
        'barrier() attrs={scope: "block"}',
        "await(%l0: lock<bulk>)",
        'barrier() attrs={scope: "block"}',
        "move(%s0: smem<64x64xf32>, %r0: rmem<64x64xf32>)",
    ]
