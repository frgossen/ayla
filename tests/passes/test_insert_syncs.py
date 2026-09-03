"""Tests for the cross-thread sync-barrier insertion pass."""
import textwrap

import pytest

from ayla import Block, Op
from ayla.analysis.sync import (
    BarrierScope,
    BarrierTarget,
    _barrier_covers,
    _barrier_scope,
)
from ayla.gpu_dialect import CollectiveScope, KernelOp, gpu_dialect
from ayla.passes.insert_syncs import (
    _insert_barriers,
    insert_syncs,
)
from tests.testing_utils import assert_contains_in_order

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
    # grid barrier discharges that effect, so the pass must not insert another.
    module = gpu_dialect.parse(textwrap.dedent(
        source.replace("__SCOPE__", "grid"),
    ))
    insert_syncs(module)
    kernel = module.body.ops[0]
    assert isinstance(kernel, KernelOp)
    assert kernel.requires_resident_grid
    assert repr(module).count("barrier(") == 1


def test_war_on_shared_buffer_gets_barrier():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @test(%a: gmem<128x4096xbf16>, %b: gmem<128x4096xbf16>, %acc: wrmem<128x128xf32 warps=4 layout="wgmma_layout_0">) grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {
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
              async(%lw: lock<wgmma>) matmul(%sa: smem<128x64xbf16 swizzle="128B">, %sb: smem<128x64xbf16 swizzle="128B">, %acc: wrmem<128x128xf32 warps=4 layout="wgmma_layout_0">) collective=[xy.block, z.thread]
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
          kernel @test(%a: gmem<128x4096xbf16>, %b: gmem<128x4096xbf16>, %acc: wrmem<128x128xf32 warps=4 layout="wgmma_layout_0">) grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {
            %sa: smem<128x64xbf16 swizzle="128B"> = alloc() attrs={id: 0}
            %sb: smem<128x64xbf16 swizzle="128B"> = alloc() attrs={id: 1}
            for (%k: index = 0 to 4) attrs={label: "k"} {
              %ak: gmem<128x64xbf16 strides=[4096, 1]> = tile(%a: gmem<128x4096xbf16>, 0, %k: index)
              %bk: gmem<128x64xbf16 strides=[4096, 1]> = tile(%b: gmem<128x4096xbf16>, 0, %k: index)
              %lm: lock<mbar> = alloc() attrs={id: 2}
              barrier() attrs={id: 1, num_threads: 128, scope: "block_leader"}
              async(%lm: lock<mbar>) {
                move(%sa: smem<128x64xbf16 swizzle="128B">, %ak: gmem<128x64xbf16 strides=[4096, 1]>) attrs={tma: True}
                move(%sb: smem<128x64xbf16 swizzle="128B">, %bk: gmem<128x64xbf16 strides=[4096, 1]>) attrs={tma: True}
              }
              await(%lm: lock<mbar>)
              %lw: lock<wgmma> = alloc() attrs={id: 3}
              async(%lw: lock<wgmma>) matmul(%sa: smem<128x64xbf16 swizzle="128B">, %sb: smem<128x64xbf16 swizzle="128B">, %acc: wrmem<128x128xf32 warps=4 layout="wgmma_layout_0">) collective=[xy.block, z.thread]
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
            %bx: index, %by: index, %bz: index = block_ids()
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
            %25: wrmem<64x256xf32 warps=4 layout="wgmma_layout_0"> = alloc() attrs={id: 12, init: 0.0}
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
                matmul(%34: smem<64x16xbf16 strides=[64, 1] swizzle="128B">, %35: smem<256x16xbf16 strides=[64, 1] swizzle="128B">, %25: wrmem<64x256xf32 warps=4 layout="wgmma_layout_0">) collective=[xy.block, z.thread] attrs={label: "mm_iter0_iter0"}
                matmul(%36: smem<64x16xbf16 strides=[64, 1] swizzle="128B">, %37: smem<256x16xbf16 strides=[64, 1] swizzle="128B">, %25: wrmem<64x256xf32 warps=4 layout="wgmma_layout_0">) collective=[xy.block, z.thread] attrs={label: "mm_iter0_iter1"}
                matmul(%38: smem<64x16xbf16 strides=[64, 1] swizzle="128B">, %39: smem<256x16xbf16 strides=[64, 1] swizzle="128B">, %25: wrmem<64x256xf32 warps=4 layout="wgmma_layout_0">) collective=[xy.block, z.thread] attrs={label: "mm_iter0_iter2"}
                matmul(%40: smem<64x16xbf16 strides=[64, 1] swizzle="128B">, %41: smem<256x16xbf16 strides=[64, 1] swizzle="128B">, %25: wrmem<64x256xf32 warps=4 layout="wgmma_layout_0">) collective=[xy.block, z.thread] attrs={label: "mm_iter0_iter3"}
              }
              await(%30: lock<wgmma>)
              async(%10: lock<mbar>) attrs={label: "mm_lhs_smem_iter0_async"} {
                move(%11: smem<128x64xbf16 swizzle="128B">, %76: gmem<128x64xbf16 strides=[4096, 1]>) attrs={label: "mm_lhs_smem_iter0", nc: True, tma: True}
                move(%9: smem<256x64xbf16 swizzle="128B">, %77: gmem<256x64xbf16 strides=[4096, 1]>) attrs={label: "mm_rhs_smem_iter0", nc: True, tma: True}
              }
              await(%7: lock<mbar>)
              async(%31: lock<wgmma>) attrs={label: "mm_iter1_async_iter0"} {
                matmul(%42: smem<64x16xbf16 strides=[64, 1] swizzle="128B">, %43: smem<256x16xbf16 strides=[64, 1] swizzle="128B">, %25: wrmem<64x256xf32 warps=4 layout="wgmma_layout_0">) collective=[xy.block, z.thread] attrs={label: "mm_iter1_iter0"}
                matmul(%44: smem<64x16xbf16 strides=[64, 1] swizzle="128B">, %45: smem<256x16xbf16 strides=[64, 1] swizzle="128B">, %25: wrmem<64x256xf32 warps=4 layout="wgmma_layout_0">) collective=[xy.block, z.thread] attrs={label: "mm_iter1_iter1"}
                matmul(%46: smem<64x16xbf16 strides=[64, 1] swizzle="128B">, %47: smem<256x16xbf16 strides=[64, 1] swizzle="128B">, %25: wrmem<64x256xf32 warps=4 layout="wgmma_layout_0">) collective=[xy.block, z.thread] attrs={label: "mm_iter1_iter2"}
                matmul(%48: smem<64x16xbf16 strides=[64, 1] swizzle="128B">, %49: smem<256x16xbf16 strides=[64, 1] swizzle="128B">, %25: wrmem<64x256xf32 warps=4 layout="wgmma_layout_0">) collective=[xy.block, z.thread] attrs={label: "mm_iter1_iter3"}
              }
              await(%31: lock<wgmma>)
              async(%7: lock<mbar>) attrs={label: "mm_lhs_smem_iter1_async"} {
                move(%8: smem<128x64xbf16 swizzle="128B">, %73: gmem<128x64xbf16 strides=[4096, 1]>) attrs={label: "mm_lhs_smem_iter1", nc: True, tma: True}
                move(%6: smem<256x64xbf16 swizzle="128B">, %74: gmem<256x64xbf16 strides=[4096, 1]>) attrs={label: "mm_rhs_smem_iter1", nc: True, tma: True}
              }
              await(%4: lock<mbar>)
              async(%32: lock<wgmma>) attrs={label: "mm_iter2_async_iter0"} {
                matmul(%50: smem<64x16xbf16 strides=[64, 1] swizzle="128B">, %51: smem<256x16xbf16 strides=[64, 1] swizzle="128B">, %25: wrmem<64x256xf32 warps=4 layout="wgmma_layout_0">) collective=[xy.block, z.thread] attrs={label: "mm_iter2_iter0"}
                matmul(%52: smem<64x16xbf16 strides=[64, 1] swizzle="128B">, %53: smem<256x16xbf16 strides=[64, 1] swizzle="128B">, %25: wrmem<64x256xf32 warps=4 layout="wgmma_layout_0">) collective=[xy.block, z.thread] attrs={label: "mm_iter2_iter1"}
                matmul(%54: smem<64x16xbf16 strides=[64, 1] swizzle="128B">, %55: smem<256x16xbf16 strides=[64, 1] swizzle="128B">, %25: wrmem<64x256xf32 warps=4 layout="wgmma_layout_0">) collective=[xy.block, z.thread] attrs={label: "mm_iter2_iter2"}
                matmul(%56: smem<64x16xbf16 strides=[64, 1] swizzle="128B">, %57: smem<256x16xbf16 strides=[64, 1] swizzle="128B">, %25: wrmem<64x256xf32 warps=4 layout="wgmma_layout_0">) collective=[xy.block, z.thread] attrs={label: "mm_iter2_iter3"}
              }
              await(%32: lock<wgmma>)
              async(%4: lock<mbar>) attrs={label: "mm_lhs_smem_iter2_async"} {
                move(%5: smem<128x64xbf16 swizzle="128B">, %70: gmem<128x64xbf16 strides=[4096, 1]>) attrs={label: "mm_lhs_smem_iter2", nc: True, tma: True}
                move(%3: smem<256x64xbf16 swizzle="128B">, %71: gmem<256x64xbf16 strides=[4096, 1]>) attrs={label: "mm_rhs_smem_iter2", nc: True, tma: True}
              }
              await(%1: lock<mbar>)
              async(%33: lock<wgmma>) attrs={label: "mm_iter3_async_iter0"} {
                matmul(%58: smem<64x16xbf16 strides=[64, 1] swizzle="128B">, %59: smem<256x16xbf16 strides=[64, 1] swizzle="128B">, %25: wrmem<64x256xf32 warps=4 layout="wgmma_layout_0">) collective=[xy.block, z.thread] attrs={label: "mm_iter3_iter0"}
                matmul(%60: smem<64x16xbf16 strides=[64, 1] swizzle="128B">, %61: smem<256x16xbf16 strides=[64, 1] swizzle="128B">, %25: wrmem<64x256xf32 warps=4 layout="wgmma_layout_0">) collective=[xy.block, z.thread] attrs={label: "mm_iter3_iter1"}
                matmul(%62: smem<64x16xbf16 strides=[64, 1] swizzle="128B">, %63: smem<256x16xbf16 strides=[64, 1] swizzle="128B">, %25: wrmem<64x256xf32 warps=4 layout="wgmma_layout_0">) collective=[xy.block, z.thread] attrs={label: "mm_iter3_iter2"}
                matmul(%64: smem<64x16xbf16 strides=[64, 1] swizzle="128B">, %65: smem<256x16xbf16 strides=[64, 1] swizzle="128B">, %25: wrmem<64x256xf32 warps=4 layout="wgmma_layout_0">) collective=[xy.block, z.thread] attrs={label: "mm_iter3_iter3"}
              }
              await(%33: lock<wgmma>)
              async(%1: lock<mbar>) attrs={label: "mm_lhs_smem_iter3_async"} {
                move(%2: smem<128x64xbf16 swizzle="128B">, %67: gmem<128x64xbf16 strides=[4096, 1]>) attrs={label: "mm_lhs_smem_iter3", nc: True, tma: True}
                move(%0: smem<256x64xbf16 swizzle="128B">, %68: gmem<256x64xbf16 strides=[4096, 1]>) attrs={label: "mm_rhs_smem_iter3", nc: True, tma: True}
              }
            }
            await(%10: lock<mbar>)
            async(%30: lock<wgmma>) attrs={label: "mm_iter0_async_iter0"} {
              matmul(%34: smem<64x16xbf16 strides=[64, 1] swizzle="128B">, %35: smem<256x16xbf16 strides=[64, 1] swizzle="128B">, %25: wrmem<64x256xf32 warps=4 layout="wgmma_layout_0">) collective=[xy.block, z.thread] attrs={label: "mm_iter0_iter0"}
              matmul(%36: smem<64x16xbf16 strides=[64, 1] swizzle="128B">, %37: smem<256x16xbf16 strides=[64, 1] swizzle="128B">, %25: wrmem<64x256xf32 warps=4 layout="wgmma_layout_0">) collective=[xy.block, z.thread] attrs={label: "mm_iter0_iter1"}
              matmul(%38: smem<64x16xbf16 strides=[64, 1] swizzle="128B">, %39: smem<256x16xbf16 strides=[64, 1] swizzle="128B">, %25: wrmem<64x256xf32 warps=4 layout="wgmma_layout_0">) collective=[xy.block, z.thread] attrs={label: "mm_iter0_iter2"}
              matmul(%40: smem<64x16xbf16 strides=[64, 1] swizzle="128B">, %41: smem<256x16xbf16 strides=[64, 1] swizzle="128B">, %25: wrmem<64x256xf32 warps=4 layout="wgmma_layout_0">) collective=[xy.block, z.thread] attrs={label: "mm_iter0_iter3"}
            }
            await(%30: lock<wgmma>)
            await(%7: lock<mbar>)
            async(%31: lock<wgmma>) attrs={label: "mm_iter1_async_iter0"} {
              matmul(%42: smem<64x16xbf16 strides=[64, 1] swizzle="128B">, %43: smem<256x16xbf16 strides=[64, 1] swizzle="128B">, %25: wrmem<64x256xf32 warps=4 layout="wgmma_layout_0">) collective=[xy.block, z.thread] attrs={label: "mm_iter1_iter0"}
              matmul(%44: smem<64x16xbf16 strides=[64, 1] swizzle="128B">, %45: smem<256x16xbf16 strides=[64, 1] swizzle="128B">, %25: wrmem<64x256xf32 warps=4 layout="wgmma_layout_0">) collective=[xy.block, z.thread] attrs={label: "mm_iter1_iter1"}
              matmul(%46: smem<64x16xbf16 strides=[64, 1] swizzle="128B">, %47: smem<256x16xbf16 strides=[64, 1] swizzle="128B">, %25: wrmem<64x256xf32 warps=4 layout="wgmma_layout_0">) collective=[xy.block, z.thread] attrs={label: "mm_iter1_iter2"}
              matmul(%48: smem<64x16xbf16 strides=[64, 1] swizzle="128B">, %49: smem<256x16xbf16 strides=[64, 1] swizzle="128B">, %25: wrmem<64x256xf32 warps=4 layout="wgmma_layout_0">) collective=[xy.block, z.thread] attrs={label: "mm_iter1_iter3"}
            }
            await(%31: lock<wgmma>)
            await(%4: lock<mbar>)
            async(%32: lock<wgmma>) attrs={label: "mm_iter2_async_iter0"} {
              matmul(%50: smem<64x16xbf16 strides=[64, 1] swizzle="128B">, %51: smem<256x16xbf16 strides=[64, 1] swizzle="128B">, %25: wrmem<64x256xf32 warps=4 layout="wgmma_layout_0">) collective=[xy.block, z.thread] attrs={label: "mm_iter2_iter0"}
              matmul(%52: smem<64x16xbf16 strides=[64, 1] swizzle="128B">, %53: smem<256x16xbf16 strides=[64, 1] swizzle="128B">, %25: wrmem<64x256xf32 warps=4 layout="wgmma_layout_0">) collective=[xy.block, z.thread] attrs={label: "mm_iter2_iter1"}
              matmul(%54: smem<64x16xbf16 strides=[64, 1] swizzle="128B">, %55: smem<256x16xbf16 strides=[64, 1] swizzle="128B">, %25: wrmem<64x256xf32 warps=4 layout="wgmma_layout_0">) collective=[xy.block, z.thread] attrs={label: "mm_iter2_iter2"}
              matmul(%56: smem<64x16xbf16 strides=[64, 1] swizzle="128B">, %57: smem<256x16xbf16 strides=[64, 1] swizzle="128B">, %25: wrmem<64x256xf32 warps=4 layout="wgmma_layout_0">) collective=[xy.block, z.thread] attrs={label: "mm_iter2_iter3"}
            }
            await(%32: lock<wgmma>)
            await(%1: lock<mbar>)
            async(%33: lock<wgmma>) attrs={label: "mm_iter3_async_iter0"} {
              matmul(%58: smem<64x16xbf16 strides=[64, 1] swizzle="128B">, %59: smem<256x16xbf16 strides=[64, 1] swizzle="128B">, %25: wrmem<64x256xf32 warps=4 layout="wgmma_layout_0">) collective=[xy.block, z.thread] attrs={label: "mm_iter3_iter0"}
              matmul(%60: smem<64x16xbf16 strides=[64, 1] swizzle="128B">, %61: smem<256x16xbf16 strides=[64, 1] swizzle="128B">, %25: wrmem<64x256xf32 warps=4 layout="wgmma_layout_0">) collective=[xy.block, z.thread] attrs={label: "mm_iter3_iter1"}
              matmul(%62: smem<64x16xbf16 strides=[64, 1] swizzle="128B">, %63: smem<256x16xbf16 strides=[64, 1] swizzle="128B">, %25: wrmem<64x256xf32 warps=4 layout="wgmma_layout_0">) collective=[xy.block, z.thread] attrs={label: "mm_iter3_iter2"}
              matmul(%64: smem<64x16xbf16 strides=[64, 1] swizzle="128B">, %65: smem<256x16xbf16 strides=[64, 1] swizzle="128B">, %25: wrmem<64x256xf32 warps=4 layout="wgmma_layout_0">) collective=[xy.block, z.thread] attrs={label: "mm_iter3_iter3"}
            }
            %78: wrmem<64x256xbf16 warps=4 layout="wgmma_layout_0"> = alloc() attrs={id: 17}
            await(%33: lock<wgmma>)
            elementwise convert(%78: wrmem<64x256xbf16 warps=4 layout="wgmma_layout_0">, %25: wrmem<64x256xf32 warps=4 layout="wgmma_layout_0">)
            move(%24: gmem<64x256xbf16 strides=[4096, 1]>, %78: wrmem<64x256xbf16 warps=4 layout="wgmma_layout_0">) attrs={label: "mm_dst_wrmem"}
          }
        }"""))  # noqa: E501
    insert_syncs(module)

    # One named (block_leader) barrier per stage reload (WAR: the next TMA load
    # overwrites the smem stage the just-awaited matmul read), in rotating-stage
    # order. It's block_leader, not block, because the TMA write is issued by a
    # single leader thread.
    text = repr(module)
    assert text.count("barrier()") == 4
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
            %acc: wrmem<64x64xf32 warps=4 layout="wgmma_layout_0"> = alloc() attrs={init: 0.0}
            %l0: lock<mbar> = alloc()
            async(%l0: lock<mbar>) {
              move(%sa: smem<64x64xbf16>, %a: gmem<64x64xbf16>) attrs={tma: True}
              move(%sb: smem<64x64xbf16>, %b: gmem<64x64xbf16>) attrs={tma: True}
              move(%sc: smem<64x64xbf16>, %c: gmem<64x64xbf16>) attrs={tma: True}
              move(%sd: smem<64x64xbf16>, %d: gmem<64x64xbf16>) attrs={tma: True}
            }
            await(%l0: lock<mbar>)
            matmul(%sa: smem<64x64xbf16>, %sb: smem<64x64xbf16>, %acc: wrmem<64x64xf32 warps=4 layout="wgmma_layout_0">) collective=[xy.block, z.thread]
            %l1: lock<mbar> = alloc()
            async(%l1: lock<mbar>) {
              move(%sa: smem<64x64xbf16>, %a: gmem<64x64xbf16>) attrs={tma: True}
              move(%sb: smem<64x64xbf16>, %b: gmem<64x64xbf16>) attrs={tma: True}
            }
            await(%l1: lock<mbar>)
            matmul(%sc: smem<64x64xbf16>, %sd: smem<64x64xbf16>, %acc: wrmem<64x64xf32 warps=4 layout="wgmma_layout_0">) collective=[xy.block, z.thread]
            %l2: lock<mbar> = alloc()
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
