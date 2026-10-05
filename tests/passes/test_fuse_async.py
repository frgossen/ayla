"""Tests for the fuse_async pass."""
import textwrap

import pytest

from ayla.gpu_dialect import gpu_dialect
from ayla.passes.fuse_async import fuse_async


def test_fuse_same_accumulator():
    """Matmuls writing to the same accumulator fuse — wgmma guarantees
    ordering within a commit group."""
    module_str = textwrap.dedent("""\
        module {
          kernel @test(%a: smem<128x16xbf16 swizzle="128B">, %b: smem<128x16xbf16 swizzle="128B">, %c: wgrmem<128x128xf32 warps=4 layout="wgmma_layout_0">) grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {
            %l0: lock<wgmma> = alloc() attrs={id: 0}
            async(%l0: lock<wgmma>) matmul(%a: smem<128x16xbf16 swizzle="128B">, %b: smem<128x16xbf16 swizzle="128B">, %c: wgrmem<128x128xf32 warps=4 layout="wgmma_layout_0">) collective=[xy.block, z.thread]
            await(%l0: lock<wgmma>)
            %l1: lock<wgmma> = alloc() attrs={id: 1}
            async(%l1: lock<wgmma>) matmul(%a: smem<128x16xbf16 swizzle="128B">, %b: smem<128x16xbf16 swizzle="128B">, %c: wgrmem<128x128xf32 warps=4 layout="wgmma_layout_0">) collective=[xy.block, z.thread]
            await(%l1: lock<wgmma>)
          }
        }""")

    module = gpu_dialect.parse_module(module_str)
    module = fuse_async(module)
    text = repr(module)

    expected = textwrap.dedent("""\
        module {
          kernel @test(%a: smem<128x16xbf16 swizzle="128B">, %b: smem<128x16xbf16 swizzle="128B">, %c: wgrmem<128x128xf32 warps=4 layout="wgmma_layout_0">) grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {
            %l0: lock<wgmma> = alloc() attrs={id: 0}
            %l1: lock<wgmma> = alloc() attrs={id: 1}
            async(%l0: lock<wgmma>) {
              matmul(%a: smem<128x16xbf16 swizzle="128B">, %b: smem<128x16xbf16 swizzle="128B">, %c: wgrmem<128x128xf32 warps=4 layout="wgmma_layout_0">) collective=[xy.block, z.thread]
              matmul(%a: smem<128x16xbf16 swizzle="128B">, %b: smem<128x16xbf16 swizzle="128B">, %c: wgrmem<128x128xf32 warps=4 layout="wgmma_layout_0">) collective=[xy.block, z.thread]
            }
            await(%l0: lock<wgmma>)
          }
        }""")
    assert text == expected


def test_fuse_tma_loads():
    """Consecutive TMA loads (mbar) fuse onto one mbarrier + one await."""
    module_str = textwrap.dedent("""\
        module {
          kernel @test(%a: gmem<128x4096xbf16>, %b: gmem<128x4096xbf16>) grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {
            %s0: smem<128x64xbf16 swizzle="128B"> = alloc() attrs={id: 0}
            %s1: smem<128x64xbf16 swizzle="128B"> = alloc() attrs={id: 1}
            %c0: index = index_constant() attrs={value: 0}
            %a0: gmem<128x64xbf16 strides=[4096, 1]> = tile(%a: gmem<128x4096xbf16>, 0, %c0: index)
            %b0: gmem<128x64xbf16 strides=[4096, 1]> = tile(%b: gmem<128x4096xbf16>, 0, %c0: index)
            %l0: lock<mbar> = alloc() attrs={id: 2}
            async(%l0: lock<mbar>) collective=xyz.block move(%s0: smem<128x64xbf16 swizzle="128B">, %a0: gmem<128x64xbf16 strides=[4096, 1]>) collective=xyz.block attrs={label: "mm_lhs_smem", nc: True, tma: True}
            await(%l0: lock<mbar>)
            %l1: lock<mbar> = alloc() attrs={id: 3}
            async(%l1: lock<mbar>) collective=xyz.block move(%s1: smem<128x64xbf16 swizzle="128B">, %b0: gmem<128x64xbf16 strides=[4096, 1]>) collective=xyz.block attrs={label: "mm_rhs_smem", nc: True, tma: True}
            await(%l1: lock<mbar>)
          }
        }""")

    module = gpu_dialect.parse_module(module_str)
    module = fuse_async(module)
    text = repr(module)

    expected = textwrap.dedent("""\
        module {
          kernel @test(%a: gmem<128x4096xbf16>, %b: gmem<128x4096xbf16>) grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {
            %s0: smem<128x64xbf16 swizzle="128B"> = alloc() attrs={id: 0}
            %s1: smem<128x64xbf16 swizzle="128B"> = alloc() attrs={id: 1}
            %c0: index = index_constant() attrs={value: 0}
            %a0: gmem<128x64xbf16 strides=[4096, 1]> = tile(%a: gmem<128x4096xbf16>, 0, %c0: index)
            %b0: gmem<128x64xbf16 strides=[4096, 1]> = tile(%b: gmem<128x4096xbf16>, 0, %c0: index)
            %l0: lock<mbar> = alloc() attrs={id: 2}
            %l1: lock<mbar> = alloc() attrs={id: 3}
            async(%l0: lock<mbar>) collective=xyz.block {
              move(%s0: smem<128x64xbf16 swizzle="128B">, %a0: gmem<128x64xbf16 strides=[4096, 1]>) collective=xyz.block attrs={label: "mm_lhs_smem", nc: True, tma: True}
              move(%s1: smem<128x64xbf16 swizzle="128B">, %b0: gmem<128x64xbf16 strides=[4096, 1]>) collective=xyz.block attrs={label: "mm_rhs_smem", nc: True, tma: True}
            }
            await(%l0: lock<mbar>)
          }
        }""")
    assert text == expected


@pytest.mark.parametrize("second_lock_kind", ["bulk"])
def test_does_not_fuse_incompatible_tma_loads(
    second_lock_kind,
):
    module_str = textwrap.dedent(f"""\
        module {{
          kernel @test(%a: gmem<16x16xf32>, %b: gmem<16x16xf32>) grid_dim=[2, 1, 1] block_dim=[32, 1, 1] cluster_dim=[2, 1, 1] {{
            %sa: smem<16x16xf32> = alloc()
            %sb: smem<16x16xf32> = alloc()
            %l0: lock<mbar> = alloc()
            %l1: lock<{second_lock_kind}> = alloc()
            async(%l0: lock<mbar>) collective=xyz.block move(%sa: smem<16x16xf32>, %a: gmem<16x16xf32>) collective=xyz.block attrs={{tma: True}}
            await(%l0: lock<mbar>)
            async(%l1: lock<{second_lock_kind}>) collective=xyz.block move(%sb: smem<16x16xf32>, %b: gmem<16x16xf32>) collective=xyz.block attrs={{tma: True}}
            await(%l1: lock<{second_lock_kind}>)
          }}
        }}""")

    module = fuse_async(gpu_dialect.parse_module(module_str))

    assert repr(module) == module_str


def test_fuses_launches_whose_awaits_follow_the_run():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @test(%a: gmem<16x16xf32>, %b: gmem<16x16xf32>) grid_dim=[1, 1, 1] block_dim=[32, 1, 1] {
            %sa: smem<16x16xf32> = alloc()
            %sb: smem<16x16xf32> = alloc()
            %l0: lock<mbar> = alloc()
            %l1: lock<mbar> = alloc()
            async(%l0: lock<mbar>) collective=xyz.block move(%sa: smem<16x16xf32>, %a: gmem<16x16xf32>) collective=xyz.block attrs={tma: True}
            async(%l1: lock<mbar>) collective=xyz.block move(%sb: smem<16x16xf32>, %b: gmem<16x16xf32>) collective=xyz.block attrs={tma: True}
            await(%l0: lock<mbar>)
            await(%l1: lock<mbar>)
          }
        }"""))

    text = repr(fuse_async(module))

    assert text.count("async(") == 1
    assert text.count("await(") == 1


@pytest.mark.parametrize("interleaved", [True, False])
def test_fuses_local_and_multicast_tma_loads_at_joined_collective(interleaved):
    local = (
        "async(%local: lock<mbar>) collective=xyz.block move(%sa: smem<16x16xf32>, %a: gmem<16x16xf32>) collective=xyz.block attrs={tma: True}",
        "await(%local: lock<mbar>)",
    )
    multicast = (
        "async(%multicast: lock<mbar>) collective=xyz.cluster move(%sb: smem<16x16xf32>, %b: gmem<16x16xf32>) collective=xyz.cluster attrs={multicast: True, tma: True}",
        "await(%multicast: lock<mbar>)",
    )
    body = (
        [*local, *multicast] if interleaved
        else [local[0], multicast[0], local[1], multicast[1]]
    )
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @test(%a: gmem<16x16xf32>, %b: gmem<16x16xf32>) grid_dim=[2, 1, 1] block_dim=[32, 1, 1] cluster_dim=[2, 1, 1] {
            %sa: smem<16x16xf32> = alloc()
            %sb: smem<16x16xf32> = alloc()
            %local: lock<mbar> = alloc()
            %multicast: lock<mbar> = alloc()
        """) + "".join(f"    {line}\n" for line in body) + "  }\n}")

    text = repr(fuse_async(module))

    assert text.count("async(") == 1
    assert text.count("await(") == 1
    assert "async(%local: lock<mbar>) collective=xyz.cluster {" in text
    assert "collective=xyz.block attrs={tma: True}" in text
    assert "collective=xyz.cluster attrs={multicast: True, tma: True}" in text


def test_rerun_ignores_a_pipelined_singleton_async():
    module_str = textwrap.dedent("""\
        module {
          kernel @test(%a: gmem<16x16xf32>, %b: gmem<16x16xf32>) grid_dim=[1, 1, 1] block_dim=[32, 1, 1] {
            %sa: smem<16x16xf32> = alloc()
            %sb: smem<16x16xf32> = alloc()
            %pipelined: lock<mbar> = alloc()
            %fresh: lock<mbar> = alloc()
            await(%pipelined: lock<mbar>)
            async(%pipelined: lock<mbar>) collective=xyz.block move(%sa: smem<16x16xf32>, %a: gmem<16x16xf32>) collective=xyz.block attrs={tma: True}
            async(%fresh: lock<mbar>) collective=xyz.block move(%sb: smem<16x16xf32>, %b: gmem<16x16xf32>) collective=xyz.block attrs={tma: True}
            await(%fresh: lock<mbar>)
          }
        }""")

    module = fuse_async(gpu_dialect.parse(module_str))

    assert repr(module) == module_str


def test_still_rejects_truly_unawaited_fusible_async():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @test(%a: gmem<16x16xf32>) grid_dim=[1, 1, 1] block_dim=[32, 1, 1] {
            %s: smem<16x16xf32> = alloc()
            %lock: lock<mbar> = alloc()
            async(%lock: lock<mbar>) collective=xyz.block move(%s: smem<16x16xf32>, %a: gmem<16x16xf32>) collective=xyz.block attrs={tma: True}
          }
        }"""))

    with pytest.raises(AssertionError, match="no adjacent await"):
        fuse_async(module)


def test_fuse_different_accumulators():
    """Matmuls writing to different accumulators can fuse."""
    module_str = textwrap.dedent("""\
        module {
          kernel @test(%a: smem<128x16xbf16 swizzle="128B">, %b: smem<128x16xbf16 swizzle="128B">, %c0: wgrmem<128x128xf32 warps=4 layout="wgmma_layout_0">, %c1: wgrmem<128x128xf32 warps=4 layout="wgmma_layout_0">) grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {
            %l0: lock<wgmma> = alloc() attrs={id: 0}
            async(%l0: lock<wgmma>) matmul(%a: smem<128x16xbf16 swizzle="128B">, %b: smem<128x16xbf16 swizzle="128B">, %c0: wgrmem<128x128xf32 warps=4 layout="wgmma_layout_0">) collective=[xy.block, z.thread]
            await(%l0: lock<wgmma>)
            %l1: lock<wgmma> = alloc() attrs={id: 1}
            async(%l1: lock<wgmma>) matmul(%a: smem<128x16xbf16 swizzle="128B">, %b: smem<128x16xbf16 swizzle="128B">, %c1: wgrmem<128x128xf32 warps=4 layout="wgmma_layout_0">) collective=[xy.block, z.thread]
            await(%l1: lock<wgmma>)
          }
        }""")

    module = gpu_dialect.parse_module(module_str)
    module = fuse_async(module)
    text = repr(module)

    expected = textwrap.dedent("""\
        module {
          kernel @test(%a: smem<128x16xbf16 swizzle="128B">, %b: smem<128x16xbf16 swizzle="128B">, %c0: wgrmem<128x128xf32 warps=4 layout="wgmma_layout_0">, %c1: wgrmem<128x128xf32 warps=4 layout="wgmma_layout_0">) grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {
            %l0: lock<wgmma> = alloc() attrs={id: 0}
            %l1: lock<wgmma> = alloc() attrs={id: 1}
            async(%l0: lock<wgmma>) {
              matmul(%a: smem<128x16xbf16 swizzle="128B">, %b: smem<128x16xbf16 swizzle="128B">, %c0: wgrmem<128x128xf32 warps=4 layout="wgmma_layout_0">) collective=[xy.block, z.thread]
              matmul(%a: smem<128x16xbf16 swizzle="128B">, %b: smem<128x16xbf16 swizzle="128B">, %c1: wgrmem<128x128xf32 warps=4 layout="wgmma_layout_0">) collective=[xy.block, z.thread]
            }
            await(%l0: lock<wgmma>)
          }
        }""")
    assert text == expected
