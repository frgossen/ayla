"""Tests for the LICM (loop-invariant code motion) pass."""
import textwrap

from ayla.base.ir import find_ops_of_type
from ayla.gpu_dialect import (
    AllocOp,
    IfOp,
    KernelOp,
    LockType,
    RmemTensorType,
    SmemTensorType,
    gpu_dialect,
)
from ayla.passes.licm import licm



def test_no_hoist_k_dependent_ops():
    """Ops that depend on the loop IV stay inside."""
    module_str = textwrap.dedent("""\
        module {
          kernel @test(%a: gmem<128x4096xbf16>, %s: smem<128x64xbf16 swizzle="128B">) grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {
            for (%k: index = 0 to 64) {
              %t: gmem<128x64xbf16 strides=[4096, 1]> = tile(%a: gmem<128x4096xbf16>, 0, %k: index)
              move(%s: smem<128x64xbf16 swizzle="128B">, %t: gmem<128x64xbf16 strides=[4096, 1]>) attrs={tma: True}
            }
          }
        }""")  # noqa: E501

    module = gpu_dialect.parse_module(module_str)
    before = repr(module)
    module = licm(module)
    after = repr(module)
    assert before == after


def test_hoist_alloc_without_init():
    """Static allocs (no init) are hoisted; allocs with init stay."""
    module_str = textwrap.dedent("""\
        module {
          kernel @test(%a: gmem<128x64xbf16>) grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {
            for (%k: index = 0 to 64) {
              %s: smem<128x64xbf16 swizzle="128B"> = alloc() attrs={id: 0}
              %r: rmem<128x64xbf16> = alloc() attrs={id: 1, init: 0.0}
              move(%s: smem<128x64xbf16 swizzle="128B">, %a: gmem<128x64xbf16>) attrs={tma: True}
            }
          }
        }""")

    module = gpu_dialect.parse_module(module_str)
    module = licm(module)
    text = repr(module)

    expected = textwrap.dedent("""\
        module {
          kernel @test(%a: gmem<128x64xbf16>) grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {
            %s: smem<128x64xbf16 swizzle="128B"> = alloc() attrs={id: 0}
            for (%k: index = 0 to 64) {
              %r: rmem<128x64xbf16> = alloc() attrs={id: 1, init: 0.0}
              move(%s: smem<128x64xbf16 swizzle="128B">, %a: gmem<128x64xbf16>) attrs={tma: True}
            }
          }
        }""")
    assert text == expected


def test_do_not_hoist_alloc_out_of_top_level_if():
    module = gpu_dialect.parse_module(textwrap.dedent("""\
        module {
          kernel @test(%a: gmem<64x64xbf16>) grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {
            %zero: index = index_constant() attrs={value: 0}
            %one: index = index_constant() attrs={value: 1}
            %condition: pred = index_lt(%zero: index, %one: index)
            if(%condition: pred) {
              %s: smem<64x64xbf16> = alloc() attrs={id: 0}
              %r: rmem<64x64xbf16> = alloc() attrs={id: 1, init: 0.0}
              move(%s: smem<64x64xbf16>, %a: gmem<64x64xbf16>) attrs={tma: True}
            }
          }
        }"""))

    module = licm(module)

    [guard] = find_ops_of_type(module, IfOp)
    smem, rmem = find_ops_of_type(module, AllocOp)
    assert smem.parent_block is guard.body
    assert rmem.parent_block is guard.body


def test_hoist_uninitialized_smem_through_loop_guard():
    module = gpu_dialect.parse_module(textwrap.dedent("""\
        module {
          kernel @test(%a: gmem<64x64xbf16>) grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {
            %zero: index = index_constant() attrs={value: 0}
            %one: index = index_constant() attrs={value: 1}
            %condition: pred = index_lt(%zero: index, %one: index)
            for (%k: index = 0 to 2) {
              if(%condition: pred) {
                %s: smem<64x64xbf16> = alloc() attrs={id: 0}
                %r: rmem<64x64xbf16> = alloc() attrs={id: 1, init: 0.0}
                %lock: lock<mbar> = alloc() attrs={id: 2}
                move(%s: smem<64x64xbf16>, %a: gmem<64x64xbf16>) attrs={tma: True}
              }
            }
          }
        }"""))

    # One pass crosses the guard; the second crosses the enclosing loop.
    module = licm(licm(module))

    [guard] = find_ops_of_type(module, IfOp)
    [kernel] = module.body.ops
    assert isinstance(kernel, KernelOp)
    smem = next(
        alloc for alloc in find_ops_of_type(module, AllocOp)
        if isinstance(alloc.result().type, SmemTensorType)
    )
    rmem = next(
        alloc for alloc in find_ops_of_type(module, AllocOp)
        if isinstance(alloc.result().type, RmemTensorType)
    )
    lock = next(
        alloc for alloc in find_ops_of_type(module, AllocOp)
        if isinstance(alloc.result().type, LockType)
    )
    assert smem.parent_block is kernel.body
    assert rmem.parent_block is guard.body
    assert lock.parent_block is guard.body
