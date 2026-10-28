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


import torch

from ayla.gpu_dialect import gpu_dialect
from ayla.passes.licm import licm
from ayla.ptx_codegen import make_ptx_kernel
from ayla.ptx_runtime import make_ptx_launcher
from tests.testing_utils import requires_cuda



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


def test_hoist_alloc_and_retain_fill():
    """Storage is hoisted while its fill remains in the loop."""
    module_str = textwrap.dedent("""\
        module {
          kernel @test(%a: gmem<128x64xbf16>) grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {
            for (%k: index = 0 to 64) {
              %s: smem<128x64xbf16 swizzle="128B"> = alloc() attrs={id: 0}
              %r: rmem<128x64xbf16> = alloc() attrs={id: 1}
              %zero: rmem<bf16> = constant() attrs={value: 0.0}
              elementwise fill(%r: rmem<128x64xbf16>, %zero: rmem<bf16>)
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
            %r: rmem<128x64xbf16> = alloc() attrs={id: 1}
            %zero: rmem<bf16> = constant() attrs={value: 0.0}
            for (%k: index = 0 to 64) {
              elementwise fill(%r: rmem<128x64xbf16>, %zero: rmem<bf16>)
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


@requires_cuda
def test_hoisted_accumulator_refills_each_iteration_on_gpu():
    module = gpu_dialect.parse_module(textwrap.dedent("""\
        module {
          kernel @refill(%out: gmem<3xf32>) grid_dim=[1, 1, 1] block_dim=[1, 1, 1] {
            %overwrite: rmem<f32> = constant() attrs={value: 99.0}
            for (%i: index = 0 to 3) {
              %acc: rmem<1xf32> = alloc() attrs={id: 7}
              %value: rmem<f32> = constant() attrs={value: 2.5}
              elementwise fill(%acc: rmem<1xf32>, %value: rmem<f32>)
              %tile: gmem<1xf32 strides=[1]> = tile(%out: gmem<3xf32>, %i: index)
              move(%tile: gmem<1xf32 strides=[1]>, %acc: rmem<1xf32>) attrs={dst_tiling_scheme: 0}
              elementwise fill(%acc: rmem<1xf32>, %overwrite: rmem<f32>)
            }
          }
        }"""))
    licm(module)
    [loop] = [op for op in module.body.ops[0].body.ops if op.op_id == "for"]
    assert [op.op_id for op in loop.body.ops].count("alloc") == 0
    out = torch.full((3,), -1.0, device="cuda")
    make_ptx_launcher(make_ptx_kernel(module))(out)
    torch.testing.assert_close(out, torch.full((3,), 2.5, device="cuda"))
