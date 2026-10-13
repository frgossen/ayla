"""Tests for the LICM (loop-invariant code motion) pass."""
import textwrap

from ayla.base.ir import find_ops_of_type
import pytest
import torch

from ayla.gpu_dialect import (
    AllocOp,
    CollectiveScope,
    ConstantOp,
    ForOp,
    IfOp,
    IndexConstantOp,
    KernelOp,
    MoveOp,
    PTXTemplateOp,
    RmemTensorType,
    gpu_dialect,
    is_side_effect_free,
    memory_effects,
)
from ayla.pipelines.canonicalize import Canonicalize
from ayla.passes.lower_to_ptx_template import lower_to_ptx_template
from ayla.ptx_codegen import make_ptx_kernel
from ayla.ptx_runtime import make_ptx_launcher
from tests.testing_utils import requires_cuda
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


def test_hoist_alloc_and_retain_reset():
    """Storage is hoisted while initialization remains in the loop."""
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
            %r: rmem<128x64xbf16> = alloc() attrs={id: 1}
            %0: rmem<bf16> = constant() attrs={value: 0.0}
            for (%k: index = 0 to 64) {
              move(%r: rmem<128x64xbf16>, %0: rmem<bf16>)
              move(%s: smem<128x64xbf16 swizzle="128B">, %a: gmem<128x64xbf16>) attrs={tma: True}
            }
          }
        }""")
    assert text == expected


@pytest.mark.parametrize("typ", [
    "rmem<2xf32>", "rmem<2xf64>", "rmem<2xbf16>", "rmem<2xf16>",
    'wgrmem<64x64xf32 warps=4 layout="wgmma_layout_0">',
    'wgrmem<64x64xbf16 warps=4 layout="wgmma_layout_0">',
    'wgrmem<64x64xf16 warps=4 layout="wgmma_layout_0">',
])
@pytest.mark.parametrize("init", [0.0, 2.5])
def test_hoist_initialized_storage_from_nested_guard(typ, init):
    module = gpu_dialect.parse_module(textwrap.dedent(f"""\
        module {{
          kernel @test() grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {{
            for (%i: index = 0 to 2) {{
              for (%j: index = 0 to 3) {{
                %condition: pred = index_lt(%i: index, %j: index)
                if(%condition: pred) {{
                  %acc: {typ} = alloc() attrs={{id: 7, init: {init}, label: "acc"}}
                }}
              }}
            }}
          }}
        }}"""))
    [alloc] = find_ops_of_type(module, AllocOp)
    original_effects = memory_effects(alloc)
    licm(module)

    [kernel] = module.body.ops
    assert isinstance(kernel, KernelOp)
    [guard] = find_ops_of_type(module, IfOp)
    [reset] = find_ops_of_type(module, MoveOp)
    [constant] = find_ops_of_type(module, ConstantOp)
    assert alloc.parent_block is kernel.body
    assert constant.parent_block is kernel.body
    assert alloc.attrs == {"id": 7, "label": "acc"}
    assert reset.parent_block is guard.body
    assert reset.dst is alloc.result()
    assert reset.src is constant.result()
    scalar_type = constant.result().type
    assert isinstance(scalar_type, RmemTensorType)
    assert scalar_type.rank == 0
    assert constant.value == init
    effects = memory_effects(reset)
    assert effects is not None
    assert [effect for effect in effects if effect.kind == "W"] == original_effects
    assert not is_side_effect_free(reset)

    before = repr(module)
    assert repr(licm(module)) == before
    assert repr(gpu_dialect.parse_module(before)) == before
    Canonicalize().run(module)
    assert find_ops_of_type(module, MoveOp) == [reset]


@pytest.mark.parametrize("typ, suffix, count", [
    ("rmem<2xf32>", "f32", 2),
    ("rmem<2xf64>", "f64", 2),
    ("rmem<2xbf16>", "b16", 2),
    ("rmem<2xf16>", "b16", 2),
    ('wgrmem<64x64xf32 warps=4 layout="wgmma_layout_0">', "f32", 32),
    ('wgrmem<64x64xbf16 warps=4 layout="wgmma_layout_0">', "b32", 16),
    ('wgrmem<64x64xf16 warps=4 layout="wgmma_layout_0">', "b32", 16),
])
def test_broadcast_reset_lowers_at_original_loop_position(typ, suffix, count):
    module = gpu_dialect.parse_module(textwrap.dedent(f"""\
        module {{
          kernel @test() grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {{
            for (%i: index = 0 to 3) {{
              %acc: {typ} = alloc() attrs={{id: 7, init: 2.5}}
            }}
          }}
        }}"""))
    licm(module)
    [reset] = find_ops_of_type(module, MoveOp)
    effects = memory_effects(reset)
    assert effects is not None
    [effect] = [effect for effect in effects if effect.kind == "W"]
    assert effect.kind == "W"
    assert effect.scope == CollectiveScope.THREAD
    lower_to_ptx_template(module)
    [loop] = find_ops_of_type(module, ForOp)
    [reset] = find_ops_of_type(loop, PTXTemplateOp)
    assert reset.template.count(f"mov.{suffix}") == count
    assert not reset.results
    storage, scalar = reset.operands
    assert isinstance(storage.definition, PTXTemplateOp)
    assert storage.definition.parent_block is loop.parent_block
    assert isinstance(scalar.definition, PTXTemplateOp)
    assert scalar.definition.parent_block is loop.parent_block


def test_initialized_shared_memory_is_not_split():
    module = gpu_dialect.parse_module(textwrap.dedent("""\
        module {
          kernel @test() {
            for (%i: index = 0 to 0) {
              %s: smem<2xf32> = alloc() attrs={id: 7, init: 1.0}
            }
          }
        }"""))
    before = repr(module)
    assert repr(licm(module)) == before


def test_guard_uses_the_same_invariance_rule_as_a_loop():
    module = gpu_dialect.parse_module(textwrap.dedent("""\
        module {
          kernel @test() {
            %bound: index = index_constant() attrs={value: 3}
            for (%i: index = 0 to 4) {
              %condition: pred = index_lt(%i: index, %bound: index)
              if(%condition: pred) {
                %step: index = index_constant() attrs={value: 2}
                %next: index = index_add(%i: index, %step: index)
                %acc: rmem<1xf32> = alloc() attrs={id: 7, init: 2.5}
              }
            }
          }
        }"""))
    licm(module)
    [loop] = find_ops_of_type(module, ForOp)
    [guard] = find_ops_of_type(module, IfOp)
    [reset] = find_ops_of_type(module, MoveOp)
    [next_op] = [op for op in loop.body.ops if op.op_id == "index_add"]
    assert next_op.operands[0] is loop.body.arguments[0]
    step = next_op.operands[1].definition
    assert isinstance(step, IndexConstantOp)
    assert step.parent_block is loop.parent_block
    assert reset.parent_block is guard.body


@requires_cuda
def test_hoisted_accumulator_resets_each_guarded_iteration_on_gpu():
    module = gpu_dialect.parse_module(textwrap.dedent("""\
        module {
          kernel @reset(%out: gmem<4xf32>) grid_dim=[1, 1, 1] block_dim=[1, 1, 1] {
            %limit: index = index_constant() attrs={value: 3}
            %overwrite: rmem<f32> = constant() attrs={value: 99.0}
            for (%i: index = 0 to 4) {
              %condition: pred = index_lt(%i: index, %limit: index)
              if(%condition: pred) {
                %acc: rmem<1xf32> = alloc() attrs={id: 7, init: 2.5}
                %tile: gmem<1xf32 strides=[1]> = tile(%out: gmem<4xf32>, %i: index)
                move(%tile: gmem<1xf32 strides=[1]>, %acc: rmem<1xf32>) attrs={dst_tiling_scheme: 0}
                move(%acc: rmem<1xf32>, %overwrite: rmem<f32>)
              }
            }
          }
        }"""))
    licm(module)
    out = torch.full((4,), -1.0, device="cuda")
    make_ptx_launcher(make_ptx_kernel(module))(out)
    torch.testing.assert_close(out, torch.tensor([2.5, 2.5, 2.5, -1.0], device="cuda"))


@requires_cuda
@pytest.mark.parametrize("dtype, torch_dtype", [
    ("bf16", torch.bfloat16), ("f16", torch.float16),
])
def test_broadcast_reset_initializes_both_halves_of_packed_registers_on_gpu(
    dtype, torch_dtype,
):
    fragment = f'wgrmem<64x64x{dtype} warps=4 layout="wgmma_layout_0">'
    module = gpu_dialect.parse_module(textwrap.dedent(f"""\
        module {{
          kernel @reset_packed(%out: gmem<128x64x{dtype}>) grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {{
            %overwrite: rmem<{dtype}> = constant() attrs={{value: 99.0}}
            for (%i: index = 0 to 2) {{
              %acc: {fragment} = alloc() attrs={{id: 7, init: 2.5}}
              %tile: gmem<64x64x{dtype} strides=[64, 1]> = tile(%out: gmem<128x64x{dtype}>, %i: index, 0)
              move(%tile: gmem<64x64x{dtype} strides=[64, 1]>, %acc: {fragment}) attrs={{dst_tiling_scheme: 0}}
              move(%acc: {fragment}, %overwrite: rmem<{dtype}>)
            }}
          }}
        }}"""))
    licm(module)
    out = torch.empty((128, 64), dtype=torch_dtype, device="cuda")
    make_ptx_launcher(make_ptx_kernel(module))(out)
    torch.testing.assert_close(out, torch.full_like(out, 2.5), rtol=0, atol=0)
