"""Descriptor LICM must preserve bank dependencies and engine participation."""

import pytest

from ayla.base.ir import find_ancestor_of_type, find_ops_of_type
from ayla.gemm_collection.blackwell_tcgen05 import blackwell_tcgen05_gemm_module
from ayla.gpu_dialect import (
    ForOp, IfOp, IndexType, PTXTemplateOp, PtrAddOp, ToPtrOp,
    gpu_dialect, is_side_effect_free,
)
from ayla.passes.licm import LICM
from ayla.ptx_codegen import make_ptx_kernel


def _module(**extra):
    cfg = dict(m=768, n=768, k=8192, dtype="f16", output_m_tile=64,
               output_n_tile=64, operand_k_tile=128, operand_stages=6,
               cyclic_operand_pipeline=True)
    cfg.update(extra)
    return blackwell_tcgen05_gemm_module(**cfg)


def test_disabled_descriptor_licm_preserves_default():
    assert make_ptx_kernel(_module()).ptx == make_ptx_kernel(
        _module(hoist_mma_descriptors=False)
    ).ptx


@pytest.mark.parametrize("value", [1, "low", None])
def test_descriptor_licm_requires_boolean(value):
    with pytest.raises(ValueError, match="must be a boolean"):
        _module(hoist_mma_descriptors=value)


def test_static_descriptor_fields_hoist_without_moving_engine_work():
    native = make_ptx_kernel(_module()).ptx
    module = _module(hoist_mma_descriptors=True)
    enabled = make_ptx_kernel(module).ptx
    preparations = [op for op in find_ops_of_type(module, PTXTemplateOp)
                    if op.attrs.get("pure")]
    assert preparations
    assert all(isinstance(op.result().type, IndexType) for op in preparations)
    assert all(find_ancestor_of_type(op, ForOp) is None for op in preparations)
    for instruction in ("tcgen05.mma.", "tcgen05.commit.", "cp.async.bulk.tensor.",
                        "tcgen05.alloc.", "tcgen05.dealloc.", "bar.sync",
                        "barrier.cluster.", "mbarrier.try_wait"):
        assert native.count(instruction) == enabled.count(instruction)


def test_dynamic_rolled_bank_descriptor_remains_in_loop():
    module = _module(m=256, n=256, k=1024, operand_k_tile=64,
                     rolled_operand_pipeline=True, cyclic_operand_pipeline=False,
                     hoist_mma_descriptors=True)
    make_ptx_kernel(module)
    preparations = [op for op in find_ops_of_type(module, PTXTemplateOp)
                    if op.attrs.get("pure")]
    assert any((loop := find_ancestor_of_type(op, ForOp)) is not None
               and loop.attrs.get("label") == "rolled_k_main"
               for op in preparations)


def test_guard_motion_preserves_dynamic_dependencies_and_side_effects():
    module = gpu_dialect.parse_module("""module {
      kernel @test(%s: smem<64x64xf16>) grid_dim=[1,1,1] block_dim=[128,1,1] {
        %zero: index = index_constant() attrs={value: 0}
        %one: index = index_constant() attrs={value: 1}
        %condition: pred = index_lt(%zero: index, %one: index)
        for (%k: index = 0 to 4) {
          if(%condition: pred) {}
        }
      }
    }""")
    kernel = module.body.ops[0]
    [loop] = find_ops_of_type(module, ForOp)
    [guard] = find_ops_of_type(module, IfOp)
    ptr = ToPtrOp.build(kernel.body.arguments[0])
    kernel.body.ops.insert(3, ptr)

    def prepare(value):
        return PTXTemplateOp.build(
            [value], [IndexType()],
            "shr.b32 ${b32:result.desc.val}, ${b32:operand.ptr.ptr}, 4;",
            operand_names=["ptr"], result_names=["desc"], attrs={"pure": True},
        )

    static = prepare(ptr.result())
    offset = PtrAddOp.build(ptr.result(), loop.iv)
    dynamic = prepare(offset.result())
    issue = PTXTemplateOp.build([static.result(), dynamic.result()], [], "issue")
    guard.body.ops = [static, offset, dynamic, issue]
    for _ in range(3):
        LICM(hoist_pure_through_guards=True).run(module)
    assert static.parent_block is kernel.body
    assert offset.parent_block is loop.body and dynamic.parent_block is loop.body
    assert issue.parent_block is guard.body
    assert is_side_effect_free(static) and not is_side_effect_free(issue)
