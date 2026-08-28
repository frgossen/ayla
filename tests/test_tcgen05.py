"""Focused Blackwell tensor-core bring-up tests."""

import textwrap

import pytest
import torch

from ayla import (
    Block,
    DeallocOp,
    KernelOp,
    MatmulOp,
    ModuleOp,
    PTXTemplateOp,
    TmemTensorType,
    Value,
    make_ptx_kernel,
)
from ayla.gpu_dialect import (
    AllocOp,
    BF16Type,
    F32Type,
    GmemTensorType,
    SmemTensorType,
)
from ayla.ptx_runtime import make_ptx_launcher
from ayla.targets import BLACKWELL


requires_blackwell = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0),
    reason="requires an SM100 GPU",
)


TCGEN05_STAGE_INPUTS = textwrap.dedent("""\
    {
      .reg .b32 %tc_tid, %tc_smem;
      .reg .b32 %tc_x0, %tc_x1, %tc_x2, %tc_x3, %tc_x4;
      .reg .b32 %tc_a0, %tc_a1, %tc_a2, %tc_a3;
      .reg .b32 %tc_a4, %tc_a5, %tc_a6, %tc_a7;
      .reg .b32 %tc_b0, %tc_b1, %tc_b2, %tc_b3;
      .reg .b32 %tc_saddr0, %tc_saddr1, %tc_saddr_b;
      .reg .b64 %tc_ga0, %tc_ga1, %tc_gb;
      .shared .align 1024 .b8 tc_smem_storage[6144];

      mov.u32 %tc_tid, %tid.x;
      mov.u32 %tc_smem, tc_smem_storage;
      mov.u32 ${b32:result.lhs.ptr}, %tc_smem;
      add.u32 ${b32:result.rhs.ptr}, %tc_smem, 4096;

      // Cooperative loads use the same swizzled layouts and descriptors as
      // the locally installed Triton SM100 backend for this exact tile.
      shl.b32 %tc_x0, %tc_tid, 3;
      and.b32 %tc_x1, %tc_x0, 1008;
      and.b32 %tc_x2, %tc_x0, 8;
      add.u32 %tc_x3, %tc_x1, %tc_x2;
      mad.wide.u32 %tc_ga0, %tc_x3, 2, ${b64:operand.a.ptr};
      add.s64 %tc_ga1, %tc_ga0, 2048;
      ld.global.v4.b32 {%tc_a0, %tc_a1, %tc_a2, %tc_a3}, [%tc_ga0];
      ld.global.v4.b32 {%tc_a4, %tc_a5, %tc_a6, %tc_a7}, [%tc_ga1];

      shl.b32 %tc_x4, %tc_tid, 4;
      and.b32 %tc_x1, %tc_x4, 1904;
      bfe.s32 %tc_x2, %tc_tid, 3, 1;
      and.b32 %tc_x2, %tc_x2, 144;
      xor.b32 %tc_x3, %tc_x2, %tc_x1;
      add.u32 %tc_saddr0, %tc_smem, %tc_x3;
      add.u32 %tc_saddr1, %tc_saddr0, 2048;
      st.shared.v4.b32 [%tc_saddr0], {%tc_a0, %tc_a1, %tc_a2, %tc_a3};
      st.shared.v4.b32 [%tc_saddr1], {%tc_a4, %tc_a5, %tc_a6, %tc_a7};

      and.b32 %tc_x1, %tc_x0, 960;
      and.b32 %tc_x2, %tc_x0, 56;
      add.u32 %tc_x3, %tc_x1, %tc_x2;
      mad.wide.u32 %tc_gb, %tc_x3, 2, ${b64:operand.b.ptr};
      ld.global.v4.b32 {%tc_b0, %tc_b1, %tc_b2, %tc_b3}, [%tc_gb];

      and.b32 %tc_x1, %tc_x4, 2032;
      shl.b32 %tc_x2, %tc_tid, 1;
      and.b32 %tc_x2, %tc_x2, 112;
      xor.b32 %tc_x3, %tc_x1, %tc_x2;
      add.u32 %tc_saddr_b, %tc_smem, %tc_x3;
      add.u32 %tc_saddr_b, %tc_saddr_b, 4096;
      st.shared.v4.b32 [%tc_saddr_b], {%tc_b0, %tc_b1, %tc_b2, %tc_b3};
      bar.sync 0;
    }
""")


TCGEN05_STORE_FIRST_COLUMN = textwrap.dedent("""\
    mov.u32 ${b32:tid}, %tid.x;
    shr.u32 ${b32:warp}, ${b32:tid}, 5;
    shl.b32 ${b32:lane_base}, ${b32:warp}, 21;
    add.u32 ${b32:taddr}, ${b32:operand.tmem.ptr}, ${b32:lane_base};
    tcgen05.ld.sync.aligned.32x32b.x1.b32 {${b32:value}}, [${b32:taddr}];
    tcgen05.wait::ld.sync.aligned;
    mul.wide.u32 ${b64:offset}, ${b32:tid}, 256;
    add.u64 ${b64:address}, ${b64:operand.c.ptr}, ${b64:offset};
    st.global.b32 [${b64:address}], ${b32:value};
""")


def _make_tcgen05_kernel():
    a_type = GmemTensorType.build((128, 16), BF16Type())
    b_type = GmemTensorType.build((16, 64), BF16Type())
    c_type = GmemTensorType.build((128, 64), F32Type())
    a = Value(name="a", type=a_type)
    b = Value(name="b", type=b_type)
    c = Value(name="c", type=c_type)
    allocation = AllocOp.build_with_fresh_id(
        TmemTensorType.build([128, 64], F32Type()),
    )
    staged = PTXTemplateOp.build(
        [a, b],
        [
            SmemTensorType.build([128, 16], BF16Type(), swizzle="32B"),
            SmemTensorType.build([64, 16], BF16Type(), swizzle="128B"),
        ],
        template=TCGEN05_STAGE_INPUTS,
        operand_names=["a", "b"],
        result_names=["lhs", "rhs"],
    )
    body = Block(arguments=[a, b, c], ops=[
        allocation,
        staged,
        MatmulOp.build(staged.results[0], staged.results[1], allocation.result()),
        PTXTemplateOp.build(
            [c, allocation.result()], [], template=TCGEN05_STORE_FIRST_COLUMN,
            operand_names=["c", "tmem"],
        ),
        DeallocOp.build(allocation.result()),
    ])
    return make_ptx_kernel(ModuleOp.build(Block(ops=[KernelOp.build(
        "tcgen05_mma_smoke", body,
        grid_dim=(1, 1, 1), block_dim=(128, 1, 1),
    )]), target=BLACKWELL))


def test_tcgen05_typed_ir_lowering():
    ptx = _make_tcgen05_kernel().ptx

    assert "tcgen05.alloc.cta_group::1" in ptx
    assert "tcgen05.mma.cta_group::1.kind::f16" in ptx
    assert "tcgen05.commit.cta_group::1.mbarrier::arrive::one" in ptx
    assert "mbarrier.try_wait.parity" in ptx
    assert "tcgen05.dealloc.cta_group::1" in ptx
    assert "wgmma" not in ptx


@requires_blackwell
@pytest.mark.parametrize("seed", [0, 1, 7])
def test_tcgen05_bf16_mma_first_column(seed):
    kernel = _make_tcgen05_kernel()

    torch.manual_seed(seed)
    a_tensor = torch.randn((128, 16), dtype=torch.bfloat16, device="cuda")
    b_tensor = torch.randn((16, 64), dtype=torch.bfloat16, device="cuda")
    actual = torch.full((128, 64), torch.nan, dtype=torch.float32, device="cuda")
    make_ptx_launcher(kernel)(a_tensor, b_tensor, actual)
    torch.cuda.synchronize()

    expected = a_tensor.float() @ b_tensor.float()
    assert torch.isfinite(actual[:, 0]).all()
    torch.testing.assert_close(actual[:, 0], expected[:, 0], rtol=2e-3, atol=2e-2)
    assert torch.isnan(actual[:, 1:]).all()
