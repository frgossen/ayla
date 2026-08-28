"""Focused Blackwell tensor-core bring-up tests."""

import textwrap

import pytest
import torch

from ayla import Block, KernelOp, ModuleOp, PTXTemplateOp, Value, make_ptx_kernel
from ayla.gpu_dialect import BF16Type, F32Type, GmemTensorType
from ayla.ptx_runtime import make_ptx_launcher
from ayla.targets import BLACKWELL


requires_blackwell = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0),
    reason="requires an SM100 GPU",
)


TCGEN05_MMA_SMOKE = textwrap.dedent("""\
    {
      .reg .pred %tc_p_warp0, %tc_p_elected, %tc_p_issue, %tc_p_done, %tc_p_false;
      .reg .b32 %tc_tid, %tc_warp, %tc_elect_lane, %tc_smem, %tc_taddr;
      .reg .b32 %tc_x0, %tc_x1, %tc_x2, %tc_x3, %tc_x4, %tc_x5;
      .reg .b32 %tc_a0, %tc_a1, %tc_a2, %tc_a3;
      .reg .b32 %tc_a4, %tc_a5, %tc_a6, %tc_a7;
      .reg .b32 %tc_b0, %tc_b1, %tc_b2, %tc_b3;
      .reg .b32 %tc_saddr0, %tc_saddr1, %tc_saddr_b;
      .reg .b32 %tc_bar, %tc_taddr_warp, %tc_out, %tc_idesc, %tc_zero;
      .reg .b64 %tc_ga0, %tc_ga1, %tc_gb, %tc_desc_a, %tc_desc_b, %tc_gc;
      .shared .align 1024 .b8 tc_smem_storage[6152];

      mov.u32 %tc_tid, %tid.x;
      mov.u32 %tc_smem, tc_smem_storage;
      setp.lt.u32 %tc_p_warp0, %tc_tid, 32;
      @%tc_p_warp0 tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32
          [%tc_smem], 64;
      bar.sync 0;
      ld.shared.b32 %tc_taddr, [tc_smem_storage];
      bar.sync 0;
      @%tc_p_warp0 tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;

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
      fence.proxy.async.shared::cta;

      shr.u32 %tc_warp, %tc_tid, 5;
      setp.eq.u32 %tc_p_warp0, %tc_warp, 0;
      elect.sync %tc_elect_lane|%tc_p_elected, -1;
      and.pred %tc_p_issue, %tc_p_warp0, %tc_p_elected;
      add.u32 %tc_bar, %tc_smem, 6144;
      @%tc_p_issue mbarrier.init.shared::cta.b64 [%tc_bar], 1;
      bar.sync 0;

      bfe.u32 %tc_x0, %tc_smem, 4, 14;
      cvt.u64.u32 %tc_desc_a, %tc_x0;
      add.u32 %tc_x1, %tc_smem, 4096;
      bfe.u32 %tc_x1, %tc_x1, 4, 14;
      cvt.u64.u32 %tc_desc_b, %tc_x1;
      or.b64 %tc_desc_a, %tc_desc_a, 0xc000401000000000;
      or.b64 %tc_desc_b, %tc_desc_b, 0x4000404000000000;
      mov.b32 %tc_idesc, 0x08110490;
      mov.b32 %tc_zero, 0;
      mov.pred %tc_p_false, 0;
      @%tc_p_issue tcgen05.mma.cta_group::1.kind::f16
          [%tc_taddr], %tc_desc_a, %tc_desc_b, %tc_idesc,
          {%tc_zero, %tc_zero, %tc_zero, %tc_zero}, %tc_p_false;
      @%tc_p_issue tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64
          [%tc_bar];

    tc_wait:
      mbarrier.try_wait.parity.shared::cta.b64 %tc_p_done, [%tc_bar], 0;
      @!%tc_p_done bra.uni tc_wait;
      bar.sync 0;
      @%tc_p_issue mbarrier.inval.shared::cta.b64 [%tc_bar];

      shl.b32 %tc_x0, %tc_warp, 21;
      add.u32 %tc_taddr_warp, %tc_taddr, %tc_x0;
      tcgen05.ld.sync.aligned.32x32b.x1.b32 {%tc_out}, [%tc_taddr_warp];
      tcgen05.wait::ld.sync.aligned;
      mul.wide.u32 %tc_gc, %tc_tid, 256;
      add.u64 %tc_gc, %tc_gc, ${b64:operand.c.ptr};
      st.global.b32 [%tc_gc], %tc_out;

      bar.sync 0;
      @%tc_p_warp0 tcgen05.dealloc.cta_group::1.sync.aligned.b32 %tc_taddr, 64;
    }
""")


@requires_blackwell
@pytest.mark.parametrize("seed", [0, 1, 7])
def test_tcgen05_bf16_mma_first_column(seed):
    a_type = GmemTensorType.build((128, 16), BF16Type())
    b_type = GmemTensorType.build((16, 64), BF16Type())
    c_type = GmemTensorType.build((128, 64), F32Type())
    a = Value(name="a", type=a_type)
    b = Value(name="b", type=b_type)
    c = Value(name="c", type=c_type)
    body = Block(arguments=[a, b, c], ops=[PTXTemplateOp.build(
        [a, b, c], [], template=TCGEN05_MMA_SMOKE,
        operand_names=["a", "b", "c"],
    )])
    kernel = make_ptx_kernel(ModuleOp.build(Block(ops=[KernelOp.build(
        "tcgen05_mma_smoke", body,
        grid_dim=(1, 1, 1), block_dim=(128, 1, 1),
    )]), target=BLACKWELL))

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
