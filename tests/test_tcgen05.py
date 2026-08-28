"""Focused Blackwell tensor-core bring-up tests."""

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
    MoveOp,
    SmemTensorType,
    TileOp,
)
from ayla.ptx_runtime import make_ptx_launcher
from ayla.targets import BLACKWELL


requires_blackwell = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0),
    reason="requires an SM100 GPU",
)


def _make_tcgen05_kernel(k=16):
    assert k % 16 == 0
    a_type = GmemTensorType.build((128, k), BF16Type())
    b_type = GmemTensorType.build((64, k), BF16Type())
    c_type = GmemTensorType.build((128, 64), F32Type())
    a = Value(name="a", type=a_type)
    b = Value(name="b", type=b_type)
    c = Value(name="c", type=c_type)
    allocation = AllocOp.build_with_fresh_id(
        TmemTensorType.build([128, 64], F32Type()), init=0.0,
    )
    lhs_smem = AllocOp.build_with_fresh_id(SmemTensorType.build(
        [128, 16], BF16Type(), swizzle="32B",
        attrs={"tcgen05_layout": "lhs_128x16"},
    ))
    rhs_smem = AllocOp.build_with_fresh_id(SmemTensorType.build(
        [64, 16], BF16Type(), swizzle="128B",
        attrs={"tcgen05_layout": "rhs_64x16"},
    ))
    ops = [
        allocation,
        lhs_smem,
        rhs_smem,
    ]
    for k_tile in range(k // 16):
        a_tile = TileOp.build(
            a, [0, k_tile],
            GmemTensorType.build([128, 16], BF16Type(), strides=[k, 1]),
        )
        b_tile = TileOp.build(
            b, [0, k_tile],
            GmemTensorType.build([64, 16], BF16Type(), strides=[k, 1]),
        )
        ops.extend([
            a_tile,
            b_tile,
            MoveOp.build(lhs_smem.result(), a_tile.result()),
            MoveOp.build(rhs_smem.result(), b_tile.result()),
            MatmulOp.build(
                lhs_smem.result(), rhs_smem.result(), allocation.result(),
            ),
        ])
    ops.extend([
        MoveOp.build(c, allocation.result()),
        DeallocOp.build(allocation.result()),
    ])
    body = Block(arguments=[a, b, c], ops=ops)
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
    assert "tcgen05.ld.sync.aligned.32x32b.x1.b32" in ptx
    assert "tcgen05.dealloc.cta_group::1" in ptx
    assert "wgmma" not in ptx


@requires_blackwell
@pytest.mark.parametrize("seed,k", [(0, 16), (1, 32), (7, 32)])
def test_tcgen05_bf16_mma(seed, k):
    kernel = _make_tcgen05_kernel(k)

    torch.manual_seed(seed)
    a_tensor = torch.randn((128, k), dtype=torch.bfloat16, device="cuda")
    b_tensor = torch.randn((64, k), dtype=torch.bfloat16, device="cuda")
    actual = torch.full((128, 64), torch.nan, dtype=torch.float32, device="cuda")
    make_ptx_launcher(kernel)(a_tensor, b_tensor, actual)
    torch.cuda.synchronize()

    expected = a_tensor.float() @ b_tensor.float().T
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-2)
