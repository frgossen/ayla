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
from ayla.gemm import blackwell_tcgen05_gemm_module
from ayla.gpu_dialect import (
    AllocOp,
    AsyncOp,
    AwaitOp,
    BF16Type,
    BlockIdsOp,
    F32Type,
    GmemTensorType,
    LockType,
    MoveOp,
    SmemTensorType,
    TmaDescType,
    TileOp,
)
from ayla.ptx_runtime import make_ptx_launcher
from ayla.targets import BLACKWELL


requires_blackwell = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0),
    reason="requires an SM100 GPU",
)


def _make_tcgen05_kernel(k=16, use_tma=False):
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
    if use_tma:
        lhs_tma_smem = AllocOp.build_with_fresh_id(SmemTensorType.build(
            [128, 16], BF16Type(),
        ))
        rhs_tma_smem = AllocOp.build_with_fresh_id(SmemTensorType.build(
            [64, 16], BF16Type(),
        ))
        ops.extend([lhs_tma_smem, rhs_tma_smem])
    for k_tile in range(k // 16):
        a_tile = TileOp.build(
            a, [0, k_tile],
            GmemTensorType.build([128, 16], BF16Type(), strides=[k, 1]),
        )
        b_tile = TileOp.build(
            b, [0, k_tile],
            GmemTensorType.build([64, 16], BF16Type(), strides=[k, 1]),
        )
        ops.extend([a_tile, b_tile])
        if use_tma:
            ops.extend([
                MoveOp.build(
                    lhs_tma_smem.result(), a_tile.result(), attrs={"tma": True},
                ),
                MoveOp.build(
                    rhs_tma_smem.result(), b_tile.result(), attrs={"tma": True},
                ),
                MoveOp.build(lhs_smem.result(), lhs_tma_smem.result()),
                MoveOp.build(rhs_smem.result(), rhs_tma_smem.result()),
            ])
        else:
            ops.extend([
                MoveOp.build(lhs_smem.result(), a_tile.result()),
                MoveOp.build(rhs_smem.result(), b_tile.result()),
            ])
        ops.append(MatmulOp.build(
            lhs_smem.result(), rhs_smem.result(), allocation.result(),
        ))
    ops.extend([
        MoveOp.build(c, allocation.result()),
        DeallocOp.build(allocation.result()),
    ])
    body = Block(arguments=[a, b, c], ops=ops)
    return make_ptx_kernel(ModuleOp.build(Block(ops=[KernelOp.build(
        "tcgen05_mma_smoke", body,
        grid_dim=(1, 1, 1), block_dim=(128, 1, 1),
    )]), target=BLACKWELL))


def _make_tcgen05_wide_tma_kernel(mma_tiles=4, n=64):
    assert 1 <= mma_tiles <= 4
    assert n in (64, 128, 256)
    bf16 = BF16Type()
    a = Value(name="a", type=GmemTensorType.build((128, 64), bf16))
    b = Value(name="b", type=GmemTensorType.build((n, 64), bf16))
    c = Value(name="c", type=GmemTensorType.build((128, n), F32Type()))
    accumulator = AllocOp.build_with_fresh_id(
        TmemTensorType.build([128, n], F32Type()), init=0.0,
    )
    lhs_smem = AllocOp.build_with_fresh_id(SmemTensorType.build(
        [128, 64], bf16, swizzle="128B", attrs={"tcgen05_matrix": True},
    ))
    rhs_smem = AllocOp.build_with_fresh_id(SmemTensorType.build(
        [n, 64], bf16, swizzle="128B", attrs={"tcgen05_matrix": True},
    ))
    tma_lock = AllocOp.build_with_fresh_id(LockType.build(kind="mbar"))
    mma_lock = AllocOp.build_with_fresh_id(LockType.build(kind="tcgen05"))
    mmas = []
    tile_ops = []
    for k_tile in range(mma_tiles):
        lhs_tile = TileOp.build(
            lhs_smem.result(), [0, k_tile],
            SmemTensorType.build(
                [128, 16], bf16, strides=[64, 1], swizzle="128B",
                attrs={"tcgen05_tma_layout": True},
            ),
        )
        rhs_tile = TileOp.build(
            rhs_smem.result(), [0, k_tile],
            SmemTensorType.build(
                [n, 16], bf16, strides=[64, 1], swizzle="128B",
                attrs={"tcgen05_tma_layout": True},
            ),
        )
        tile_ops.extend([lhs_tile, rhs_tile])
        mmas.append(MatmulOp.build(
            lhs_tile.result(), rhs_tile.result(), accumulator.result(),
        ))
    body = Block(arguments=[a, b, c], ops=[
        accumulator,
        lhs_smem,
        rhs_smem,
        tma_lock,
        mma_lock,
        AsyncOp.build(tma_lock.result(), body=Block(ops=[
            MoveOp.build(lhs_smem.result(), a, attrs={"tma": True}),
            MoveOp.build(rhs_smem.result(), b, attrs={"tma": True}),
        ])),
        AwaitOp.build(tma_lock.result()),
        *tile_ops,
        AsyncOp.build(mma_lock.result(), body=Block(ops=mmas)),
        AwaitOp.build(mma_lock.result()),
        MoveOp.build(c, accumulator.result()),
        DeallocOp.build(accumulator.result()),
    ])
    return make_ptx_kernel(ModuleOp.build(Block(ops=[KernelOp.build(
        "tcgen05_wide_tma_smoke", body,
        grid_dim=(1, 1, 1), block_dim=(128, 1, 1),
    )]), target=BLACKWELL))


def _make_tcgen05_cta_group2_kernel():
    bf16 = BF16Type()
    a = Value(name="a", type=GmemTensorType.build((256, 16), bf16))
    b = Value(name="b", type=GmemTensorType.build((64, 16), bf16))
    c = Value(name="c", type=GmemTensorType.build((256, 64), F32Type()))
    block_ids = BlockIdsOp.build()
    a_tile = TileOp.build(
        a, [block_ids.x, 0],
        GmemTensorType.build((128, 16), bf16, strides=[16, 1]),
    )
    c_tile = TileOp.build(
        c, [block_ids.x, 0],
        GmemTensorType.build((128, 64), F32Type(), strides=[64, 1]),
    )
    b_tile = TileOp.build(
        b, [block_ids.x, 0],
        GmemTensorType.build((32, 16), bf16, strides=[16, 1]),
    )
    accumulator = AllocOp.build_with_fresh_id(TmemTensorType.build(
        [128, 64], F32Type(), attrs={"cta_group": 2},
    ), init=0.0)
    lhs_smem = AllocOp.build_with_fresh_id(SmemTensorType.build(
        [128, 16], bf16, swizzle="32B",
        attrs={"tcgen05_layout": "lhs_128x16"},
    ))
    rhs_smem = AllocOp.build_with_fresh_id(SmemTensorType.build(
        [64, 16], bf16, swizzle="128B",
        attrs={"tcgen05_layout": "rhs_32x16_cta2"},
    ))
    mma_lock = AllocOp.build_with_fresh_id(LockType.build(kind="tcgen05"))
    mma = MatmulOp.build(
        lhs_smem.result(), rhs_smem.result(), accumulator.result(),
    )
    body = Block(arguments=[a, b, c], ops=[
        block_ids,
        a_tile,
        b_tile,
        c_tile,
        accumulator,
        lhs_smem,
        rhs_smem,
        mma_lock,
        MoveOp.build(lhs_smem.result(), a_tile.result()),
        MoveOp.build(rhs_smem.result(), b_tile.result()),
        AsyncOp.build(mma_lock.result(), body=Block(ops=[mma])),
        AwaitOp.build(mma_lock.result()),
        MoveOp.build(c_tile.result(), accumulator.result()),
        DeallocOp.build(accumulator.result()),
    ])
    return make_ptx_kernel(ModuleOp.build(Block(ops=[KernelOp.build(
        "tcgen05_cta_group2_smoke", body,
        grid_dim=(2, 1, 1), block_dim=(128, 1, 1),
        cluster_dim=(2, 1, 1),
    )]), target=BLACKWELL))


def test_tcgen05_typed_ir_lowering():
    ptx = _make_tcgen05_kernel().ptx

    assert "tcgen05.alloc.cta_group::1" in ptx
    assert "tcgen05.st.sync.aligned.32x32b.x64.b32" in ptx
    assert "tcgen05.mma.cta_group::1.kind::f16" in ptx
    assert "tcgen05.commit.cta_group::1.mbarrier::arrive::one" in ptx
    assert "mbarrier.try_wait.parity" in ptx
    assert "tcgen05.ld.sync.aligned.32x32b.x64.b32" in ptx
    assert "tcgen05.dealloc.cta_group::1" in ptx
    assert "wgmma" not in ptx


def test_tcgen05_gemm_tma_staging_lowering():
    ptx = make_ptx_kernel(blackwell_tcgen05_gemm_module(
        m=128, n=64, k=32, operand_stages=2, tma_staging="repack",
        output_n_tile=64,
    )).ptx

    assert "cp.async.bulk.tensor.2d.shared::cluster.global" in ptx
    assert "ld.shared.v4.b32" in ptx
    assert "ld.shared.b16" in ptx
    assert ptx.count("tcgen05.mma.cta_group::1.kind::f16") == 2


def test_tcgen05_wide_tma_staging_lowering():
    ptx = _make_tcgen05_wide_tma_kernel().ptx
    bf16_output_ptx = make_ptx_kernel(blackwell_tcgen05_gemm_module(
        m=128, n=128, k=64,
    )).ptx

    assert ptx.count("cp.async.bulk.tensor.2d.shared::cluster.global") == 2
    assert ptx.count("tcgen05.mma.cta_group::1.kind::f16") == 4
    assert "tcgen05.ld.sync.aligned.32x32b.x64.b32 {%f" in bf16_output_ptx
    assert "mov.b32 %f" not in bf16_output_ptx
    assert bf16_output_ptx.count("st.global.v8.b32") == 8


def test_tcgen05_k128_tma_staging_lowering():
    ptx = make_ptx_kernel(blackwell_tcgen05_gemm_module(
        m=128, n=128, k=128, operand_stages=8,
    )).ptx

    assert ptx.count("cp.async.bulk.tensor.2d.shared::cluster.global") == 4
    assert ptx.count("tcgen05.mma.cta_group::1.kind::f16") == 8
    assert ptx.count("0x4000404000000000") == 16
    assert ".shared .align 1024 .b8" in ptx
    assert ptx.count("tcgen05.st.sync.aligned.32x32b.x128.b32") == 1


def test_tcgen05_split_tma_barriers_lowering():
    ptx = make_ptx_kernel(blackwell_tcgen05_gemm_module(
        m=128, n=256, k=128,
        output_n_tile=256, operand_stages=8,
        split_tma_barriers=True,
    )).ptx

    assert ptx.count("mbarrier.arrive.expect_tx") == 2
    assert ptx.count("tcgen05.st.sync.aligned.32x32b.x128.b32") == 2
    assert ptx.count(", 49152;") == 2
    assert ptx.count("mbarrier.try_wait.parity.acquire") == 3
    first_wait = ptx.index("mbarrier.try_wait.parity.acquire")
    mma_offsets = []
    start = 0
    for _ in range(8):
        start = ptx.index("tcgen05.mma.cta_group::1.kind::f16", start)
        mma_offsets.append(start)
        start += 1
    second_wait = ptx.index("mbarrier.try_wait.parity.acquire", first_wait + 1)
    assert first_wait < mma_offsets[0] < mma_offsets[3]
    assert mma_offsets[3] < second_wait < mma_offsets[4]

    with pytest.raises(ValueError, match="split TMA barriers"):
        blackwell_tcgen05_gemm_module(
            m=128, n=256, k=64,
            output_n_tile=256, operand_stages=4,
            split_tma_barriers=True,
        )


def test_tcgen05_gemm_jit_optimization_level():
    kernel = make_ptx_kernel(blackwell_tcgen05_gemm_module(
        m=128, n=128, k=64, ptx_optimization_level=4,
    ))

    assert kernel.ptx_optimization_level == 4


def test_tcgen05_transposed_cta_grid_mapping():
    kernel = make_ptx_kernel(blackwell_tcgen05_gemm_module(
        m=128, n=256, k=64, transpose_cta_grid=True,
    ))

    assert kernel.grid_dim == (2, 1, 1)
    assert "mov.u32" in kernel.ptx
    assert "%ctaid.x" in kernel.ptx and "%ctaid.y" in kernel.ptx


def test_tcgen05_split_last_n_tile_lowering():
    kernel = make_ptx_kernel(blackwell_tcgen05_gemm_module(
        m=128, n=4096, k=128,
        output_n_tile=256, operand_stages=8,
        transpose_cta_grid=True, split_last_n_tile=True,
    ))

    assert kernel.grid_dim == (17, 1, 1)
    assert "0x08400490" in kernel.ptx
    assert "0x08200490" in kernel.ptx
    assert "shl.b32" in kernel.ptx
    assert "setp.ge.u32" in kernel.ptx
    assert "hybrid_n_done" not in kernel.ptx

    with pytest.raises(ValueError, match="splitting the last N tile"):
        blackwell_tcgen05_gemm_module(
            m=128, n=256, k=128,
            output_n_tile=128, operand_stages=8,
            transpose_cta_grid=True, split_last_n_tile=True,
        )


def test_tcgen05_partial_split_last_n_tile_lowering():
    kernel = make_ptx_kernel(blackwell_tcgen05_gemm_module(
        m=4096, n=4096, k=128,
        output_n_tile=256, operand_stages=8,
        transpose_cta_grid=True, split_last_n_tile_rows=24,
    ))

    assert kernel.grid_dim == (536, 1, 1)
    assert "setp.lt.u32" in kernel.ptx and ", 408;" in kernel.ptx
    assert "min.u32" in kernel.ptx
    assert "0x08400490" in kernel.ptx
    assert "0x08200490" in kernel.ptx
    assert "rem.u32" in kernel.ptx and ", 17;" in kernel.ptx
    assert "hybrid_n_done" not in kernel.ptx

    with pytest.raises(ValueError, match="strictly between"):
        blackwell_tcgen05_gemm_module(
            m=256, n=512, k=128,
            output_n_tile=256, operand_stages=8,
            transpose_cta_grid=True, split_last_n_tile_rows=2,
        )


def test_tcgen05_operand_l2_cache_hints():
    ptx = make_ptx_kernel(blackwell_tcgen05_gemm_module(
        m=128, n=128, k=128, operand_stages=8,
        lhs_l2_cache_hint="evict_last", rhs_l2_cache_hint="evict_first",
    )).ptx

    assert ptx.count("createpolicy.fractional.L2::evict_last") == 1
    assert ptx.count("createpolicy.fractional.L2::evict_first") == 1
    assert ptx.count(".L2::cache_hint") == 4

    promoted = make_ptx_kernel(blackwell_tcgen05_gemm_module(
        m=128, n=256, k=128, output_n_tile=256, operand_stages=8,
        lhs_l2_promotion="128B", rhs_l2_promotion="256B",
    ))
    input_descs = {
        typ.arg_idx: typ for typ in promoted.signature
        if isinstance(typ, TmaDescType) and typ.arg_idx in (0, 1)
    }
    assert input_descs[0].l2_promotion == "128B"
    assert input_descs[1].l2_promotion == "256B"


def test_tcgen05_gemm_cluster_a_multicast_lowering():
    kernel = make_ptx_kernel(blackwell_tcgen05_gemm_module(
        m=128, n=256, k=64, cluster_multicast_a=True,
    ))

    assert kernel.grid_dim == (1, 2, 1)
    assert kernel.cluster_dim == (1, 2, 1)
    assert ".reqnctapercluster 1, 2, 1" in kernel.ptx
    assert kernel.ptx.count("barrier.cluster.arrive.aligned") == 1
    assert kernel.ptx.count("barrier.cluster.wait.aligned") == 1
    assert kernel.ptx.count(
        "cp.async.bulk.tensor.2d.shared::cluster.global."
        "mbarrier::complete_tx::bytes.multicast::cluster"
    ) == 1
    assert "mov.u32 %r" in kernel.ptx and "%cluster_ctarank" in kernel.ptx
    assert "setp.eq.u32" in kernel.ptx


def test_tcgen05_gemm_branches_nonissuers_around_issue_sequence():
    ptx = make_ptx_kernel(blackwell_tcgen05_gemm_module(
        m=128, n=128, k=64,
    )).ptx

    branch = ptx.index("bra $L")
    descriptor = ptx.index("mov.b32", branch)
    mma = ptx.index("tcgen05.mma.cta_group::1", descriptor)
    join = ptx.index("$L", mma)
    assert branch < descriptor < mma < join


def test_tcgen05_m64_tma_staging_lowering():
    kernel = make_ptx_kernel(blackwell_tcgen05_gemm_module(
        m=128, n=128, k=64, output_m_tile=64,
    ))

    assert kernel.grid_dim == (2, 1, 1)
    assert "mov.b32 %r" in kernel.ptx and "0x04200490" in kernel.ptx
    assert "setp.lt.u32" in kernel.ptx
    assert "and.b32" in kernel.ptx
    assert "@%p" in kernel.ptx and "st.global.v8.b32" in kernel.ptx


def test_tcgen05_cta_group2_lowering():
    ptx = _make_tcgen05_cta_group2_kernel().ptx

    assert ".reqnctapercluster 2, 1, 1" in ptx
    assert "tcgen05.alloc.cta_group::2" in ptx
    assert "tcgen05.mma.cta_group::2.kind::f16" in ptx
    assert "multicast::cluster" in ptx
    assert "tcgen05.dealloc.cta_group::2" in ptx


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


@requires_blackwell
@pytest.mark.parametrize("seed", [0, 1, 7])
def test_tcgen05_bf16_mma_with_tma_staging(seed):
    kernel = _make_tcgen05_kernel(use_tma=True)

    torch.manual_seed(seed)
    a_tensor = torch.randn((128, 16), dtype=torch.bfloat16, device="cuda")
    b_tensor = torch.randn((64, 16), dtype=torch.bfloat16, device="cuda")
    actual = torch.empty((128, 64), dtype=torch.float32, device="cuda")
    make_ptx_launcher(kernel)(a_tensor, b_tensor, actual)
    torch.cuda.synchronize()

    expected = a_tensor.float() @ b_tensor.float().T
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-2)


@requires_blackwell
@pytest.mark.parametrize("seed,n", [(0, 64), (1, 128), (7, 256)])
def test_tcgen05_bf16_mma_with_wide_tma_staging(seed, n):
    kernel = _make_tcgen05_wide_tma_kernel(n=n)

    torch.manual_seed(seed)
    a_tensor = torch.randn((128, 64), dtype=torch.bfloat16, device="cuda")
    b_tensor = torch.randn((n, 64), dtype=torch.bfloat16, device="cuda")
    actual = torch.empty((128, n), dtype=torch.float32, device="cuda")
    make_ptx_launcher(kernel)(a_tensor, b_tensor, actual)
    torch.cuda.synchronize()

    expected = a_tensor.float() @ b_tensor.float().T
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-2)


@requires_blackwell
@pytest.mark.parametrize("seed", [0, 1, 7])
def test_tcgen05_m64_gemm(seed):
    kernel = make_ptx_kernel(blackwell_tcgen05_gemm_module(
        m=64, n=128, k=64, output_m_tile=64,
    ))

    torch.manual_seed(seed)
    a_tensor = torch.randn((64, 64), dtype=torch.bfloat16, device="cuda")
    b_tensor = torch.randn((128, 64), dtype=torch.bfloat16, device="cuda")
    actual = torch.empty((64, 128), dtype=torch.bfloat16, device="cuda")
    make_ptx_launcher(kernel)(a_tensor, b_tensor, actual)
    torch.cuda.synchronize()

    expected = torch.mm(a_tensor, b_tensor.T)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=2.0)


@requires_blackwell
@pytest.mark.parametrize("seed", [0, 1, 7])
def test_tcgen05_k128_gemm(seed):
    kernel = make_ptx_kernel(blackwell_tcgen05_gemm_module(
        m=128, n=128, k=128, operand_stages=8,
    ))

    torch.manual_seed(seed)
    a_tensor = torch.randn((128, 128), dtype=torch.bfloat16, device="cuda")
    b_tensor = torch.randn((128, 128), dtype=torch.bfloat16, device="cuda")
    actual = torch.empty((128, 128), dtype=torch.bfloat16, device="cuda")
    make_ptx_launcher(kernel)(a_tensor, b_tensor, actual)
    torch.cuda.synchronize()

    expected = torch.mm(a_tensor, b_tensor.T)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=2.0)


@requires_blackwell
@pytest.mark.parametrize("seed", [0, 1, 7])
def test_tcgen05_bf16_mma_cta_group2(seed):
    kernel = _make_tcgen05_cta_group2_kernel()

    torch.manual_seed(seed)
    a_tensor = torch.randn((256, 16), dtype=torch.bfloat16, device="cuda")
    b_tensor = torch.randn((64, 16), dtype=torch.bfloat16, device="cuda")
    actual = torch.empty((256, 64), dtype=torch.float32, device="cuda")
    make_ptx_launcher(kernel)(a_tensor, b_tensor, actual)
    torch.cuda.synchronize()

    expected = a_tensor.float() @ b_tensor.float().T
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-2)
