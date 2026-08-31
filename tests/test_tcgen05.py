"""Focused Blackwell tensor-core bring-up tests."""

import re

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
from ayla.base.ir import iter_ops
from ayla.gemm import (
    blackwell_tcgen05_cta_group2_gemm_module,
    blackwell_tcgen05_gemm_module,
)
from ayla.gpu_dialect import (
    AllocOp,
    AsyncCommitOp,
    AsyncOp,
    AwaitOp,
    BF16Type,
    BlockIdsOp,
    F32Type,
    ForOp,
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
    assert re.search(r"@%p\d+ tcgen05\.mma\.cta_group::1", ptx)
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


@pytest.mark.parametrize("stages", [16, 32])
def test_tcgen05_derives_descriptors_for_long_direct_groups(stages):
    ordinary = make_ptx_kernel(blackwell_tcgen05_gemm_module(
        m=64, n=64, k=512,
        output_m_tile=64, output_n_tile=64, operand_stages=stages,
    )).ptx
    derived = make_ptx_kernel(blackwell_tcgen05_gemm_module(
        m=64, n=64, k=512,
        output_m_tile=64, output_n_tile=64, operand_stages=stages,
        derive_tcgen05_descriptors=True,
    )).ptx

    assert derived.count("tcgen05.mma.cta_group::1.kind::f16") == stages
    assert derived.count("0x4000404000000000") == stages // 2
    assert derived.count("add.u64") > ordinary.count("add.u64")
    assert len(derived) < len(ordinary)


def test_tcgen05_unrolls_direct_mainloop():
    module = blackwell_tcgen05_gemm_module(
        m=64, n=64, k=1024,
        output_m_tile=64, output_n_tile=64, operand_stages=16,
        derive_tcgen05_descriptors=True, unroll_direct_mainloop=True,
    )
    direct_loops = [
        op for op in iter_ops(module)
        if isinstance(op, ForOp)
        and op.attrs.get("label") == "tcgen05_direct_k"
    ]
    assert len(direct_loops) == 1 and direct_loops[0].trip_count == 0
    ptx = make_ptx_kernel(module).ptx
    assert ptx.count("tcgen05.mma.cta_group::1.kind::f16") == 64


def test_tcgen05_overlaps_initial_tma_with_tensor_memory_setup():
    module = blackwell_tcgen05_gemm_module(
        m=512, n=512, k=512,
        output_m_tile=64, output_n_tile=32, operand_stages=32,
        derive_tcgen05_descriptors=True,
        overlap_initial_tma_setup=True,
    )
    assert not any(
        isinstance(op, ForOp)
        and op.attrs.get("label") == "tcgen05_direct_k"
        for op in iter_ops(module)
    )

    ptx = make_ptx_kernel(module).ptx
    first_tma = ptx.index("cp.async.bulk.tensor.2d.shared::cluster.global")
    tmem_alloc = ptx.index("tcgen05.alloc.cta_group::1")
    tma_wait = ptx.index("mbarrier.try_wait.parity.acquire")
    first_mma = ptx.index("tcgen05.mma.cta_group::1")
    assert first_tma < tmem_alloc < tma_wait < first_mma
    assert ptx.count("tcgen05.mma.cta_group::1.kind::f16") == 32

    with pytest.raises(ValueError, match="must be a boolean"):
        blackwell_tcgen05_gemm_module(overlap_initial_tma_setup=1)
    with pytest.raises(ValueError, match="one non-persistent direct"):
        blackwell_tcgen05_gemm_module(
            m=64, n=64, k=1024,
            output_m_tile=64, output_n_tile=64, operand_stages=16,
            prefetch_next_banks=1,
            overlap_initial_tma_setup=True,
        )


def test_tcgen05_prefetches_next_k256_banks():
    kernel = make_ptx_kernel(blackwell_tcgen05_gemm_module(
        m=64, n=64, k=1024,
        output_m_tile=64, output_n_tile=64, operand_stages=16,
        derive_tcgen05_descriptors=True,
        prefetch_next_banks=2,
    ))
    ptx = kernel.ptx

    assert ptx.count("elect.sync") == 0
    assert ptx.count("tcgen05.mma.cta_group::1.kind::f16") == 64
    assert ptx.count("tcgen05.commit.cta_group::1") == 10
    assert ptx.count("mbarrier.arrive.expect_tx") == 12
    assert ptx.count(", 16384;") == 8
    assert ptx.count(", 32768;") == 4


def test_tcgen05_overlaps_k256_rest_wait_after_prefetched_banks():
    ptx = make_ptx_kernel(blackwell_tcgen05_gemm_module(
        m=64, n=64, k=1024,
        output_m_tile=64, output_n_tile=64, operand_stages=16,
        derive_tcgen05_descriptors=True,
        prefetch_next_banks=2, overlap_k256_rest_wait=True,
    )).ptx

    group_start = ptx.index("async(%0: lock<tcgen05>")
    group_end = ptx.index("await(%0: lock<tcgen05>", group_start)
    group = ptx[group_start:group_end]
    rest_wait = group.index("mbarrier.try_wait.parity.acquire")
    assert group[:rest_wait].count("tcgen05.mma.cta_group::1") == 8
    assert group[rest_wait:].count("tcgen05.mma.cta_group::1") == 8

    with pytest.raises(ValueError, match="requires prefetched banks"):
        blackwell_tcgen05_gemm_module(overlap_k256_rest_wait=True)


def test_tcgen05_prefetched_persistent_pairs_overwrite_each_output():
    kernel = make_ptx_kernel(blackwell_tcgen05_gemm_module(
        m=64, n=128, k=1024,
        output_m_tile=64, output_n_tile=64, operand_stages=16,
        derive_tcgen05_descriptors=True, prefetch_next_banks=1,
        transpose_cta_grid=True, persistent_n_pairs=True,
        output_tmem_load_shape="16x32bx2",
    ))
    ptx = kernel.ptx

    assert kernel.grid_dim == (1, 1, 1)
    overwrite = ptx.index("mov.pred")
    first_mma = ptx.index("tcgen05.mma.cta_group::1", overwrite)
    restore = ptx.index("mov.pred", overwrite + 1)
    assert ptx[overwrite:ptx.index("\n", overwrite)].endswith(", 0;")
    assert ptx[restore:ptx.index("\n", restore)].endswith(", -1;")
    assert overwrite < first_mma < restore


def test_tcgen05_overwrites_fresh_accumulator_on_first_static_mma():
    module = blackwell_tcgen05_gemm_module(
        m=64, n=64, k=1024,
        output_m_tile=64, output_n_tile=64, operand_stages=16,
        derive_tcgen05_descriptors=True, prefetch_next_banks=1,
        overwrite_first_mma=True,
    )
    accumulator = next(
        op for op in iter_ops(module)
        if isinstance(op, AllocOp)
        and isinstance(op.result().type, TmemTensorType)
    )
    assert accumulator.init is None
    first_mma = next(op for op in iter_ops(module) if isinstance(op, MatmulOp))
    assert first_mma.attrs["overwrite_accumulator"] is True

    with pytest.raises(ValueError, match="statically identifiable"):
        blackwell_tcgen05_gemm_module(
            m=64, n=64, k=1024,
            output_m_tile=64, output_n_tile=64, operand_stages=4,
            overwrite_first_mma=True,
        )


def test_tcgen05_batches_lock_initialization_sync():
    ptx = make_ptx_kernel(blackwell_tcgen05_gemm_module(
        m=64, n=64, k=1024,
        output_m_tile=64, output_n_tile=64, operand_stages=16,
        derive_tcgen05_descriptors=True, prefetch_next_banks=1,
        batch_lock_initialization=True,
    )).ptx

    assert ptx.count("mbarrier.init.shared::cta.b64") == 4
    assert ptx.count("fence.mbarrier_init.release.cluster") == 1
    # Tensor-memory/issuer setup contributes two rendezvous; the lock batch
    # contributes only its final one instead of one per lock.
    assert ptx[:ptx.index("cp.async.bulk.tensor")].count("bar.sync 0;") == 3


def test_tcgen05_overlaps_deallocation_with_final_m64_store():
    ptx = make_ptx_kernel(blackwell_tcgen05_gemm_module(
        m=64, n=64, k=1024,
        output_m_tile=64, output_n_tile=64, operand_stages=16,
        derive_tcgen05_descriptors=True, prefetch_next_banks=1,
        output_tmem_load_shape="16x32bx2",
        overlap_tmem_deallocation=True,
    )).ptx

    final_load = ptx.rfind("tcgen05.ld.sync.aligned.16x32bx2")
    dealloc = ptx.index("tcgen05.dealloc.cta_group::1", final_load)
    final_store = ptx.index("st.global", dealloc)
    assert final_load < dealloc < final_store
    assert ptx.count("tcgen05.dealloc.cta_group::1") == 1

    with pytest.raises(ValueError, match="overlapped tensor-memory"):
        blackwell_tcgen05_gemm_module(
            m=128, n=128, k=256,
            output_m_tile=128, output_n_tile=128, operand_stages=8,
            output_tmem_load_shape="16x32bx2",
            overlap_tmem_deallocation=True,
        )


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


def test_tcgen05_split_rhs_tma_lowering():
    kernel = make_ptx_kernel(blackwell_tcgen05_gemm_module(
        m=128, n=256, k=128,
        output_n_tile=256, operand_stages=8,
        split_tma_barriers=True, split_rhs_tma=True,
    ))
    input_descs = {
        typ.arg_idx: typ for typ in kernel.signature
        if isinstance(typ, TmaDescType) and typ.arg_idx in (0, 1)
    }

    assert kernel.ptx.count(
        "cp.async.bulk.tensor.2d.shared::cluster.global"
    ) == 6
    assert kernel.ptx.count(", 49152;") == 2
    assert input_descs[0].tile_sizes == (128, 64)
    assert input_descs[1].tile_sizes == (128, 64)

    with pytest.raises(ValueError, match="require direct"):
        blackwell_tcgen05_gemm_module(
            m=128, n=128, k=64,
            output_n_tile=128, tma_staging=None, split_rhs_tma=True,
        )


def test_tcgen05_n248_tail_lowering():
    kernel = make_ptx_kernel(blackwell_tcgen05_gemm_module(
        m=128, n=4096, k=256,
        output_n_tile=248, tail_n_tile=128,
        operand_stages=8, transpose_cta_grid=True,
        split_tma_barriers=True, split_rhs_tma=True,
        prefetch_lhs_next=True, prefetch_rhs_next=True,
    ))
    input_descs = {
        typ.arg_idx: typ for typ in kernel.signature
        if isinstance(typ, TmaDescType) and typ.arg_idx in (0, 1)
    }

    assert kernel.grid_dim == (17, 1, 1)
    assert input_descs[1].tile_sizes == (124, 64)
    assert "0x083e0490" in kernel.ptx  # M128/N248
    assert "0x08200490" in kernel.ptx  # M128/N128 tail
    assert "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32" in kernel.ptx
    assert ", 256;" in kernel.ptx
    assert "st.global.v8.b32" in kernel.ptx

    assert "st.global.v4.b32" in kernel.ptx

    with pytest.raises(ValueError, match="positive 8-column multiple"):
        blackwell_tcgen05_gemm_module(
            m=128, n=4096, k=256,
            output_n_tile=248, tail_n_tile=66,
            transpose_cta_grid=True,
        )


def test_tcgen05_n240_tail16_lowering():
    kernel = make_ptx_kernel(blackwell_tcgen05_gemm_module(
        m=128, n=4096, k=256,
        output_n_tile=240, tail_n_tile=16,
        operand_stages=8, transpose_cta_grid=True,
        split_tma_barriers=True, split_rhs_tma=True,
        prefetch_lhs_next=True, prefetch_rhs_next=True,
    ))
    input_descs = {
        typ.arg_idx: typ for typ in kernel.signature
        if isinstance(typ, TmaDescType) and typ.arg_idx in (0, 1)
    }

    assert kernel.grid_dim == (18, 1, 1)
    assert input_descs[1].tile_sizes == (120, 64)
    assert "0x083c0490" in kernel.ptx  # M128/N240
    assert "0x08040490" in kernel.ptx  # M128/N16 tail
    # Fifteen stores cover the full N240 tile; the sixteenth is the N16
    # partial-chunk path selected only by the tail CTA.
    assert kernel.ptx.count("st.global.v8.b32") == 16

    predicated = make_ptx_kernel(blackwell_tcgen05_gemm_module(
        m=128, n=4096, k=256,
        output_n_tile=240, tail_n_tile=16,
        predicated_tail_stores=True,
        operand_stages=8, transpose_cta_grid=True,
        split_tma_barriers=True, split_rhs_tma=True,
        prefetch_lhs_next=True, prefetch_rhs_next=True,
    ))
    assert predicated.ptx.count("st.global.v8.b32") == 15
    assert "@!" in predicated.ptx


def test_tcgen05_n232_tail152_lowering():
    kernel = make_ptx_kernel(blackwell_tcgen05_gemm_module(
        m=128, n=4096, k=256,
        output_n_tile=232, tail_n_tile=152,
        operand_stages=8, transpose_cta_grid=True,
        split_tma_barriers=True, split_rhs_tma=True,
        prefetch_lhs_next=True, prefetch_rhs_next=True,
    ))
    input_descs = {
        typ.arg_idx: typ for typ in kernel.signature
        if isinstance(typ, TmaDescType) and typ.arg_idx in (0, 1)
    }

    assert kernel.grid_dim == (18, 1, 1)
    assert input_descs[1].tile_sizes == (116, 64)
    assert "0x083a0490" in kernel.ptx  # M128/N232
    assert "0x08260490" in kernel.ptx  # M128/N152 tail
    assert "st.global.v8.b32" in kernel.ptx
    assert "st.global.v4.b32" in kernel.ptx


def test_tcgen05_balanced_n232_n224_lowering():
    kernel = make_ptx_kernel(blackwell_tcgen05_gemm_module(
        m=128, n=4096, k=256,
        output_n_tile=232, balanced_n_tile=224,
        operand_stages=8, transpose_cta_grid=True,
        split_tma_barriers=True, split_rhs_tma=True,
        prefetch_lhs_next=True, prefetch_rhs_next=True,
    ))
    input_descs = {
        typ.arg_idx: typ for typ in kernel.signature
        if isinstance(typ, TmaDescType) and typ.arg_idx in (0, 1)
    }

    assert kernel.grid_dim == (18, 1, 1)
    assert input_descs[1].tile_sizes == (116, 64)
    assert "0x083a0490" in kernel.ptx  # M128/N232
    assert "0x08380490" in kernel.ptx  # M128/N224
    assert "and.b64" in kernel.ptx

    evict_first = make_ptx_kernel(blackwell_tcgen05_gemm_module(
        m=128, n=4096, k=256,
        output_n_tile=232, balanced_n_tile=224,
        operand_stages=8, transpose_cta_grid=True,
        split_tma_barriers=True, split_rhs_tma=True,
        prefetch_lhs_next=True, prefetch_rhs_next=True,
        output_l2_eviction_priority="evict_first",
    ))
    assert evict_first.ptx.count(
        "st.global.L2::evict_first.v8.b32"
    ) == 28
    assert "st.global.v8.b32" not in evict_first.ptx
    assert evict_first.ptx.count("st.global.v4.b32") == 10

    no_allocate = make_ptx_kernel(blackwell_tcgen05_gemm_module(
        m=128, n=4096, k=256,
        output_n_tile=232, balanced_n_tile=224,
        operand_stages=8, transpose_cta_grid=True,
        split_tma_barriers=True, split_rhs_tma=True,
        prefetch_lhs_next=True, prefetch_rhs_next=True,
        output_l1_cache_policy="no_allocate",
        output_l2_eviction_priority="evict_first",
    ))
    assert no_allocate.ptx.count(
        "st.global.L1::no_allocate.L2::evict_first.v8.b32"
    ) == 28
    assert no_allocate.ptx.count(
        "st.global.L1::no_allocate.v4.b32"
    ) == 10

    paired = make_ptx_kernel(blackwell_tcgen05_gemm_module(
        m=128, n=4096, k=256,
        output_n_tile=232, balanced_n_tile=224,
        operand_stages=8, transpose_cta_grid=True,
        split_tma_barriers=True, split_rhs_tma=True,
        prefetch_lhs_next=True, prefetch_rhs_next=True,
        output_l1_cache_policy="no_allocate",
        output_l2_eviction_priority="evict_first",
        output_tmem_load_shape="16x32bx2",
    ))
    assert paired.ptx.count(
        "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
    ) == 21
    assert paired.ptx.count(
        "tcgen05.ld.sync.aligned.16x32bx2.x4.b32"
    ) == 2
    assert "tcgen05.ld.sync.aligned.32x32b.x64.b32" not in paired.ptx
    assert paired.ptx.count(
        "st.global.L1::no_allocate.L2::evict_first.v8.b32"
    ) == 21
    assert paired.ptx.count("st.global.L1::no_allocate.v2.b32") == 2

    exact_n256_paired = make_ptx_kernel(blackwell_tcgen05_gemm_module(
        m=128, n=256, k=256,
        output_n_tile=256, operand_stages=8,
        output_tmem_load_shape="16x32bx2",
    ))
    assert exact_n256_paired.ptx.count(
        "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
    ) == 8
    assert exact_n256_paired.ptx.count("st.global.v8.b32") == 8

    exact_m64n64_paired = make_ptx_kernel(blackwell_tcgen05_gemm_module(
        m=64, n=64, k=256,
        output_m_tile=64, output_n_tile=64, operand_stages=16,
        output_tmem_load_shape="16x32bx2",
    ))
    assert exact_m64n64_paired.ptx.count(
        "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
    ) == 2
    assert exact_m64n64_paired.ptx.count("st.global.v8.b32") == 2

    with pytest.raises(ValueError, match="balanced_n_tile"):
        blackwell_tcgen05_gemm_module(
            m=128, n=4096, k=256,
            output_n_tile=232, balanced_n_tile=220,
            operand_stages=8, transpose_cta_grid=True,
        )

    with pytest.raises(ValueError, match="output_l2_eviction_priority"):
        blackwell_tcgen05_gemm_module(
            m=128, n=4096, k=256,
            output_n_tile=232, balanced_n_tile=224,
            operand_stages=8, transpose_cta_grid=True,
            output_l2_eviction_priority="persist",
        )

    with pytest.raises(ValueError, match="output_l1_cache_policy"):
        blackwell_tcgen05_gemm_module(
            m=128, n=4096, k=256,
            output_n_tile=232, balanced_n_tile=224,
            operand_stages=8, transpose_cta_grid=True,
            output_l1_cache_policy="streaming",
        )

    with pytest.raises(ValueError, match="output_tmem_load_shape"):
        blackwell_tcgen05_gemm_module(
            m=128, n=4096, k=256,
            output_n_tile=232, balanced_n_tile=224,
            operand_stages=8, transpose_cta_grid=True,
            output_tmem_load_shape="16x64b",
        )

    with pytest.raises(ValueError, match="16x32bx2 output"):
        blackwell_tcgen05_gemm_module(
            m=128, n=4096, k=256,
            output_n_tile=232, balanced_n_tile=216,
            operand_stages=8, transpose_cta_grid=True,
            output_tmem_load_shape="16x32bx2",
        )


def test_tcgen05_persistent_balanced_n_pairs_lowering():
    kernel = make_ptx_kernel(blackwell_tcgen05_gemm_module(
        m=128, n=4096, k=256,
        output_n_tile=232, balanced_n_tile=224,
        operand_stages=8, transpose_cta_grid=True,
        split_tma_barriers=True, split_rhs_tma=True,
        prefetch_lhs_next=True, prefetch_rhs_next=True,
        output_tmem_load_shape="16x32bx2",
        persistent_n_pairs=True,
    ))
    ptx = kernel.ptx

    assert kernel.grid_dim == (9, 1, 1)
    assert "disjoint_grid_iterations: True" in ptx
    assert ptx.count("tcgen05.mma.cta_group::1.kind::f16") == 16
    assert "tcgen05.st.sync" not in ptx
    dealloc = ptx.index("// PTX template `dealloc(")
    assert not ptx[:dealloc].rstrip().endswith("bar.sync 0;")
    overwrite = ptx.index("mov.pred")
    first_mma = ptx.index("tcgen05.mma.cta_group::1", overwrite)
    restore = ptx.index("mov.pred", overwrite + 1)
    assert ptx[overwrite:ptx.index("\n", overwrite)].endswith(", 0;")
    assert ptx[restore:ptx.index("\n", restore)].endswith(", -1;")
    assert overwrite < first_mma < restore

    with pytest.raises(ValueError, match="persistent N pairing"):
        blackwell_tcgen05_gemm_module(
            m=128, n=768, k=256,
            output_n_tile=256, operand_stages=8,
            transpose_cta_grid=True,
            persistent_n_pairs=True,
        )

    invalid = blackwell_tcgen05_gemm_module(
        m=128, n=4096, k=256,
        output_n_tile=232, balanced_n_tile=224,
        operand_stages=8, transpose_cta_grid=True,
        split_tma_barriers=True, split_rhs_tma=True,
        prefetch_lhs_next=True, prefetch_rhs_next=True,
        persistent_n_pairs=True,
    )
    mma_group = next(
        op for op in iter_ops(invalid)
        if isinstance(op, AsyncOp)
        and sum(isinstance(body_op, MatmulOp) for body_op in op.body.ops) > 1
    )
    mmas = [op for op in mma_group.body.ops if isinstance(op, MatmulOp)]
    mmas[1].attrs["overwrite_accumulator"] = True
    with pytest.raises(ValueError, match="only the first tcgen05 matmul"):
        make_ptx_kernel(invalid)


def test_tcgen05_persistent_exact_n_pairs_lowering():
    kernel = make_ptx_kernel(blackwell_tcgen05_gemm_module(
        m=128, n=1024, k=1024,
        output_n_tile=256, operand_stages=8,
        transpose_cta_grid=True,
        split_tma_barriers=True, split_rhs_tma=True,
        prefetch_lhs_next=True, prefetch_rhs_next=True,
        output_tmem_load_shape="16x32bx2",
        persistent_n_pairs=True, lean_bookkeeping=True,
    ))
    ptx = kernel.ptx

    assert kernel.grid_dim == (2, 1, 1)
    assert "disjoint_grid_iterations: True" in ptx
    assert ptx.count("tcgen05.mma.cta_group::1.kind::f16") == 40
    assert ptx.count("cp.async.bulk.tensor.2d.shared::cluster.global") == 30


def test_tcgen05_persistent_pair_reuses_lhs_in_reverse_k_order():
    kernel = make_ptx_kernel(blackwell_tcgen05_gemm_module(
        m=128, n=4096, k=4096,
        output_n_tile=232, balanced_n_tile=224,
        operand_stages=8, transpose_cta_grid=True,
        split_tma_barriers=True, split_rhs_tma=True,
        prefetch_lhs_next=True, prefetch_rhs_next=True,
        output_tmem_load_shape="16x32bx2",
        persistent_n_pairs=True, lean_bookkeeping=True,
        reuse_persistent_lhs=True,
    ))
    ptx = kernel.ptx

    # The two persistent phases are static, but both retain compact grouped
    # K loops. Phase two descends from K128 iteration 30 and reuses K64 banks
    # 62/63, removing exactly two of the original 60 dynamic TMA requests. A
    # terminal half commit releases B0 for the second phase's early prefetch.
    assert "for (%0: index = 30 to 0 step -3)" in ptx
    assert ptx.count("setp.gt.s32") == 1
    assert ptx.count("tcgen05.mma.cta_group::1.kind::f16") == 80
    assert ptx.count("tcgen05.commit.cta_group::1") == 19
    assert ptx.count("cp.async.bulk.tensor.2d.shared::cluster.global") == 58

    with pytest.raises(ValueError, match="persistent LHS reuse requires"):
        blackwell_tcgen05_gemm_module(
            m=128, n=4096, k=4096,
            output_n_tile=232, balanced_n_tile=224,
            operand_stages=8, transpose_cta_grid=True,
            split_tma_barriers=True, split_rhs_tma=True,
            prefetch_lhs_next=True, prefetch_rhs_next=True,
            output_tmem_load_shape="16x32bx2",
            persistent_n_pairs=False, lean_bookkeeping=True,
            reuse_persistent_lhs=True,
        )


@pytest.mark.parametrize("k", [1024, 1152, 1280])
def test_tcgen05_persistent_lhs_reuse_supports_all_k128_phases(k):
    kernel = make_ptx_kernel(blackwell_tcgen05_gemm_module(
        m=128, n=4096, k=k,
        output_n_tile=232, balanced_n_tile=224,
        operand_stages=8, transpose_cta_grid=True,
        split_tma_barriers=True, split_rhs_tma=True,
        prefetch_lhs_next=True, prefetch_rhs_next=True,
        output_tmem_load_shape="16x32bx2",
        persistent_n_pairs=True, lean_bookkeeping=True,
        reuse_persistent_lhs=True,
    ))

    assert kernel.grid_dim == (9, 1, 1)
    assert "persistent_reverse_k_pair" not in kernel.ptx


def test_tcgen05_odd_balanced_n_pairs_lowering():
    kernel = make_ptx_kernel(blackwell_tcgen05_gemm_module(
        m=128, n=688, k=256,
        output_n_tile=232, balanced_n_tile=224,
        operand_stages=8, transpose_cta_grid=True,
        split_tma_barriers=True, split_rhs_tma=True,
        prefetch_lhs_next=True, prefetch_rhs_next=True,
        output_tmem_load_shape="16x32bx2",
        persistent_n_pairs=True, lean_bookkeeping=True,
    ))

    assert kernel.grid_dim == (2, 1, 1)
    assert kernel.ptx.count("tcgen05.mma.cta_group::1.kind::f16") == 16


def test_tcgen05_persistent_m_and_n_pairs_lowering():
    kernel = make_ptx_kernel(blackwell_tcgen05_gemm_module(
        m=512, n=4096, k=256,
        output_n_tile=232, balanced_n_tile=224,
        operand_stages=8, transpose_cta_grid=True,
        split_tma_barriers=True, split_rhs_tma=True,
        prefetch_lhs_next=True, prefetch_rhs_next=True,
        output_tmem_load_shape="16x32bx2",
        persistent_n_pairs=True, persistent_m_pairs=True,
        lean_bookkeeping=True,
    ))

    assert kernel.grid_dim == (9, 2, 1)
    assert "// 2" in kernel.ptx

    reused = make_ptx_kernel(blackwell_tcgen05_gemm_module(
        m=512, n=4096, k=1024,
        output_n_tile=232, balanced_n_tile=224,
        operand_stages=8, transpose_cta_grid=True,
        split_tma_barriers=True, split_rhs_tma=True,
        prefetch_lhs_next=True, prefetch_rhs_next=True,
        output_tmem_load_shape="16x32bx2",
        persistent_n_pairs=True, persistent_m_pairs=True,
        lean_bookkeeping=True, reuse_persistent_lhs=True,
    ))
    assert reused.grid_dim == (9, 2, 1)
    assert reused.ptx.count("cp.async.bulk.tensor.2d.shared::cluster.global") == 58

    rhs_reused = make_ptx_kernel(blackwell_tcgen05_gemm_module(
        m=512, n=4096, k=1024,
        output_n_tile=232, balanced_n_tile=224,
        operand_stages=8, transpose_cta_grid=True,
        split_tma_barriers=True, split_rhs_tma=True,
        prefetch_lhs_next=True, prefetch_rhs_next=True,
        output_tmem_load_shape="16x32bx2",
        persistent_n_pairs=True, persistent_m_pairs=True,
        lean_bookkeeping=True, reuse_persistent_rhs=True,
    ))
    assert rhs_reused.grid_dim == (9, 2, 1)
    assert rhs_reused.ptx.count(
        "cp.async.bulk.tensor.2d.shared::cluster.global"
    ) == 56
    assert "for (%0: index = 6 to 0 step -3)" in rhs_reused.ptx

    rhs_reused_b_major = make_ptx_kernel(blackwell_tcgen05_gemm_module(
        m=512, n=4096, k=1024,
        output_n_tile=232, balanced_n_tile=224,
        operand_stages=8, transpose_cta_grid=False,
        split_tma_barriers=True, split_rhs_tma=True,
        prefetch_lhs_next=True, prefetch_rhs_next=True,
        output_tmem_load_shape="16x32bx2",
        persistent_n_pairs=True, persistent_m_pairs=True,
        lean_bookkeeping=True, reuse_persistent_rhs=True,
    ))
    assert rhs_reused_b_major.grid_dim == (2, 9, 1)
    assert rhs_reused_b_major.ptx.count(
        "cp.async.bulk.tensor.2d.shared::cluster.global"
    ) == 56

    with pytest.raises(ValueError, match="persistent M pairing"):
        blackwell_tcgen05_gemm_module(
            m=128, n=4096, k=256,
            output_n_tile=232, balanced_n_tile=224,
            operand_stages=8, transpose_cta_grid=True,
            split_tma_barriers=True, split_rhs_tma=True,
            prefetch_lhs_next=True, prefetch_rhs_next=True,
            persistent_n_pairs=True, persistent_m_pairs=True,
        )

    with pytest.raises(ValueError, match="persistent RHS reuse requires"):
        blackwell_tcgen05_gemm_module(
            m=512, n=4096, k=1024,
            output_n_tile=232, balanced_n_tile=224,
            operand_stages=8, transpose_cta_grid=True,
            split_tma_barriers=True, split_rhs_tma=True,
            prefetch_lhs_next=True, prefetch_rhs_next=True,
            persistent_n_pairs=True, persistent_m_pairs=False,
            lean_bookkeeping=True, reuse_persistent_rhs=True,
        )


def test_tcgen05_persistent_pair_resident_phase_order_lowering():
    kernel = make_ptx_kernel(blackwell_tcgen05_gemm_module(
        m=4096, n=4096, k=4096,
        output_n_tile=232, balanced_n_tile=224,
        operand_stages=8, transpose_cta_grid=True,
        split_tma_barriers=True, split_rhs_tma=True,
        prefetch_lhs_next=True, prefetch_rhs_next=True,
        output_tmem_load_shape="16x32bx2",
        persistent_n_pairs=True, lean_bookkeeping=True,
        reuse_persistent_lhs=True,
        resident_phase_order_split=148,
    ))
    ptx = kernel.ptx

    assert kernel.grid_dim == (9, 32, 1)
    assert "// 148" in ptx
    assert ptx.count("div.u32") >= 4
    assert ptx.count("tcgen05.mma.cta_group::1.kind::f16") == 80
    assert ptx.count("cp.async.bulk.tensor.2d.shared::cluster.global") == 58

    with pytest.raises(ValueError, match="resident phase-order splitting"):
        blackwell_tcgen05_gemm_module(
            m=128, n=4096, k=256,
            output_n_tile=232, balanced_n_tile=224,
            operand_stages=8, transpose_cta_grid=True,
            split_tma_barriers=True, split_rhs_tma=True,
            prefetch_lhs_next=True, prefetch_rhs_next=True,
            resident_phase_order_split=4,
        )


def test_tcgen05_lean_bookkeeping_lowering():
    kernel = make_ptx_kernel(blackwell_tcgen05_gemm_module(
        m=128, n=4096, k=256,
        output_n_tile=232, balanced_n_tile=224,
        operand_stages=8, transpose_cta_grid=True,
        split_tma_barriers=True, split_rhs_tma=True,
        prefetch_lhs_next=True, prefetch_rhs_next=True,
        output_tmem_load_shape="16x32bx2",
        persistent_n_pairs=True, lean_bookkeeping=True,
    ))
    ptx = kernel.ptx

    # One predicate, formed next to the required warp-collective tmem alloc,
    # serves every instruction-mandated single-thread issue site. The four
    # non-root descriptors in each K64 bank are integer recurrences from the
    # root descriptor, and next A0/B0 share one 45 KiB completion phase.
    assert ptx.count("elect.sync") == 0
    assert ptx.count(".shared .align 8 .b64") == 4
    assert ptx.count("mbarrier.arrive.expect_tx") == 4
    assert ptx.count("], 46080;") == 4
    assert ptx.count("], 16384;") == 0
    assert ptx.count("add.u64") == 35
    assert ptx.count("tcgen05.mma.cta_group::1.kind::f16") == 16
    assert not re.search(r"@%p\d+ tcgen05\.mma\.cta_group::1", ptx)
    assert "\n  tcgen05.mma.cta_group::1.kind::f16" in ptx
    assert re.search(r"@%p\d+ tcgen05\.commit\.cta_group::1", ptx)

    with pytest.raises(ValueError, match="lean bookkeeping requires"):
        blackwell_tcgen05_gemm_module(
            m=128, n=4096, k=256,
            output_n_tile=232, balanced_n_tile=224,
            operand_stages=8, transpose_cta_grid=True,
            split_tma_barriers=True, split_rhs_tma=True,
            prefetch_lhs_next=True, prefetch_rhs_next=False,
            persistent_n_pairs=True, lean_bookkeeping=True,
        )

    invalid_arrival = blackwell_tcgen05_gemm_module(
        m=128, n=4096, k=256,
        output_n_tile=232, balanced_n_tile=224,
        operand_stages=8, transpose_cta_grid=True,
        split_tma_barriers=True, split_rhs_tma=True,
        prefetch_lhs_next=True, prefetch_rhs_next=True,
        persistent_n_pairs=True, lean_bookkeeping=True,
    )
    deferred = next(
        op for op in iter_ops(invalid_arrival)
        if isinstance(op, AsyncOp) and op.attrs.get("defer_tma_arrival")
    )
    deferred.attrs["tma_extra_expected_bytes"] = 1
    with pytest.raises(ValueError, match="deferred TMA arrival"):
        make_ptx_kernel(invalid_arrival)

    invalid_descriptors = blackwell_tcgen05_gemm_module(
        m=128, n=128, k=64,
        output_n_tile=128, operand_stages=4,
    )
    mma_group = next(
        op for op in iter_ops(invalid_descriptors)
        if isinstance(op, AsyncOp)
        and any(isinstance(body_op, MatmulOp) for body_op in op.body.ops)
    )
    mma_group.attrs["derive_tcgen05_descriptors"] = True
    mma_group.body.ops.remove(next(
        op for op in reversed(mma_group.body.ops) if isinstance(op, MatmulOp)
    ))
    with pytest.raises(ValueError, match="complete four-MMA"):
        make_ptx_kernel(invalid_descriptors)


def test_tcgen05_m64_lean_bookkeeping_deferred_bytes():
    module = blackwell_tcgen05_gemm_module(
        m=64, n=64, k=256,
        output_m_tile=64, output_n_tile=64,
        operand_stages=8, transpose_cta_grid=True,
        split_tma_barriers=True, split_rhs_tma=True,
        prefetch_lhs_next=True, prefetch_rhs_next=True,
        lean_bookkeeping=True, overwrite_first_mma=True,
    )

    joined = [
        op for op in iter_ops(module)
        if isinstance(op, AsyncOp)
        and "tma_extra_expected_bytes" in op.attrs
    ]
    assert joined
    assert {
        op.attrs["tma_extra_expected_bytes"] for op in joined
    } == {64 * 64 * 2}


@requires_blackwell
@pytest.mark.parametrize("seed", [0, 1, 7])
def test_tcgen05_m64_lean_bookkeeping_gemm(seed):
    kernel = make_ptx_kernel(blackwell_tcgen05_gemm_module(
        m=64, n=64, k=256,
        output_m_tile=64, output_n_tile=64,
        operand_stages=8, transpose_cta_grid=True,
        split_tma_barriers=True, split_rhs_tma=True,
        prefetch_lhs_next=True, prefetch_rhs_next=True,
        lean_bookkeeping=True, overwrite_first_mma=True,
    ))

    torch.manual_seed(seed)
    a_tensor = torch.randn((64, 256), dtype=torch.bfloat16, device="cuda")
    b_tensor = torch.randn((64, 256), dtype=torch.bfloat16, device="cuda")
    actual = torch.empty((64, 64), dtype=torch.bfloat16, device="cuda")
    make_ptx_launcher(kernel)(a_tensor, b_tensor, actual)
    torch.cuda.synchronize()

    expected = torch.mm(a_tensor, b_tensor.T)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=2.0)


def test_tcgen05_next_lhs_prefetch_lowering():
    kernel = make_ptx_kernel(blackwell_tcgen05_gemm_module(
        m=128, n=512, k=4096,
        output_n_tile=256, operand_stages=8,
        transpose_cta_grid=True, split_n_tile=0,
        split_tma_barriers=True, split_rhs_tma=True,
        prefetch_lhs_next=True,
    ))
    ptx = kernel.ptx

    # Three 16 KiB A buffers plus two 32 KiB B buffers stay at the exact
    # two-CTA shared-memory boundary by placing the four barriers before them.
    assert ptx.count(".shared .align 1024 .b8") == 5
    first_barrier = ptx.index(".shared .align 8 .b64")
    first_operand = ptx.index(".shared .align 1024 .b8")
    assert first_barrier < first_operand
    assert ptx.count("mbarrier.arrive.expect_tx") == 14
    assert ptx.count("], 16384;") == 4
    assert ptx.count("], 32768;") == 4
    assert ptx.count("tcgen05.mma.cta_group::1.kind::f16") == 40
    assert "for (" in ptx and "step 3" in ptx

    with pytest.raises(ValueError, match="next-iteration LHS prefetch"):
        blackwell_tcgen05_gemm_module(
            m=128, n=512, k=256,
            output_n_tile=256, operand_stages=8,
            transpose_cta_grid=True, split_n_tile=0,
            prefetch_lhs_next=True,
        )


def test_tcgen05_next_rhs_prefetch_lowering():
    module = blackwell_tcgen05_gemm_module(
        m=128, n=512, k=256,
        output_n_tile=256, operand_stages=8,
        transpose_cta_grid=True, split_n_tile=0,
        split_tma_barriers=True, split_rhs_tma=True,
        prefetch_lhs_next=True, prefetch_rhs_next=True,
    )
    assert any(
        isinstance(op, AsyncCommitOp)
        for op in iter_ops(module)
    )
    ptx = make_ptx_kernel(module).ptx

    # Five barriers still fit before the 1 KiB-aligned operands. The first
    # iteration issues B0 normally, then one partial tcgen commit lets the next
    # iteration's two B0 requests overlap the second four MMAs.
    assert ptx.count(".shared .align 8 .b64") == 5
    assert ptx.count("tcgen05.commit.cta_group::1") == 3
    assert ptx.count("mbarrier.arrive.expect_tx") == 5
    assert ptx.count("], 32768;") == 1
    assert ptx.count("tcgen05.mma.cta_group::1.kind::f16") == 16
    first_partial_commit = ptx.index("tcgen05.commit.cta_group::1")
    overlapped_b0_arrive = ptx.index("], 32768;")
    assert "bar.sync" not in ptx[first_partial_commit:overlapped_b0_arrive]

    with pytest.raises(ValueError, match="next-iteration RHS prefetch"):
        blackwell_tcgen05_gemm_module(
            m=128, n=512, k=256,
            output_n_tile=256, operand_stages=8,
            split_tma_barriers=True, split_rhs_tma=True,
            prefetch_rhs_next=True,
        )


def test_tcgen05_partial_commit_completion_mode_is_per_lock():
    module = blackwell_tcgen05_gemm_module(
        m=128, n=512, k=256,
        output_n_tile=256, operand_stages=8,
        transpose_cta_grid=True, split_n_tile=0,
        split_tma_barriers=True, split_rhs_tma=True,
        prefetch_lhs_next=True, prefetch_rhs_next=True,
    )
    commit = next(
        op for op in iter_ops(module) if isinstance(op, AsyncCommitOp)
    )
    commit.lock.type.attrs.update({"arrival_count": 2, "multicast_mask": 0x3})

    ptx = make_ptx_kernel(module).ptx
    multicast = (
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one."
        "shared::cluster.multicast::cluster.b64"
    )
    local = (
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one."
        "shared::cluster.b64"
    )
    assert ptx.count(multicast) == 1
    assert ptx.count(local) == 2


def test_tcgen05_cluster_phase_order_split_requires_cluster_alignment():
    common = dict(
        m=2048, n=1024, k=256,
        output_n_tile=256, operand_stages=8,
        transpose_cta_grid=True,
        split_tma_barriers=True, split_rhs_tma=True,
        prefetch_lhs_next=True, prefetch_rhs_next=True,
        lean_bookkeeping=True, persistent_n_pairs=True,
        cluster_multicast_b=True,
        cluster_multicast_b_ctas=8,
        cluster_multicast_b_issuers=4,
    )
    with pytest.raises(ValueError, match="complete CTA-cluster row"):
        blackwell_tcgen05_gemm_module(
            **common, resident_phase_order_split=20,
        )
    blackwell_tcgen05_gemm_module(
        **common, resident_phase_order_split=16,
    )


@pytest.mark.parametrize("invalid", ["kind", "position"])
def test_tcgen05_partial_commit_verifier(invalid):
    module = blackwell_tcgen05_gemm_module(
        m=128, n=512, k=256,
        output_n_tile=256, operand_stages=8,
        transpose_cta_grid=True, split_n_tile=0,
        split_tma_barriers=True, split_rhs_tma=True,
        prefetch_lhs_next=True, prefetch_rhs_next=True,
    )
    commit = next(
        op for op in iter_ops(module) if isinstance(op, AsyncCommitOp)
    )
    if invalid == "kind":
        mbar_lock = next(
            op.result() for op in iter_ops(module)
            if isinstance(op, AllocOp)
            and isinstance(op.result().type, LockType)
            and op.result().type.kind == "mbar"
        )
        commit.operands[0] = mbar_lock
    else:
        assert commit.parent_block is not None
        body = commit.parent_block.ops
        body.remove(commit)
        body.insert(0, commit)

    with pytest.raises(ValueError, match="partial commits"):
        make_ptx_kernel(module)


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


def test_tcgen05_selected_n_tile_split_lowering():
    kernel = make_ptx_kernel(blackwell_tcgen05_gemm_module(
        m=128, n=4096, k=128,
        output_n_tile=256, operand_stages=8,
        transpose_cta_grid=True, split_n_tile=0,
    ))

    assert kernel.grid_dim == (17, 1, 1)
    assert "setp.gt.u32" in kernel.ptx and ", 0;" in kernel.ptx
    assert "0x08400490" in kernel.ptx
    assert "0x08200490" in kernel.ptx

    with pytest.raises(ValueError, match="existing logical N tile"):
        blackwell_tcgen05_gemm_module(
            m=128, n=4096, k=128,
            output_n_tile=256, operand_stages=8,
            transpose_cta_grid=True, split_n_tile=16,
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


def test_tcgen05_strided_output():
    module = blackwell_tcgen05_gemm_module(
        m=128, n=256, k=128,
        output_n_tile=256, operand_stages=8,
        output_row_stride=512,
    )

    assert "gmem<128x256xbf16 strides=[512, 1]>" in str(module)
    with pytest.raises(ValueError, match="at least n"):
        blackwell_tcgen05_gemm_module(
            m=128, n=256, k=128,
            output_n_tile=256, operand_stages=8,
            output_row_stride=128,
        )


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


def test_tcgen05_gemm_cluster_b_multicast_lowering():
    kernel = make_ptx_kernel(blackwell_tcgen05_gemm_module(
        m=256, n=512, k=1024,
        output_m_tile=128, output_n_tile=256, operand_stages=8,
        transpose_cta_grid=True,
        split_tma_barriers=True, split_rhs_tma=True,
        prefetch_lhs_next=True, prefetch_rhs_next=True,
        lean_bookkeeping=True, persistent_n_pairs=True,
        cluster_multicast_b=True,
        output_tmem_load_shape="16x32bx2",
    ))

    assert kernel.grid_dim == (1, 2, 1)
    assert kernel.cluster_dim == (1, 2, 1)
    assert ".reqnctapercluster 1, 2, 1" in kernel.ptx
    assert kernel.ptx.count("barrier.cluster.arrive.aligned") == 1
    assert kernel.ptx.count("mbarrier.init.shared::cta.b64") == 4
    assert len(re.findall(
        r"mbarrier\.init\.shared::cta\.b64 \[%r\d+\], 2;", kernel.ptx,
    )) == 2
    assert kernel.ptx.count(
        "mbarrier::complete_tx::bytes.multicast::cluster"
    ) == 20
    assert kernel.ptx.count(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one."
        "shared::cluster.multicast::cluster.b64"
    ) == 9
    assert "setp.eq.u32" in kernel.ptx and "%cluster_ctarank" in kernel.ptx


def test_tcgen05_gemm_cluster_b4_multicast_lowering():
    kernel = make_ptx_kernel(blackwell_tcgen05_gemm_module(
        m=512, n=512, k=1024,
        output_m_tile=128, output_n_tile=256, operand_stages=8,
        transpose_cta_grid=True,
        split_tma_barriers=True, split_rhs_tma=True,
        prefetch_lhs_next=True, prefetch_rhs_next=True,
        lean_bookkeeping=True, persistent_n_pairs=True,
        cluster_multicast_b=True, cluster_multicast_b_ctas=4,
        output_tmem_load_shape="16x32bx2",
    ))

    assert kernel.grid_dim == (1, 4, 1)
    assert kernel.cluster_dim == (1, 4, 1)
    assert ".reqnctapercluster 1, 4, 1" in kernel.ptx
    assert kernel.ptx.count("barrier.cluster.arrive.aligned") == 1
    assert len(re.findall(
        r"mbarrier\.init\.shared::cta\.b64 \[%r\d+\], 4;", kernel.ptx,
    )) == 2
    assert "0xf" in kernel.ptx
    for rank in range(4):
        assert f", {rank};" in kernel.ptx


def test_tcgen05_gemm_cluster_b8_multicast_lowering():
    kernel = make_ptx_kernel(blackwell_tcgen05_gemm_module(
        m=1024, n=512, k=1024,
        output_m_tile=128, output_n_tile=256, operand_stages=8,
        transpose_cta_grid=True,
        split_tma_barriers=True, split_rhs_tma=True,
        prefetch_lhs_next=True, prefetch_rhs_next=True,
        lean_bookkeeping=True, persistent_n_pairs=True,
        cluster_multicast_b=True, cluster_multicast_b_ctas=8,
        cluster_multicast_b_issuers=4,
        output_tmem_load_shape="16x32bx2",
    ))

    assert kernel.grid_dim == (1, 8, 1)
    assert kernel.cluster_dim == (1, 8, 1)
    assert ".reqnctapercluster 1, 8, 1" in kernel.ptx
    assert len(re.findall(
        r"mbarrier\.init\.shared::cta\.b64 \[%r\d+\], 8;", kernel.ptx,
    )) == 2
    assert "0xff" in kernel.ptx
    assert kernel.ptx.count(
        "mbarrier::complete_tx::bytes.multicast::cluster"
    ) == 40
    for rank in range(4):
        assert f", {rank};" in kernel.ptx


def test_tcgen05_gemm_cluster_b8_single_output_lowering():
    kernel = make_ptx_kernel(blackwell_tcgen05_gemm_module(
        m=1024, n=256, k=1024,
        output_m_tile=128, output_n_tile=256, operand_stages=8,
        transpose_cta_grid=True,
        split_tma_barriers=True, split_rhs_tma=True,
        prefetch_lhs_next=True, prefetch_rhs_next=True,
        lean_bookkeeping=True, overwrite_first_mma=True,
        cluster_multicast_b=True, cluster_multicast_b_ctas=8,
        cluster_multicast_b_issuers=4,
        output_tmem_load_shape="16x32bx2",
    ))

    assert kernel.grid_dim == (1, 8, 1)
    assert kernel.cluster_dim == (1, 8, 1)
    assert ".reqnctapercluster 1, 8, 1" in kernel.ptx
    assert kernel.ptx.count("barrier.cluster.arrive.aligned") == 1
    assert len(re.findall(
        r"mbarrier\.init\.shared::cta\.b64 \[%r\d+\], 8;", kernel.ptx,
    )) == 2
    assert "0xff" in kernel.ptx

    with pytest.raises(ValueError, match="first-MMA overwrite"):
        blackwell_tcgen05_gemm_module(
            m=1024, n=256, k=1024,
            output_m_tile=128, output_n_tile=256, operand_stages=8,
            transpose_cta_grid=True,
            split_tma_barriers=True, split_rhs_tma=True,
            prefetch_lhs_next=True, prefetch_rhs_next=True,
            lean_bookkeeping=True,
            cluster_multicast_b=True, cluster_multicast_b_ctas=8,
            cluster_multicast_b_issuers=4,
            output_tmem_load_shape="16x32bx2",
        )


def test_tcgen05_gemm_cluster_b16_single_output_lowering():
    kernel = make_ptx_kernel(blackwell_tcgen05_gemm_module(
        m=2048, n=256, k=1024,
        output_m_tile=128, output_n_tile=256, operand_stages=8,
        transpose_cta_grid=True,
        split_tma_barriers=True, split_rhs_tma=True,
        prefetch_lhs_next=True, prefetch_rhs_next=True,
        lean_bookkeeping=True, overwrite_first_mma=True,
        cluster_multicast_b=True, cluster_multicast_b_ctas=16,
        cluster_multicast_b_issuers=4,
        output_tmem_load_shape="16x32bx2",
    ))

    assert kernel.grid_dim == (1, 16, 1)
    assert kernel.cluster_dim == (1, 16, 1)
    assert ".reqnctapercluster 1, 16, 1" in kernel.ptx
    assert len(re.findall(
        r"mbarrier\.init\.shared::cta\.b64 \[%r\d+\], 16;", kernel.ptx,
    )) == 2
    assert "0xffff" in kernel.ptx
    assert kernel.ptx.count(
        "mbarrier::complete_tx::bytes.multicast::cluster"
    ) == 40

    issuer_only = make_ptx_kernel(blackwell_tcgen05_gemm_module(
        m=2048, n=256, k=1024,
        output_m_tile=128, output_n_tile=256, operand_stages=8,
        transpose_cta_grid=True,
        split_tma_barriers=True, split_rhs_tma=True,
        prefetch_lhs_next=True, prefetch_rhs_next=True,
        lean_bookkeeping=True, overwrite_first_mma=True,
        cluster_multicast_b=True, cluster_multicast_b_ctas=16,
        cluster_multicast_b_issuers=4,
        cluster_completion_issuers_only=True,
        output_tmem_load_shape="16x32bx2",
    )).ptx
    assert issuer_only.count("selp.b32") >= 2
    assert issuer_only.count(
        ".reg .b16 %ayla_cluster_completion_mask;"
    ) == 1
    assert issuer_only.count(
        "mov.u32 %ayla_cluster_completion_rank, %cluster_ctarank;"
    ) == 1
    assert issuer_only.count("shl.b32") >= 1
    assert "0xf;" in issuer_only

    with pytest.raises(ValueError, match="issuer-only cluster completion"):
        blackwell_tcgen05_gemm_module(
            cluster_completion_issuers_only=True,
        )


def test_tcgen05_gemm_branches_nonissuers_around_issue_sequence():
    ptx = make_ptx_kernel(blackwell_tcgen05_gemm_module(
        m=128, n=128, k=64,
    )).ptx

    branch = ptx.index("bra $L")
    descriptor = ptx.index("mov.b32", branch)
    mma = ptx.index("tcgen05.mma.cta_group::1", descriptor)
    join = ptx.index("$L", mma)
    assert branch < descriptor < mma < join


def test_tcgen05_exact_n128_paired_epilogue_lowering():
    kernel = make_ptx_kernel(blackwell_tcgen05_gemm_module(
        m=128, n=128, k=128,
        output_n_tile=128, operand_stages=8,
        output_tmem_load_shape="16x32bx2",
    ))
    assert kernel.ptx.count("tcgen05.ld.sync.aligned.16x32bx2.x16.b32") == 4
    assert "tcgen05.ld.sync.aligned.16x32bx2.x4.b32" not in kernel.ptx


def test_tcgen05_m64_tma_staging_lowering():
    kernel = make_ptx_kernel(blackwell_tcgen05_gemm_module(
        m=128, n=128, k=64, output_m_tile=64,
    ))

    assert kernel.grid_dim == (2, 1, 1)
    assert "mov.b32 %r" in kernel.ptx and "0x04200490" in kernel.ptx
    assert "setp.lt.u32" in kernel.ptx
    assert "and.b32" in kernel.ptx
    assert "@%p" in kernel.ptx and "st.global.v8.b32" in kernel.ptx


@pytest.mark.parametrize("stages", [8, 16, 32])
def test_tcgen05_narrow_tma_staging_lowering(stages):
    n = 32
    kernel = make_ptx_kernel(blackwell_tcgen05_gemm_module(
        m=64, n=n, k=512, output_m_tile=64, output_n_tile=n,
        operand_stages=stages, split_tma_barriers=stages == 8,
    ))

    assert kernel.grid_dim == (1, 1, 1)
    assert f"0x{((4 << 24) | ((n // 8) << 17) | 0x490):08x}" in kernel.ptx
    assert f"tcgen05.ld.sync.aligned.32x32b.x{n}.b32" in kernel.ptx
    assert "tcgen05.ld.sync.aligned.32x32b.x64.b32" not in kernel.ptx
    assert f"st.global.v{min(8, n // 2)}.b32" in kernel.ptx


def test_tcgen05_cta_group2_lowering():
    ptx = _make_tcgen05_cta_group2_kernel().ptx

    assert ".reqnctapercluster 2, 1, 1" in ptx
    assert "tcgen05.alloc.cta_group::2" in ptx
    assert "tcgen05.mma.cta_group::2.kind::f16" in ptx
    assert "multicast::cluster" in ptx
    assert "tcgen05.dealloc.cta_group::2" in ptx


def test_tcgen05_cta_group2_m128_n256_k128_lowering():
    kernel = make_ptx_kernel(blackwell_tcgen05_cta_group2_gemm_module(
        m=128, n=256, k=128,
        output_m_tile=128, output_n_tile=256, operand_stages=8,
    ))

    assert kernel.grid_dim == (2, 1, 1)
    assert kernel.cluster_dim == (2, 1, 1)
    assert kernel.ptx.count("tcgen05.mma.cta_group::2.kind::f16") == 8
    assert "0x08400490" in kernel.ptx  # M128/N256
    assert "2-SM M128 accumulator: 2x2 TMEM datapath mapping" in kernel.ptx


@requires_blackwell
def test_tcgen05_cta_group2_m128_n256_k128_gemm():
    kernel = make_ptx_kernel(blackwell_tcgen05_cta_group2_gemm_module(
        m=256, n=256, k=128,
        output_m_tile=128, output_n_tile=256, operand_stages=8,
    ))
    launcher = make_ptx_launcher(kernel)

    for seed in (0, 1, 7):
        torch.manual_seed(seed)
        a_tensor = torch.randn(
            (256, 128), dtype=torch.bfloat16, device="cuda",
        )
        b_tensor = torch.randn(
            (256, 128), dtype=torch.bfloat16, device="cuda",
        )
        actual = torch.empty(
            (256, 256), dtype=torch.bfloat16, device="cuda",
        )
        launcher(a_tensor, b_tensor, actual)
        torch.cuda.synchronize()

        expected = a_tensor @ b_tensor.T
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, rtol=0, atol=2.0)


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
@pytest.mark.parametrize("seed,stages", [(0, 8), (1, 16), (7, 32)])
def test_tcgen05_m64_narrow_gemm(seed, stages):
    n = 32
    kernel = make_ptx_kernel(blackwell_tcgen05_gemm_module(
        m=64, n=n, k=512, output_m_tile=64, output_n_tile=n,
        operand_stages=stages, split_tma_barriers=stages == 8,
    ))

    torch.manual_seed(seed)
    a_tensor = torch.randn((64, 512), dtype=torch.bfloat16, device="cuda")
    b_tensor = torch.randn((n, 512), dtype=torch.bfloat16, device="cuda")
    actual = torch.empty((64, n), dtype=torch.bfloat16, device="cuda")
    make_ptx_launcher(kernel)(a_tensor, b_tensor, actual)
    torch.cuda.synchronize()

    expected = torch.mm(a_tensor, b_tensor.T)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=2.0)


@requires_blackwell
@pytest.mark.parametrize("seed", [0, 1, 7])
def test_tcgen05_odd_persistent_pair_gemm(seed):
    kernel = make_ptx_kernel(blackwell_tcgen05_gemm_module(
        m=128, n=688, k=256,
        output_n_tile=232, balanced_n_tile=224,
        operand_stages=8, transpose_cta_grid=True,
        split_tma_barriers=True, split_rhs_tma=True,
        prefetch_lhs_next=True, prefetch_rhs_next=True,
        output_tmem_load_shape="16x32bx2",
        persistent_n_pairs=True, lean_bookkeeping=True,
    ))

    torch.manual_seed(seed)
    a_tensor = torch.randn((128, 256), dtype=torch.bfloat16, device="cuda")
    b_tensor = torch.randn((688, 256), dtype=torch.bfloat16, device="cuda")
    actual = torch.empty((128, 688), dtype=torch.bfloat16, device="cuda")
    make_ptx_launcher(kernel)(a_tensor, b_tensor, actual)
    torch.cuda.synchronize()

    expected = torch.mm(a_tensor, b_tensor.T)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=2.0)


@requires_blackwell
@pytest.mark.parametrize("seed,k", [(0, 1024), (1, 1152), (7, 1280)])
def test_tcgen05_generalized_persistent_lhs_reuse_gemm(seed, k):
    kernel = make_ptx_kernel(blackwell_tcgen05_gemm_module(
        m=128, n=4096, k=k,
        output_n_tile=232, balanced_n_tile=224,
        operand_stages=8, transpose_cta_grid=True,
        split_tma_barriers=True, split_rhs_tma=True,
        prefetch_lhs_next=True, prefetch_rhs_next=True,
        output_tmem_load_shape="16x32bx2",
        persistent_n_pairs=True, lean_bookkeeping=True,
        reuse_persistent_lhs=True,
    ))

    torch.manual_seed(seed)
    a_tensor = torch.randn((128, k), dtype=torch.bfloat16, device="cuda")
    b_tensor = torch.randn((4096, k), dtype=torch.bfloat16, device="cuda")
    actual = torch.empty((128, 4096), dtype=torch.bfloat16, device="cuda")
    make_ptx_launcher(kernel)(a_tensor, b_tensor, actual)
    torch.cuda.synchronize()

    expected = torch.mm(a_tensor, b_tensor.T)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=2.0)


@requires_blackwell
@pytest.mark.parametrize("transpose_cta_grid", [True, False])
@pytest.mark.parametrize("seed,k", [(0, 1024), (1, 1152), (7, 1280)])
def test_tcgen05_generalized_persistent_rhs_reuse_gemm(
    seed, k, transpose_cta_grid,
):
    kernel = make_ptx_kernel(blackwell_tcgen05_gemm_module(
        m=256, n=4096, k=k,
        output_n_tile=232, balanced_n_tile=224,
        operand_stages=8, transpose_cta_grid=transpose_cta_grid,
        split_tma_barriers=True, split_rhs_tma=True,
        prefetch_lhs_next=True, prefetch_rhs_next=True,
        output_tmem_load_shape="16x32bx2",
        persistent_n_pairs=True, persistent_m_pairs=True,
        lean_bookkeeping=True, reuse_persistent_rhs=True,
    ))

    torch.manual_seed(seed)
    a_tensor = torch.randn((256, k), dtype=torch.bfloat16, device="cuda")
    b_tensor = torch.randn((4096, k), dtype=torch.bfloat16, device="cuda")
    actual = torch.empty((256, 4096), dtype=torch.bfloat16, device="cuda")
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
def test_tcgen05_next_lhs_prefetch_gemm(seed):
    kernel = make_ptx_kernel(blackwell_tcgen05_gemm_module(
        m=128, n=512, k=256,
        output_n_tile=256, operand_stages=8,
        transpose_cta_grid=True, split_n_tile=0,
        split_tma_barriers=True, split_rhs_tma=True,
        prefetch_lhs_next=True,
    ))

    torch.manual_seed(seed)
    a_tensor = torch.randn((128, 256), dtype=torch.bfloat16, device="cuda")
    b_tensor = torch.randn((512, 256), dtype=torch.bfloat16, device="cuda")
    actual = torch.empty((128, 512), dtype=torch.bfloat16, device="cuda")
    make_ptx_launcher(kernel)(a_tensor, b_tensor, actual)
    torch.cuda.synchronize()

    expected = torch.mm(a_tensor, b_tensor.T)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=2.0)


@requires_blackwell
@pytest.mark.parametrize("seed", [0, 1, 7])
def test_tcgen05_next_rhs_prefetch_gemm(seed):
    kernel = make_ptx_kernel(blackwell_tcgen05_gemm_module(
        m=128, n=512, k=256,
        output_n_tile=256, operand_stages=8,
        transpose_cta_grid=True, split_n_tile=0,
        split_tma_barriers=True, split_rhs_tma=True,
        prefetch_lhs_next=True, prefetch_rhs_next=True,
    ))

    torch.manual_seed(seed)
    a_tensor = torch.randn((128, 256), dtype=torch.bfloat16, device="cuda")
    b_tensor = torch.randn((512, 256), dtype=torch.bfloat16, device="cuda")
    actual = torch.empty((128, 512), dtype=torch.bfloat16, device="cuda")
    make_ptx_launcher(kernel)(a_tensor, b_tensor, actual)
    torch.cuda.synchronize()

    expected = torch.mm(a_tensor, b_tensor.T)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=2.0)


@requires_blackwell
@pytest.mark.parametrize("seed", [0, 1, 7])
def test_tcgen05_balanced_n232_n224_gemm(seed):
    kernel = make_ptx_kernel(blackwell_tcgen05_gemm_module(
        m=128, n=4096, k=256,
        output_n_tile=232, balanced_n_tile=224,
        operand_stages=8, transpose_cta_grid=True,
        split_tma_barriers=True, split_rhs_tma=True,
        prefetch_lhs_next=True, prefetch_rhs_next=True,
    ))

    torch.manual_seed(seed)
    a_tensor = torch.randn((128, 256), dtype=torch.bfloat16, device="cuda")
    b_tensor = torch.randn((4096, 256), dtype=torch.bfloat16, device="cuda")
    actual = torch.empty((128, 4096), dtype=torch.bfloat16, device="cuda")
    make_ptx_launcher(kernel)(a_tensor, b_tensor, actual)
    torch.cuda.synchronize()

    expected = torch.mm(a_tensor, b_tensor.T)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=2.0)


@requires_blackwell
@pytest.mark.parametrize("seed", [0, 1, 7])
def test_tcgen05_persistent_balanced_n_pairs_gemm(seed):
    kernel = make_ptx_kernel(blackwell_tcgen05_gemm_module(
        m=128, n=4096, k=256,
        output_n_tile=232, balanced_n_tile=224,
        operand_stages=8, transpose_cta_grid=True,
        split_tma_barriers=True, split_rhs_tma=True,
        prefetch_lhs_next=True, prefetch_rhs_next=True,
        output_tmem_load_shape="16x32bx2",
        persistent_n_pairs=True,
    ))

    torch.manual_seed(seed)
    a_tensor = torch.randn((128, 256), dtype=torch.bfloat16, device="cuda")
    b_tensor = torch.randn((4096, 256), dtype=torch.bfloat16, device="cuda")
    actual = torch.empty((128, 4096), dtype=torch.bfloat16, device="cuda")
    make_ptx_launcher(kernel)(a_tensor, b_tensor, actual)
    torch.cuda.synchronize()

    expected = torch.mm(a_tensor, b_tensor.T)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=2.0)


@requires_blackwell
@pytest.mark.parametrize("seed", [0, 1, 7])
def test_tcgen05_persistent_exact_n256_pairs_gemm(seed):
    kernel = make_ptx_kernel(blackwell_tcgen05_gemm_module(
        m=128, n=1024, k=1024,
        output_n_tile=256, operand_stages=8,
        transpose_cta_grid=True,
        split_tma_barriers=True, split_rhs_tma=True,
        prefetch_lhs_next=True, prefetch_rhs_next=True,
        output_tmem_load_shape="16x32bx2",
        persistent_n_pairs=True, lean_bookkeeping=True,
    ))

    torch.manual_seed(seed)
    a_tensor = torch.randn((128, 1024), dtype=torch.bfloat16, device="cuda")
    b_tensor = torch.randn((1024, 1024), dtype=torch.bfloat16, device="cuda")
    actual = torch.empty((128, 1024), dtype=torch.bfloat16, device="cuda")
    make_ptx_launcher(kernel)(a_tensor, b_tensor, actual)
    torch.cuda.synchronize()

    expected = torch.mm(a_tensor, b_tensor.T)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=2.0)


@requires_blackwell
@pytest.mark.parametrize("seed", [0, 1, 7])
def test_tcgen05_cluster_b_multicast_gemm(seed):
    kernel = make_ptx_kernel(blackwell_tcgen05_gemm_module(
        m=256, n=512, k=1024,
        output_m_tile=128, output_n_tile=256, operand_stages=8,
        transpose_cta_grid=True,
        split_tma_barriers=True, split_rhs_tma=True,
        prefetch_lhs_next=True, prefetch_rhs_next=True,
        lean_bookkeeping=True, persistent_n_pairs=True,
        cluster_multicast_b=True,
        output_tmem_load_shape="16x32bx2",
    ))

    torch.manual_seed(seed)
    a_tensor = torch.randn((256, 1024), dtype=torch.bfloat16, device="cuda")
    b_tensor = torch.randn((512, 1024), dtype=torch.bfloat16, device="cuda")
    actual = torch.empty((256, 512), dtype=torch.bfloat16, device="cuda")
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
