"""Two resident single-warpgroup task programs sharing operand stages."""

import pytest
import torch

from ayla.base.ir import Block
from ayla.gemm_collection.gemm import auto_matmul_4096_wgmma256_wg_tma_store_module
from ayla.gpu_dialect import (
    AllocOp,
    AwaitOp,
    BarrierOp,
    ForOp,
    IndexConstantOp,
    IndexType,
    KernelOp,
    LockType,
    ModuleOp,
    PTXTemplateOp,
)
from ayla.passes.pair_warpgroup_ctas import PairWarpgroupCtas, shared_operand_stages
from ayla.ptx_codegen import PtxLoweringPipeline, make_ptx_kernel
from ayla.ptx_runtime import make_ptx_launcher


def module(
    m=1024,
    n=336,
    k=1024,
    dtype="e4m3",
    output="bf16",
    grid=8,
    ctas=2,
    cyclic=True,
    stages=4,
):
    fp8 = dtype in ("e4m3", "e5m2")
    return auto_matmul_4096_wgmma256_wg_tma_store_module(
        m=m,
        n=n,
        k=k,
        dtype=dtype,
        output_dtype=output,
        scaled=fp8,
        output_m_tile=128,
        warpgroup_m_tile=128,
        output_n_tile=168,
        warpgroup_n_tile=168,
        operand_k_tile=128 if fp8 else 64,
        operand_stages=stages,
        cyclic_operand_pipeline=cyclic,
        output_grid_size=grid,
        output_grid_fast_axis="m",
        output_fragment_cols=64,
        output_fragment_swizzle="128B",
        output_store_buffers=2,
        warpgroup_ctas=ctas,
    )


def paired(mod):
    """Run the lowering pipeline through the pairing pass."""
    for step in PtxLoweringPipeline().passes:
        mod = step.run(mod) or mod
        if isinstance(step, PairWarpgroupCtas):
            return mod
    raise AssertionError("pairing pass is not in the PTX pipeline")


def test_operand_stages_are_shared_and_handed_off_once_per_task():
    mod = paired(module())
    [kernel] = mod.body.ops
    stages = shared_operand_stages(kernel)
    assert len(stages) == 8 and all(alloc.attrs["warpgroup_shared"] for alloc in stages)
    [loop] = [op for op in kernel.body.ops if isinstance(op, ForOp)]
    releases = [
        op
        for op in loop.body.ops
        if isinstance(op, PTXTemplateOp)
        and op.attrs.get("description", "").startswith("release")
    ]
    events = {op.operands[0] for op in releases}
    assert len(releases) == len(events) == 4
    acquires = [
        op for op in loop.body.ops if isinstance(op, AwaitOp) and op.lock in events
    ]
    assert len(acquires) == 4
    # Each stage is acquired before its first write and released after its
    # final MMA retires, so the copies alternate reductions.
    assert max(map(loop.body.ops.index, acquires)) < min(
        map(loop.body.ops.index, releases)
    )
    ptx = make_ptx_kernel(module()).ptx
    assert ".reqntid 256, 1, 1" in ptx and ptx.count("bar.sync 0;") == 1
    assert "%tid.x" not in ptx.split("mov.u32 %vnctaid")[1]


@pytest.mark.parametrize("damage", ["partial_wave", "block_barrier"])
def test_pairing_rejects_unbalanced_or_coupled_programs(damage):
    with pytest.raises(ValueError):
        mod = module(grid=6 if damage == "partial_wave" else 8)
        for step in PtxLoweringPipeline().passes:
            if isinstance(step, PairWarpgroupCtas) and damage == "block_barrier":
                [kernel] = mod.body.ops
                [loop] = [op for op in kernel.body.ops if isinstance(op, ForOp)]
                loop.body.ops.insert(0, BarrierOp.build("block"))
            mod = step.run(mod) or mod


def test_cyclic_tail_keeps_one_mma_group_queued():
    mod = module(ctas=1, grid=4)
    [kernel] = mod.body.ops
    [loop] = [op for op in kernel.body.ops if isinstance(op, ForOp)]
    waits = [
        op
        for op in loop.body.ops
        if isinstance(op, AwaitOp) and op.lock.type.kind == "wgmma"
    ]
    # Only the drain before accumulator conversion, and possibly the first
    # issue before the queued loop, wait for every group.
    assert not waits[-1].attrs.get("pending")
    assert sum(not op.attrs.get("pending") for op in waits) <= 2 < len(waits)


def test_rotating_ring_uses_physical_storage_and_completion_events():
    mod = paired(module(m=1536, n=168, k=4096, grid=4, stages=5))
    [kernel] = mod.body.ops
    assert kernel.attrs["paired_warpgroup_ctas"] == dict(
        stages=5, k_steps=32, rotating=True
    )
    storage = [
        op
        for op in kernel.body.ops
        if isinstance(op, AllocOp) and op.attrs.get("warpgroup_shared")
    ]
    events = [
        op
        for op in kernel.body.ops
        if isinstance(op, AllocOp) and op.attrs.get("array_count")
    ]
    assert len(storage) == len(events) == 1
    assert (
        storage[0].result().type.static_shape[0] == events[0].attrs["array_count"] == 5
    )
    [loop] = [op for op in kernel.body.ops if isinstance(op, ForOp)]
    releases = [
        op
        for op in loop.body.ops
        if isinstance(op, PTXTemplateOp)
        and op.attrs.get("description", "").startswith("release")
    ]
    assert len(releases) == 5
    handles = {op.operands[0] for op in releases}
    assert all(handle.type == LockType.build("mbar") for handle in handles)
    assert (
        sum(isinstance(op, AwaitOp) and op.lock in handles for op in loop.body.ops) == 5
    )
    before = str(mod)
    PairWarpgroupCtas().run(mod)
    assert str(mod) == before


@pytest.mark.parametrize("ctas", [True, False, 0, 3, 2.0])
def test_copy_count_is_a_supported_integer(ctas):
    with pytest.raises(ValueError, match="warpgroup_ctas"):
        module(ctas=ctas)


@pytest.mark.parametrize(
    "damage", ["flat_block", "cluster", "raw_barrier", "future_prefetch"]
)
def test_invalid_pairing_fails_before_mutating_ir(damage):
    mod = module()
    for step in PtxLoweringPipeline().passes:
        if isinstance(step, PairWarpgroupCtas):
            break
        mod = step.run(mod) or mod
    [kernel] = mod.body.ops
    if damage == "flat_block":
        kernel.attrs["block_dim"] = (64, 2, 1)
    elif damage == "cluster":
        kernel.attrs["cluster_dim"] = (2, 1, 1)
    elif damage == "raw_barrier":
        kernel.body.ops.append(PTXTemplateOp.build([], [], "bar.sync 0;", [], []))
    else:
        loop = next(op for op in kernel.body.ops if isinstance(op, ForOp))
        loop.attrs["output_head_prefetch"] = True
    before = str(mod)
    with pytest.raises(ValueError):
        PairWarpgroupCtas().run(mod)
    assert str(mod) == before


def test_unbound_ptx_value_placeholder_cannot_become_scratch_register():
    index = IndexConstantOp.build(0)
    malformed = PTXTemplateOp.build(
        [index.result()],
        [IndexType()],
        "mov.u32 ${b32:result.dst.val}, ${b32:operand.slot};",
        ["slot"],
        ["dst"],
    )
    mod = ModuleOp.build(
        Block(
            ops=[
                KernelOp.build(
                    "test",
                    Block(ops=[index, malformed]),
                    block_dim=(128, 1, 1),
                    grid_dim=(1, 1, 1),
                )
            ]
        )
    )
    with pytest.raises(ValueError, match="unbound PTX value placeholder: operand.slot"):
        make_ptx_kernel(mod)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires Hopper")
@pytest.mark.parametrize(
    "dtype,output,m,n,cyclic,k,stages",
    [
        ("e4m3", "bf16", 1024, 336, True, 1024, 4),
        ("e4m3", "bf16", 1024, 336, False, 1024, 4),
        ("e5m2", "f16", 2048, 168, True, 1024, 4),
        ("bf16", "bf16", 1024, 336, True, 1024, 4),
        ("e4m3", "bf16", 1536, 168, True, 4096, 5),
        ("e4m3", "bf16", 1472, 152, True, 4096, 5),
        ("e5m2", "f16", 768, 168, True, 2048, 5),
        ("bf16", "bf16", 768, 168, True, 1024, 5),
        ("f16", "f16", 768, 168, True, 1024, 5),
    ],
)
def test_pairs_replay_changed_inputs_and_scales(dtype, output, m, n, cyclic, k, stages):
    types = dict(
        e4m3=torch.float8_e4m3fn,
        e5m2=torch.float8_e5m2,
        bf16=torch.bfloat16,
        f16=torch.float16,
    )
    tasks = ((m + 127) // 128) * ((n + 167) // 168)
    run = make_ptx_launcher(
        make_ptx_kernel(
            module(
                m,
                n,
                k,
                dtype,
                output,
                grid=4 if m >= 1400 else max(2, tasks // 4 * 2),
                cyclic=cyclic,
                stages=stages,
            )
        )
    )
    gen = torch.Generator(device="cuda").manual_seed(20261120)
    a = torch.zeros(m, k, device="cuda", dtype=types[dtype])
    b = torch.zeros(n, k, device="cuda", dtype=types[dtype])
    actual = torch.empty(m, n, device="cuda", dtype=types[output])
    sa, sb = (torch.tensor(v, device="cuda") for v in (0.25, 0.125))
    suffix = (sa, sb) if dtype in ("e4m3", "e5m2") else ()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(3):
            run(a, b, actual, *suffix)
    for scale in (0.25, 0.5):
        a.copy_(torch.randint(-4, 5, a.shape, generator=gen, device="cuda").float() / 4)
        b.copy_(torch.randint(-4, 5, b.shape, generator=gen, device="cuda").float() / 4)
        sa.fill_(scale)
        actual.fill_(float("nan"))
        graph.replay()
        expected = a.double() @ b.double().T
        if suffix:
            expected *= scale * 0.125
        torch.testing.assert_close(actual, expected.to(actual.dtype), rtol=0, atol=0)
