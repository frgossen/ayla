"""Predicated direct Split-K stores preserve tails and private graph scratch."""

from __future__ import annotations

import pytest
import torch

from ayla.gemm_jit import _seed, build_gemm_module, select_gemm_schedule
from ayla.ptx_codegen import make_ptx_kernel
from ayla.ptx_runtime import make_ptx_launcher


def test_paired_row_tail_store_keeps_reshaped_parent_bounds():
    from ayla.base.ir import find_ops_of_type
    from ayla.gpu_dialect import IfOp, MoveOp, gpu_dialect
    from ayla.passes.lower_to_scalar import LowerToScalar

    module = gpu_dialect.parse_module("""module {
      kernel @partial(%memory: gmem<1x1x17x6xf32>) grid_dim=[1,1,1] block_dim=[128,1,1] {
        %outer: gmem<1x1x17x8xf32 strides=[102,102,6,1]> = tile(%memory: gmem<1x1x17x6xf32>, 0, 0, 0, 0) attrs={maybe_partial: true}
        %flat: gmem<17x8xf32 strides=[6,1]> = reshape(%outer: gmem<1x1x17x8xf32 strides=[102,102,6,1]>)
        %view: gmem<64x8xf32 strides=[6,1]> = tile(%flat: gmem<17x8xf32 strides=[6,1]>, 0, 0) attrs={maybe_partial: true}
        %fragment: wgrmem<64x8xf32 warps=4 layout="wgmma_layout_0"> = alloc() attrs={id: 0}
        move(%view: gmem<64x8xf32 strides=[6,1]>, %fragment: wgrmem<64x8xf32 warps=4 layout="wgmma_layout_0">) collective=[x.wg,yz.grid]
      }
    }""")
    lowered = LowerToScalar().run(module)
    assert find_ops_of_type(lowered, IfOp)
    assert all(
        op.src.type.num_elements == 1 for op in find_ops_of_type(lowered, MoveOp)
    )


def build(operation, dtype, m, n, k, parts, *, output_m_tile=128, reduction_threads=32):
    cfg = dict(
        select_gemm_schedule("mm", "float16", 4096, 4096, 4096).captured_config,
        output_cta_count=None,
        transpose_output=False,
        output_tiles_per_cta=1,
        output_n_tiles_per_cta=1,
        output_tma_inner_size=None,
        output_slices=1,
        output_store_buffers=1,
        cyclic_operand_pipeline=False,
        output_m_tile=output_m_tile,
        output_n_tile=128,
        warpgroup_m_tile=64,
        warpgroup_n_tile=64,
        output_store="global",
        operand_k_tile=64,
        operand_stages=3,
        split_k_parts=parts,
        split_k_reduce_threads=reduction_threads,
        split_k_reduce_unroll=4,
        reschedule_async=True,
    )
    batch = 2 if operation == "bmm" else 1
    seed = _seed(cfg, operation, m, n, k, (batch, m, n, k), True)
    assert seed is not None
    return build_gemm_module(operation, dtype, batch, m, n, k, seed)


@pytest.mark.parametrize("threads", [32, 64, 128, 256])
def test_direct_finalizer_distributes_outputs_over_selected_threads(threads):
    module = build("mm", "float16", 31, 128, 4104, 4, reduction_threads=threads)
    kernel = make_ptx_kernel(module.reduction)
    assert kernel.block_dim == (threads, 1, 1)
    assert kernel.grid_dim[0] == (31 * 128 + threads - 1) // threads


def test_direct_finalizer_rejects_incomplete_warp_launch():
    with pytest.raises(ValueError, match="split_k_reduce_threads"):
        build("mm", "float16", 31, 128, 4104, 4, reduction_threads=31)


@pytest.mark.parametrize(
    "m,n,paired", [(31, 128, True), (96, 128, True), (192, 128, True), (96, 129, False)]
)
def test_direct_fragment_stores_mask_logical_output(m, n, paired):
    module = build("mm", "float16", m, n, 4104, 4)
    ptx = make_ptx_kernel(module.partials).ptx
    assert "cp.async.bulk.tensor.3d.global.shared" not in ptx
    assert ("st.global.v2.b32" in ptx) == paired
    assert "setp.lt.u32" in ptx


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("operation", ["mm", "addmm", "bmm"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "m,n,k,parts",
    [(31, 128, 4104, 4), (96, 128, 4104, 4), (192, 128, 4104, 4), (192, 129, 264, 16)],
)
def test_direct_split_k_changed_graphs_and_output_tails(
    operation, dtype, m, n, k, parts
):
    module = build(operation, str(dtype).removeprefix("torch."), m, n, k, parts)
    kernels = [make_ptx_kernel(phase) for phase in (module.partials, module.reduction)]
    launches = [make_ptx_launcher(kernel) for kernel in kernels]
    lead = (2,) if operation == "bmm" else ()
    states = []
    for _ in range(2):
        a = torch.randn(*lead, m, k, dtype=dtype, device="cuda")
        bt = torch.randn(*lead, n, k, dtype=dtype, device="cuda")
        bias = (
            torch.randn(n, dtype=dtype, device="cuda") if operation == "addmm" else None
        )
        out = torch.empty(*lead, m, n, dtype=dtype, device="cuda")
        scratch = torch.empty(
            module.workspace_shape, dtype=torch.float32, device="cuda"
        )
        args = ([bias] if bias is not None else []) + [a, bt, out, scratch]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for launch in launches:
                launch(*args)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for launch in launches:
                launch(*args)
        states.append((a, bt, bias, out, scratch, stream, graph))
    for iteration in range(4):
        for a, bt, bias, out, scratch, stream, graph in states:
            with torch.cuda.stream(stream):
                a.mul_(0.75 if iteration % 2 else 1.25)
                bt.mul_(0.875 if iteration % 2 else 1.125)
                if bias is not None:
                    bias.mul_(1.125)
                out.fill_(float("nan"))
                scratch.fill_(float("nan"))
                graph.replay()
        for a, bt, bias, out, scratch, stream, graph in states:
            stream.synchronize()
            gold = a.float() @ bt.float().transpose(-2, -1)
            if bias is not None:
                gold += bias.float()
            rms = (out.float() - gold).norm() / gold.norm().clamp_min(1e-20)
            assert torch.isfinite(out).all()
            assert rms < (0.01 if dtype == torch.bfloat16 else 0.002)
