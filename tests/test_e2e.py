"""End-to-end tests: parse serialized IR → codegen → load + launch on CUDA."""
import json
import textwrap

import pytest
import torch

from ayla.base.pipeline import Pipeline
from ayla.gpu_dialect import CollectiveAxis, CollectiveLevel, gpu_dialect
from ayla.passes.licm import LICM
from ayla.passes.pipeline_loop import PipelineLoop
from ayla.passes.promote_buffers import PromoteBuffers
from ayla.passes.tile_matmul import TileMatmul
from ayla.passes.unroll_loop import UnrollLoop
from ayla.ptx_codegen import make_ptx_kernel
from ayla.ptx_runtime import make_ptx_launcher
from tests.testing_utils import requires_cuda


@requires_cuda
@pytest.mark.parametrize("m", [1, 4, 32, 64, 512, 2048])
def test_muse_swiglu_benchmark_route_matches_torch(m: int):
    """Exercise the full benchmark route until torch.compile owns it."""
    from benchmarks.benchmark_bf16_muse_swiglu import (
        _check,
        _inputs,
        _torch_swiglu,
        build_ayla_swiglu,
    )

    inputs = _inputs(m)
    expected = _torch_swiglu(*inputs)
    ayla_swiglu, _ = build_ayla_swiglu(m)
    actual = ayla_swiglu(*inputs)
    torch.cuda.synchronize()
    _check("Ayla benchmark route", actual, expected)


CWISE_ADD_TEMPLATE = textwrap.dedent(
    """\
    mov.u32             ${b32:ntid}, %ntid.x;
    mad.lo.s32          ${b32:i}, ${b32:operand[3].val}, ${b32:ntid}, ${b32:operand[6].val};
    cvta.to.global.u64  ${b64:lhs_g}, ${b64:operand[0].ptr};
    mul.wide.s32        ${b64:off}, ${b32:i}, 4;
    add.s64             ${b64:lhs_a}, ${b64:lhs_g}, ${b64:off};
    cvta.to.global.u64  ${b64:rhs_g}, ${b64:operand[1].ptr};
    add.s64             ${b64:rhs_a}, ${b64:rhs_g}, ${b64:off};
    ld.global.nc.f32    ${f32:b}, [${b64:rhs_a}];
    ld.global.nc.f32    ${f32:a}, [${b64:lhs_a}];
    add.f32             ${f32:s}, ${f32:a}, ${f32:b};
    cvta.to.global.u64  ${b64:dst_g}, ${b64:operand[2].ptr};
    add.s64             ${b64:dst_a}, ${b64:dst_g}, ${b64:off};
    st.global.f32       [${b64:dst_a}], ${f32:s};
    """
)


@requires_cuda
def test_cwise_add_from_ir():
    n = 1024

    template_str = json.dumps(CWISE_ADD_TEMPLATE)
    module_str = textwrap.dedent(f"""\
        module {{
          kernel @cwise_add(%lhs: gmem<1024xf32>, %rhs: gmem<1024xf32>, %dst: gmem<1024xf32>) grid_dim=[4, 1, 1] block_dim=[256, 1, 1] {{
            %gx: index, %gy: index, %gz: index = block_ids()
            %bx: index, %by: index, %bz: index = thread_ids()
            ptx_template(%lhs: gmem<1024xf32>, %rhs: gmem<1024xf32>, %dst: gmem<1024xf32>, %gx: index, %gy: index, %gz: index, %bx: index, %by: index, %bz: index) attrs={{template: {template_str}}}
          }}
        }}""")  # noqa: E501

    module = gpu_dialect.parse(module_str)
    ptx_kernel = make_ptx_kernel(module)

    print()
    print(ptx_kernel.ptx)

    a = torch.arange(n, dtype=torch.float32, device="cuda")
    b = a * 2
    c = torch.empty_like(a)

    ptx_launcher = make_ptx_launcher(ptx_kernel)
    ptx_launcher(a, b, c)

    torch.cuda.synchronize()
    assert torch.allclose(c, a + b)


@requires_cuda
def test_reshape_from_ir():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @reshape_copy(%src: gmem<2x2xf32>, %dst: gmem<1x4xf32>) grid_dim=[1, 1, 1] block_dim=[1, 1, 1] {
            %view: gmem<1x4xf32> = reshape(%src: gmem<2x2xf32>)
            %values: rmem<1x4xf32> = alloc() attrs={id: 0}
            move(%values: rmem<1x4xf32>, %view: gmem<1x4xf32>) attrs={nc: true}
            move(%dst: gmem<1x4xf32>, %values: rmem<1x4xf32>)
          }
        }"""))  # noqa: E501

    kernel = make_ptx_kernel(module)
    src = torch.randn(2, 2, dtype=torch.float32, device="cuda")
    dst = torch.empty(1, 4, dtype=torch.float32, device="cuda")

    make_ptx_launcher(kernel)(src, dst)

    torch.cuda.synchronize()
    torch.testing.assert_close(dst, src.reshape(1, 4))


# 16x16 f32 matmul, row-major. One block of 16x16 threads, one grid
# block. Each thread computes one C[i][j] via a k-loop.
MATMUL_16_TEMPLATE = textwrap.dedent(
    """\
    cvta.to.global.u64  ${b64:a_g}, ${b64:operand[0].ptr};
    cvta.to.global.u64  ${b64:b_g}, ${b64:operand[1].ptr};
    cvta.to.global.u64  ${b64:c_g}, ${b64:operand[2].ptr};

    mul.lo.s32          ${b32:i16}, ${b32:operand[7].val}, 16;
    mul.wide.s32        ${b64:i_byte}, ${b32:i16}, 4;
    add.s64             ${b64:a_row}, ${b64:a_g}, ${b64:i_byte};

    mul.lo.s32          ${b32:j16}, ${b32:operand[6].val}, 16;
    mul.wide.s32        ${b64:j_byte}, ${b32:j16}, 4;
    add.s64             ${b64:b_row}, ${b64:b_g}, ${b64:j_byte};

    mov.f32             ${f32:acc}, 0f00000000;
    mov.u32             ${b32:k}, 0;

    ${label:loop}:
    mul.wide.s32        ${b64:k_a}, ${b32:k}, 4;
    add.s64             ${b64:a_addr}, ${b64:a_row}, ${b64:k_a};
    ld.global.nc.f32    ${f32:a}, [${b64:a_addr}];
    mul.wide.s32        ${b64:k_b}, ${b32:k}, 4;
    add.s64             ${b64:b_addr}, ${b64:b_row}, ${b64:k_b};
    ld.global.nc.f32    ${f32:b}, [${b64:b_addr}];
    fma.rn.f32          ${f32:acc}, ${f32:a}, ${f32:b}, ${f32:acc};
    add.s32             ${b32:k}, ${b32:k}, 1;
    setp.lt.s32         ${pred:p}, ${b32:k}, 16;
    @${pred:p} bra      ${label:loop};

    add.s32             ${b32:c_idx}, ${b32:i16}, ${b32:operand[6].val};
    mul.wide.s32        ${b64:c_byte}, ${b32:c_idx}, 4;
    add.s64             ${b64:c_addr}, ${b64:c_g}, ${b64:c_byte};
    st.global.f32       [${b64:c_addr}], ${f32:acc};
    """
)


@requires_cuda
def test_matmul_16_from_ir():
    template_str = json.dumps(MATMUL_16_TEMPLATE)
    module_str = textwrap.dedent(f"""\
        module {{
          kernel @matmul_16(%a: gmem<16x16xf32>, %b: gmem<16x16xf32>, %c: gmem<16x16xf32>) grid_dim=[1, 1, 1] block_dim=[16, 16, 1] {{
            %gx: index, %gy: index, %gz: index = block_ids()
            %bx: index, %by: index, %bz: index = thread_ids()
            ptx_template(%a: gmem<16x16xf32>, %b: gmem<16x16xf32>, %c: gmem<16x16xf32>, %gx: index, %gy: index, %gz: index, %bx: index, %by: index, %bz: index) attrs={{template: {template_str}}}
          }}
        }}""")  # noqa: E501

    module = gpu_dialect.parse(module_str)
    ptx_kernel = make_ptx_kernel(module)

    print()
    print(ptx_kernel.ptx)

    torch.manual_seed(0)
    a = torch.randn(16, 16, dtype=torch.float32, device="cuda")
    b = torch.randn(16, 16, dtype=torch.float32, device="cuda")
    c = torch.empty(16, 16, dtype=torch.float32, device="cuda")

    ptx_launcher = make_ptx_launcher(ptx_kernel)
    ptx_launcher(a, b, c)

    torch.cuda.synchronize()
    assert torch.allclose(c, a @ b.T, atol=1e-4)


@requires_cuda
def test_naive_matmul_4096_from_ir():
    module_str = textwrap.dedent("""\
        module {
          kernel @naive_matmul_4096(%a: gmem<4096x4096xf32>, %b: gmem<4096x4096xf32>, %c: gmem<4096x4096xf32>) grid_dim=[64, 64, 1] block_dim=[32, 32, 1] {
            %cx: index, %cy: index, %cz: index = block_ids()
            %tx: index, %ty: index, %tz: index = thread_ids()
            %a_block: gmem<64x4096xf32> = tile(%a: gmem<4096x4096xf32>, %cx: index, 0)
            %b_block: gmem<64x4096xf32> = tile(%b: gmem<4096x4096xf32>, %cy: index, 0)
            %c_block: gmem<64x64xf32 strides=[4096, 1]> = tile(%c: gmem<4096x4096xf32>, %cx: index, %cy: index)
            %a_thread: gmem<2x4096xf32> = tile(%a_block: gmem<64x4096xf32>, %tx: index, 0)
            %b_thread: gmem<2x4096xf32> = tile(%b_block: gmem<64x4096xf32>, %ty: index, 0)
            %c_thread: gmem<2x2xf32 strides=[4096, 1]> = tile(%c_block: gmem<64x64xf32 strides=[4096, 1]>, %tx: index, %ty: index)
            %c_thread_rmem: rmem<2x2xf32> = alloc() attrs={id: 0, init: 0.0}
            for (%k: index = 0 to 4096) {
              %a_k: gmem<2x1xf32 strides=[4096, 1]> = tile(%a_thread: gmem<2x4096xf32>, 0, %k: index)
              %b_k: gmem<2x1xf32 strides=[4096, 1]> = tile(%b_thread: gmem<2x4096xf32>, 0, %k: index)
              %a_k_rmem: rmem<2x1xf32> = alloc() attrs={id: 1}
              move(%a_k_rmem: rmem<2x1xf32>, %a_k: gmem<2x1xf32 strides=[4096, 1]>) attrs={nc: true}
              %b_k_rmem: rmem<2x1xf32> = alloc() attrs={id: 2}
              move(%b_k_rmem: rmem<2x1xf32>, %b_k: gmem<2x1xf32 strides=[4096, 1]>) attrs={nc: true}
              matmul(%a_k_rmem: rmem<2x1xf32>, %b_k_rmem: rmem<2x1xf32>, %c_thread_rmem: rmem<2x2xf32>)
            }
            move(%c_thread: gmem<2x2xf32 strides=[4096, 1]>, %c_thread_rmem: rmem<2x2xf32>)
          }
        }""")  # noqa: E501

    module = gpu_dialect.parse(module_str)
    ptx_kernel = make_ptx_kernel(module)

    torch.manual_seed(0)
    a = torch.randn(4096, 4096, dtype=torch.float32, device="cuda")
    b = torch.randn(4096, 4096, dtype=torch.float32, device="cuda")
    c = torch.empty(4096, 4096, dtype=torch.float32, device="cuda")

    ptx_launcher = make_ptx_launcher(ptx_kernel)
    ptx_launcher(a, b, c)

    torch.cuda.synchronize()
    assert torch.allclose(c, a @ b.T, atol=1e-2)


@requires_cuda
def test_tma():
    """High-level IR: tiles + load(gmem → smem). Nice to write."""
    module_str = textwrap.dedent("""\
        module {
          kernel @tma(%a: gmem<4096x4096xf32>) grid_dim=[32, 64, 1] block_dim=[128, 1, 1] {
            %cx: index, %cy: index, %cz: index = block_ids()
            %tx: index, %ty: index, %tz: index = thread_ids()
            %a_block: gmem<128x64xf32 strides=[4096, 1]> = tile(%a: gmem<4096x4096xf32>, %cx: index, %cy: index)
            %a_thread: gmem<64x32xf32 strides=[4096, 1]> = tile(%a_block: gmem<128x64xf32 strides=[4096, 1]>, %tx: index, %ty: index)
            %a_smem: smem<64x32xf32 swizzle="128B"> = alloc() attrs={id: 0}
            move(%a_smem: smem<64x32xf32 swizzle="128B">, %a_thread: gmem<64x32xf32 strides=[4096, 1]>) attrs={tma: True}
          }
        }""")  # noqa: E501

    module = gpu_dialect.parse(module_str)
    assert repr(module) == module_str

    ptx_kernel = make_ptx_kernel(module)

    print()
    print(ptx_kernel.ptx)

    a = torch.randn(4096, 4096, dtype=torch.float32, device="cuda")

    ptx_launcher = make_ptx_launcher(ptx_kernel)
    ptx_launcher(a)

    torch.cuda.synchronize()


@requires_cuda
def test_tma_lowered():
    """Lowered IR: one TMA load, nothing else. Minimal e2e for TMA codegen + runtime."""
    module_str = textwrap.dedent("""\
        module {
          kernel @tma_load(%a: gmem<64x64xf32>, %a_desc: tma_desc<arg_idx=0 gmem<64x64xf32> tile=64x64>) grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {
            %gx: index, %gy: index, %gz: index = block_ids()
            %tx: index, %ty: index, %tz: index = thread_ids()
            %a_smem: smem<64x64xf32> = alloc() attrs={id: 0}
            %lock: lock<mbar> = alloc() attrs={id: 1}
            async(%lock: lock<mbar>) tma_move(%a_smem: smem<64x64xf32>, %a_desc: tma_desc<arg_idx=0 gmem<64x64xf32> tile=64x64>, %gx: index, %gy: index)
            await(%lock: lock<mbar>)
          }
        }""")  # noqa: E501

    module = gpu_dialect.parse(module_str)

    ptx_kernel = make_ptx_kernel(module)

    print()
    print(ptx_kernel.ptx)

    a = torch.randn(64, 64, dtype=torch.float32, device="cuda")

    ptx_launcher = make_ptx_launcher(ptx_kernel)
    ptx_launcher(a)

    torch.cuda.synchronize()


@requires_cuda
def test_experimental_auto_matmul_4096_with_excess_block():
    """Build a WGMMA matmul whose fixed grid has one inactive block."""
    module_str = textwrap.dedent("""\
        module {
          kernel @auto_matmul_4096(%a: gmem<4096x4096xbf16>, %b: gmem<4096x4096xbf16>, %c: gmem<4096x4096xbf16>) grid_dim=[33, 32, 1] block_dim=[128, 1, 1] {
            matmul(%a: gmem<4096x4096xbf16>, %b: gmem<4096x4096xbf16>, %c: gmem<4096x4096xbf16>) collective=xyz.grid attrs={init: 0.0, label: "mm"}
          }
        }""")
    module = gpu_dialect.parse(module_str)
    pipeline = Pipeline()
    pipeline.add_pass(TileMatmul(
        op_dims=["M", "N"], tile_sizes=[128, 128],
        iteration_bindings=[
            (CollectiveLevel.BLOCK, CollectiveAxis.X),
            (CollectiveLevel.BLOCK, CollectiveAxis.Y),
        ],
        for_loop_labels=[None, None],
        op_label_filter="mm",
    ))
    pipeline.add_pass(PromoteBuffers(
        target_type="wgrmem", operands=["dst"],
        swizzle=None, layout="wgmma_layout_0", warps=4,
        tma=False, nc=False, target_dtype="f32", label_filter="mm",
    ))
    pipeline.add_pass(TileMatmul(
        op_dims=["K"], tile_sizes=[64],
        iteration_bindings=[None], for_loop_labels=["k"],
        op_label_filter="mm",
    ))
    pipeline.add_pass(PromoteBuffers(
        target_type="smem", operands=["lhs", "rhs"],
        swizzle="128B", layout=None, warps=None,
        tma=True, label_filter="mm",
    ))
    pipeline.add_pass(TileMatmul(
        op_dims=["K"], tile_sizes=[16],
        iteration_bindings=[None], for_loop_labels=["l"],
        op_label_filter="mm",
    ))
    pipeline.add_pass(UnrollLoop(unroll_factor=2, label_filter="k"))
    pipeline.add_pass(LICM())
    pipeline.add_pass(PipelineLoop(
        label_filter_loop="k",
        label_filter_op="mm_(lhs|rhs)_smem_iter0",
        pipelining_offset=1,
    ))
    module = pipeline.run(module)

    print(repr(module))

    ptx_kernel = make_ptx_kernel(module)
    ptx_launcher = make_ptx_launcher(ptx_kernel)

    a = torch.randn(4096, 4096, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(4096, 4096, dtype=torch.bfloat16, device="cuda")
    c = torch.randn(4096, 4096, dtype=torch.bfloat16, device="cuda")
    ptx_launcher(a, b, c)
    torch.cuda.synchronize()

    expected = a @ b.T
    assert torch.allclose(c.float(), expected.float(), atol=2.0), (
        f"max error: {(c.float() - expected.float()).abs().max().item()}"
    )


@requires_cuda
def test_dbg_module():
    module_str = textwrap.dedent("""\
        module {
          kernel @auto_matmul_4096(%a: gmem<4096x4096xbf16>, %b: gmem<4096x4096xbf16>, %c: gmem<4096x4096xbf16>) grid_dim=[32, 16, 1] block_dim=[128, 1, 1] {
            matmul(%a: gmem<4096x4096xbf16>, %b: gmem<4096x4096xbf16>, %c: gmem<4096x4096xbf16>) collective=xyz.grid attrs={init: 0.0, label: "mm"}
          }
        }""")
    module = gpu_dialect.parse(module_str)
    pipeline = Pipeline()
    pipeline.add_pass(TileMatmul(
        op_dims=["M", "N"], tile_sizes=[128, 256],
        iteration_bindings=[
            (CollectiveLevel.BLOCK, CollectiveAxis.X),
            (CollectiveLevel.BLOCK, CollectiveAxis.Y),
        ],
        for_loop_labels=[None, None],
        op_label_filter="mm",
    ))
    pipeline.add_pass(PromoteBuffers(
        target_type="wgrmem", operands=["dst"],
        swizzle=None, layout="wgmma_layout_0", warps=4,
        tma=False, nc=False, target_dtype="f32", label_filter="mm",
    ))
    pipeline.add_pass(TileMatmul(
        op_dims=["K"], tile_sizes=[64],
        iteration_bindings=[None], for_loop_labels=["k"],
        op_label_filter="mm",
    ))
    pipeline.add_pass(PromoteBuffers(
        target_type="smem", operands=["lhs", "rhs"],
        swizzle="128B", layout=None, warps=None,
        tma=True, label_filter="mm",
    ))
    pipeline.add_pass(TileMatmul(
        op_dims=["K"], tile_sizes=[16],
        iteration_bindings=[None], for_loop_labels=["k"],
        op_label_filter="mm",
    ))
    module = pipeline.run(module)

    print(module)

    ptx_kernel = make_ptx_kernel(module)
    ptx_launcher = make_ptx_launcher(ptx_kernel)

    a = torch.randn(4096, 4096, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(4096, 4096, dtype=torch.bfloat16, device="cuda")
    c = torch.randn(4096, 4096, dtype=torch.bfloat16, device="cuda")
    ptx_launcher(a, b, c)
    torch.cuda.synchronize()

    expected = a @ b.T
    assert torch.allclose(c.float(), expected.float(), atol=2.0), (
        f"max error: {(c.float() - expected.float()).abs().max().item()}"
    )
