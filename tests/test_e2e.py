"""End-to-end tests: parse serialized IR → codegen → load + launch on CUDA."""
import json
import textwrap

import pytest
import torch

from ayla.passes.canonicalize import canonicalize
from ayla.gpu_dialect import gpu_dialect
from ayla.passes.licm import licm
from ayla.passes.pipeline_loop import pipeline_loop
from ayla.passes.promote_buffers import promote_buffers
from ayla.ptx_codegen import make_ptx_kernel
from ayla.ptx_runtime import make_ptx_launcher
from ayla.passes.tile_matmul import tile_matmul
from ayla.passes.unroll_loop import unroll_loop

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="no CUDA"
)


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
          kernel @cwise_add(%lhs: gmem_tensor<1024xf32>, %rhs: gmem_tensor<1024xf32>, %dst: gmem_tensor<1024xf32>) grid_dim=[4, 1, 1] block_dim=[256, 1, 1] {{
            %gx: index, %gy: index, %gz: index = block_ids()
            %bx: index, %by: index, %bz: index = thread_ids()
            ptx_template(%lhs: gmem_tensor<1024xf32>, %rhs: gmem_tensor<1024xf32>, %dst: gmem_tensor<1024xf32>, %gx: index, %gy: index, %gz: index, %bx: index, %by: index, %bz: index) attrs={{template: {template_str}}}
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
          kernel @matmul_16(%a: gmem_tensor<16x16xf32>, %b: gmem_tensor<16x16xf32>, %c: gmem_tensor<16x16xf32>) grid_dim=[1, 1, 1] block_dim=[16, 16, 1] {{
            %gx: index, %gy: index, %gz: index = block_ids()
            %bx: index, %by: index, %bz: index = thread_ids()
            ptx_template(%a: gmem_tensor<16x16xf32>, %b: gmem_tensor<16x16xf32>, %c: gmem_tensor<16x16xf32>, %gx: index, %gy: index, %gz: index, %bx: index, %by: index, %bz: index) attrs={{template: {template_str}}}
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
          kernel @naive_matmul_4096(%a: gmem_tensor<4096x4096xf32>, %b: gmem_tensor<4096x4096xf32>, %c: gmem_tensor<4096x4096xf32>) grid_dim=[64, 64, 1] block_dim=[32, 32, 1] {
            %cx: index, %cy: index, %cz: index = block_ids()
            %tx: index, %ty: index, %tz: index = thread_ids()
            %a_block: gmem_tensor<64x4096xf32> = tile(%a: gmem_tensor<4096x4096xf32>, %cx: index, 0)
            %b_block: gmem_tensor<64x4096xf32> = tile(%b: gmem_tensor<4096x4096xf32>, %cy: index, 0)
            %c_block: gmem_tensor<64x64xf32 strides=[4096, 1]> = tile(%c: gmem_tensor<4096x4096xf32>, %cx: index, %cy: index)
            %a_thread: gmem_tensor<2x4096xf32> = tile(%a_block: gmem_tensor<64x4096xf32>, %tx: index, 0)
            %b_thread: gmem_tensor<2x4096xf32> = tile(%b_block: gmem_tensor<64x4096xf32>, %ty: index, 0)
            %c_thread: gmem_tensor<2x2xf32 strides=[4096, 1]> = tile(%c_block: gmem_tensor<64x64xf32 strides=[4096, 1]>, %tx: index, %ty: index)
            %c_thread_rmem: rmem_tensor<2x2xf32> = alloc() attrs={init: 0.0}
            for (%k: index = 0 to 4096) {
              %a_k: gmem_tensor<2x1xf32 strides=[4096, 1]> = tile(%a_thread: gmem_tensor<2x4096xf32>, 0, %k: index)
              %b_k: gmem_tensor<2x1xf32 strides=[4096, 1]> = tile(%b_thread: gmem_tensor<2x4096xf32>, 0, %k: index)
              %a_k_rmem: rmem_tensor<2x1xf32> = alloc()
              move(%a_k_rmem: rmem_tensor<2x1xf32>, %a_k: gmem_tensor<2x1xf32 strides=[4096, 1]>) attrs={nc: true}
              %b_k_rmem: rmem_tensor<2x1xf32> = alloc()
              move(%b_k_rmem: rmem_tensor<2x1xf32>, %b_k: gmem_tensor<2x1xf32 strides=[4096, 1]>) attrs={nc: true}
              matmul(%a_k_rmem: rmem_tensor<2x1xf32>, %b_k_rmem: rmem_tensor<2x1xf32>, %c_thread_rmem: rmem_tensor<2x2xf32>)
            }
            move(%c_thread: gmem_tensor<2x2xf32 strides=[4096, 1]>, %c_thread_rmem: rmem_tensor<2x2xf32>)
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
          kernel @tma(%a: gmem_tensor<4096x4096xf32>) grid_dim=[32, 64, 1] block_dim=[128, 1, 1] {
            %cx: index, %cy: index, %cz: index = block_ids()
            %tx: index, %ty: index, %tz: index = thread_ids()
            %a_block: gmem_tensor<128x64xf32 strides=[4096, 1]> = tile(%a: gmem_tensor<4096x4096xf32>, %cx: index, %cy: index)
            %a_thread: gmem_tensor<64x32xf32 strides=[4096, 1]> = tile(%a_block: gmem_tensor<128x64xf32 strides=[4096, 1]>, %tx: index, %ty: index)
            %a_smem: smem_tensor<64x32xf32 swizzle="128B"> = alloc()
            move(%a_smem: smem_tensor<64x32xf32 swizzle="128B">, %a_thread: gmem_tensor<64x32xf32 strides=[4096, 1]>) attrs={tma: True}
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
          kernel @tma_load(%a: gmem_tensor<64x64xf32>, %a_desc: tma_desc<arg_idx=0 gmem_tensor<64x64xf32> tile=64x64>) grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {
            %gx: index, %gy: index, %gz: index = block_ids()
            %tx: index, %ty: index, %tz: index = thread_ids()
            %a_smem: smem_tensor<64x64xf32> = alloc()
            %lock: lock<mbar> = alloc()
            async(%lock: lock<mbar>) tma_move(%a_smem: smem_tensor<64x64xf32>, %a_desc: tma_desc<arg_idx=0 gmem_tensor<64x64xf32> tile=64x64>, %gx: index, %gy: index)
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
def test_experimental_auto_matmul_4096():
    """Experimental: build a wgmma matmul from high-level tiling passes."""
    module_str = textwrap.dedent("""\
        module {
          kernel @auto_matmul_4096(%a: gmem_tensor<4096x4096xbf16>, %b: gmem_tensor<4096x4096xbf16>, %c: gmem_tensor<4096x4096xbf16>) {
            matmul(%a: gmem_tensor<4096x4096xbf16>, %b: gmem_tensor<4096x4096xbf16>, %c: gmem_tensor<4096x4096xbf16>) attrs={init: 0.0, label: "mm"}
          }
        }""")
    module = gpu_dialect.parse(module_str)
    module = tile_matmul(
        module,
        tile_dims=["M", "N"], tile_sizes=[128, 128],
        tile_levels=["block.x", "block.y"], tile_labels=[None, None],
        label_filter="mm",
    )
    module = promote_buffers(
        module, target_type="wrmem", operands=["dst"],
        swizzle=None, layout="wgmma_layout_0", warps=4,
        tma=False, nc=False, target_dtype="f32", label_filter="mm",
    )
    module = tile_matmul(
        module,
        tile_dims=["K"], tile_sizes=[64],
        tile_levels=["for"], tile_labels=["k"],
        label_filter="mm",
    )
    module = promote_buffers(
        module, target_type="smem", operands=["lhs", "rhs"],
        swizzle="128B", layout=None, warps=None,
        tma=True, nc=True, label_filter="mm",
    )
    module = tile_matmul(
        module,
        tile_dims=["M"], tile_sizes=[1],
        tile_levels=["thread.x"], tile_labels=[None],
        label_filter="mm", collective=True,
    )
    module = tile_matmul(
        module,
        tile_dims=["K"], tile_sizes=[16],
        tile_levels=["for"], tile_labels=["l"],
        label_filter="mm",
    )
    module = unroll_loop(module, unroll_factor=2, label_filter="k")
    module = licm(module)
    module = pipeline_loop(module, label_filter_loop="k", label_filter_op="mm_(lhs|rhs)_smem_iter0", pipelining_offset=1)

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
          kernel @auto_matmul_4096(%a: gmem_tensor<4096x4096xbf16>, %b: gmem_tensor<4096x4096xbf16>, %c: gmem_tensor<4096x4096xbf16>) grid_dim=[32, 16, 1] block_dim=[128, 1, 1] {
            matmul(%a: gmem_tensor<4096x4096xbf16>, %b: gmem_tensor<4096x4096xbf16>, %c: gmem_tensor<4096x4096xbf16>) attrs={init: 0.0, label: "mm"}
          }
        }""")
    module = gpu_dialect.parse(module_str)
    module = tile_matmul(
        module,
        tile_dims=["M", "N"], tile_sizes=[128, 256],
        tile_levels=["block.x", "block.y"], tile_labels=[None, None],
        label_filter="mm",
    )
    module = promote_buffers(
        module, target_type="wrmem", operands=["dst"],
        swizzle=None, layout="wgmma_layout_0", warps=4,
        tma=False, nc=False, target_dtype="f32", label_filter="mm",
    )
    module = tile_matmul(
        module,
        tile_dims=["K"], tile_sizes=[64],
        tile_levels=["for"], tile_labels=["k"],
        label_filter="mm",
    )
    module = promote_buffers(
        module, target_type="smem", operands=["lhs", "rhs"],
        swizzle="128B", layout=None, warps=None,
        tma=True, nc=True, label_filter="mm",
    )
    module = tile_matmul(
        module,
        tile_dims=["K"], tile_sizes=[16],
        tile_levels=["for"], tile_labels=["k"],
        label_filter="mm",
    )

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
