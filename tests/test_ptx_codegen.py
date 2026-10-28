"""Smoke test for PTX codegen: build a tiny module and check the PTX."""
# flake8: noqa: E501
import textwrap

import pytest

from ayla import (
    Block,
    GmemTensorType,
    KernelOp,
    ModuleOp,
    PTXTemplateOp,
    Value,
    make_ptx_kernel,
)
from ayla.gpu_dialect import (
    ClusterBlockIdsOp,
    ConstantOp,
    F32Type,
    IfOp,
    IndexConstantOp,
    IndexLtOp,
    RmemTensorType,
    ThreadIdsOp,
    gpu_dialect,
)


def test_codegen_rank_zero_constant():
    constant = ConstantOp.build(
        1e-5, RmemTensorType.build([], F32Type()),
    )
    consume = PTXTemplateOp.build(
        operands=[constant.result()],
        result_types=[],
        template="mov.f32 ${f32:sink}, ${f32:operand[0].r0};\n",
    )
    module = ModuleOp.build(Block(ops=[KernelOp.build(
        "constant", Block(ops=[constant, consume]),
        grid_dim=(1, 1, 1), block_dim=(1, 1, 1),
    )]))

    ptx = make_ptx_kernel(module).ptx

    assert "mov.f32 %f0, 0f3727C5AC;" in ptx
    assert "mov.f32 %f1, %f0;" in ptx


def test_codegen_cwise_add():
    n = 1 << 20
    buf_type = GmemTensorType.build((n,), F32Type())
    lhs = Value(name="lhs", type=buf_type)
    rhs = Value(name="rhs", type=buf_type)
    dst = Value(name="dst", type=buf_type)
    template = "add.f32 ${f32:s}, ${f32:a}, ${f32:b};\n"
    ptx_op = PTXTemplateOp.build([lhs, rhs, dst], [], template=template)
    body_block = Block(
        arguments=[lhs, rhs, dst],
        ops=[ptx_op],
    )
    kernel_op = KernelOp.build(
        "cwise_add", body_block,
        grid_dim=(4, 1, 1), block_dim=(256, 1, 1),
        ptx_optimization_level=1,
    )
    module = ModuleOp.build(Block(ops=[kernel_op]))

    kernel = make_ptx_kernel(module)
    assert kernel.name == "cwise_add"
    assert kernel.grid_dim == (4, 1, 1)
    assert kernel.block_dim == (256, 1, 1)
    assert kernel.ptx_optimization_level == 1
    assert kernel.ptx == textwrap.dedent("""\
        .version 9.0
        .target sm_90a
        .address_size 64

        .visible .entry cwise_add(
          .param .b64 lhs_ptr,
          .param .b64 rhs_ptr,
          .param .b64 dst_ptr
        )
        .reqntid 256, 1, 1
        {
          .reg .b64  %rd<3>;
          .reg .f32  %f<3>;

          ld.param.b64 %rd0, [lhs_ptr];
          cvta.to.global.u64 %rd0, %rd0;
          ld.param.b64 %rd1, [rhs_ptr];
          cvta.to.global.u64 %rd1, %rd1;
          ld.param.b64 %rd2, [dst_ptr];
          cvta.to.global.u64 %rd2, %rd2;

          // PTX template `ptx_template(%lhs: gmem<1048576xf32>, %rhs: gmem<1048576xf32>, %dst: gmem<104857...`
          add.f32 %f0, %f1, %f2;

          ret;
        }""")


def test_codegen_resolves_implicit_block_dimensions_without_mutating_ir():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @k() grid_dim=[1, 1, 1] block_dim=[128, None, None] num_warp_groups=1 {
          }
        }"""))
    [input_kernel] = module.body.ops

    kernel = make_ptx_kernel(module)
    assert input_kernel.block_dim == (128, None, None)
    assert kernel.block_dim == (128, 1, 1)
    assert kernel.ptx.count(".reqntid 128, 1, 1") == 1


def test_codegen_resolves_implicit_cluster_dimensions_without_mutating_ir():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @k() grid_dim=[4, 1, 1] block_dim=[128, 1, 1] cluster_dim=[2, None, None] {
            %cbx: index, %cby: index, %cbz: index = cluster_block_ids()
          }
        }"""))
    [input_kernel] = module.body.ops
    [ids] = input_kernel.body.ops
    assert isinstance(ids, ClusterBlockIdsOp)
    input_kernel.body.append(PTXTemplateOp.build(
        operands=ids.results,
        result_types=[],
        template="",
    ))

    kernel = make_ptx_kernel(module)

    assert input_kernel.cluster_dim == (2, None, None)
    assert kernel.cluster_dim == (2, 1, 1)
    assert ".reqnctapercluster 2, 1, 1" in kernel.ptx
    assert "%cluster_ctaid.x" in kernel.ptx
    assert "%cluster_ctaid.y" in kernel.ptx
    assert "%cluster_ctaid.z" in kernel.ptx


def test_codegen_rejects_cluster_dimension_after_grid_resolution():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @k() grid_dim=[None, 1, 1] block_dim=[128, 1, 1] cluster_dim=[2, 1, 1] {
          }
        }"""))

    with pytest.raises(ValueError, match="cluster_dim must divide grid_dim"):
        make_ptx_kernel(module)


def test_codegen_if_branches_over_its_body():
    lhs = IndexConstantOp.build(3)
    rhs = IndexConstantOp.build(4)
    condition = IndexLtOp.build(lhs.result(), rhs.result())
    guarded_value = IndexConstantOp.build(1)
    conditional = IfOp.build(condition.result(), Block(ops=[guarded_value]))
    module = ModuleOp.build(Block(ops=[KernelOp.build(
        "conditional", Block(ops=[lhs, rhs, condition, conditional]),
        grid_dim=(1, 1, 1), block_dim=(32, 1, 1),
    )]))

    ptx = make_ptx_kernel(module).ptx
    assert "setp.lt.u32 %p0" in ptx
    assert "@!%p0 bra" in ptx


def test_codegen_naive_matmul_4096():
    """Codegen smoke test for the tile/alloc/load/store/matmul module."""
    module_str = textwrap.dedent("""\
        module {
          kernel @naive_matmul_4096(%a: gmem<4096x4096xf32>, %b: gmem<4096x4096xf32>, %c: gmem<4096x4096xf32>) grid_dim=[64, 64, 1] block_dim=[32, 32, 1] {
            %cx: index = block_id.x
            %cy: index = block_id.y
            %tx: index = thread_id.x
            %ty: index = thread_id.y
            %a_block: gmem<64x4096xf32> = tile(%a: gmem<4096x4096xf32>, %cx: index, 0)
            %b_block: gmem<64x4096xf32> = tile(%b: gmem<4096x4096xf32>, %cy: index, 0)
            %c_block: gmem<64x64xf32 strides=[4096, 1]> = tile(%c: gmem<4096x4096xf32>, %cx: index, %cy: index)
            %a_thread: gmem<2x4096xf32> = tile(%a_block: gmem<64x4096xf32>, %tx: index, 0)
            %b_thread: gmem<2x4096xf32> = tile(%b_block: gmem<64x4096xf32>, %ty: index, 0)
            %c_thread: gmem<2x2xf32 strides=[4096, 1]> = tile(%c_block: gmem<64x64xf32 strides=[4096, 1]>, %tx: index, %ty: index)
            %c_thread_rmem: rmem<2x2xf32> = alloc() attrs={id: 0}
            %c_thread_rmem_init: rmem<f32> = constant() attrs={value: 0.0}
            elementwise fill(%c_thread_rmem: rmem<2x2xf32>, %c_thread_rmem_init: rmem<f32>)
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
    kernel = make_ptx_kernel(module)
    assert kernel.name == "naive_matmul_4096"
    assert kernel.grid_dim == (64, 64, 1)
    assert kernel.block_dim == (32, 32, 1)
    assert kernel.ptx == textwrap.dedent("""\
        .version 9.0
        .target sm_90a
        .address_size 64

        .visible .entry naive_matmul_4096(
          .param .b64 a_ptr,
          .param .b64 b_ptr,
          .param .b64 c_ptr
        )
        .reqntid 32, 32, 1
        {
          .reg .b32  %r<19>;
          .reg .b64  %rd<22>;
          .reg .f32  %f<9>;
          .reg .pred %p<1>;

          ld.param.b64 %rd0, [a_ptr];
          cvta.to.global.u64 %rd0, %rd0;
          ld.param.b64 %rd1, [b_ptr];
          cvta.to.global.u64 %rd1, %rd1;
          ld.param.b64 %rd2, [c_ptr];
          cvta.to.global.u64 %rd2, %rd2;

          mov.u32 %r0, %ctaid.x;

          mov.u32 %r1, %ctaid.y;

          mov.u32 %r2, %tid.x;

          mov.u32 %r3, %tid.y;

          mov.u64 %rd3, %rd0;           // to_ptr(%rd0)
          mov.u32 %r4, 1048576;         // 1048576
          shl.b32 %r5, %r0, 20;         // %r0 * 1048576
          cvt.u64.u32 %rd5, %r5;
          add.u64 %rd4, %rd3, %rd5;     // to_ptr(%rd0) + %r0 * 1048576
          mov.u64 %rd6, %rd1;           // to_ptr(%rd1)
          shl.b32 %r6, %r1, 20;         // %r1 * 1048576
          cvt.u64.u32 %rd8, %r6;
          add.u64 %rd7, %rd6, %rd8;     // to_ptr(%rd1) + %r1 * 1048576
          mov.u64 %rd9, %rd2;           // to_ptr(%rd2)
          mov.u32 %r7, 256;             // 256
          shl.b32 %r8, %r1, 8;          // %r1 * 256
          add.s32 %r9, %r5, %r8;        // %r0 * 1048576 + %r1 * 256
          cvt.u64.u32 %rd11, %r9;
          add.u64 %rd10, %rd9, %rd11;   // to_ptr(%rd2) + %r0 * 1048576 + %r1 * 256
          mov.u32 %r12, 32768;          // 32768
          shl.b32 %r13, %r3, 15;        // %r3 * 32768
          cvt.u64.u32 %rd13, %r13;
          add.u64 %rd12, %rd4, %rd13;   // to_ptr(%rd0) + %r0 * 1048576 + %r3 * 32768
          shl.b32 %r14, %r4, 15;        // %r4 * 32768
          cvt.u64.u32 %rd15, %r14;
          add.u64 %rd14, %rd7, %rd15;   // to_ptr(%rd1) + %r1 * 1048576 + %r4 * 32768
          mov.u32 %r15, 8;              // 8
          shl.b32 %r16, %r4, 3;         // %r4 * 8
          add.s32 %r17, %r13, %r16;     // %r3 * 32768 + %r4 * 8
          cvt.u64.u32 %rd17, %r17;
          add.u64 %rd16, %rd10, %rd17;  // to_ptr(%rd2) + %r0 * 1048576 + %r1 * 256 + %r3 * 32768 + %r4 * 8

          // PTX template `%c_thread_rmem: rmem<2x2xf32> = alloc() attrs={id: 0}`

          // PTX template `%c_thread_rmem_init: rmem<f32> = constant() attrs={value: 0.0}`
          mov.f32 %f4, 0f00000000;

          // PTX template `elementwise fill(%c_thread_rmem: rmem<2x2xf32>, %c_thread_rmem_init: rmem<f32>)`
          mov.f32 %f0, %f4;
          mov.f32 %f1, %f4;
          mov.f32 %f2, %f4;
          mov.f32 %f3, %f4;

          // BEGIN: `for (%k: index = 0 to 4096) {...`
          mov.b32 %r16, 0;
          $L0:

          mov.u32 %r17, 4;              // 4
          shl.b32 %r18, %r16, 2;        // %r16 * 4
          cvt.u64.u32 %rd19, %r18;
          add.u64 %rd18, %rd12, %rd19;  // to_ptr(%rd0) + %r0 * 1048576 + %r2 * 32768 + %r16 * 4
          cvt.u64.u32 %rd21, %r18;
          add.u64 %rd20, %rd14, %rd21;  // to_ptr(%rd1) + %r1 * 1048576 + %r3 * 32768 + %r16 * 4

          // PTX template `%a_k_rmem: rmem<2x1xf32> = alloc() attrs={}`

          // PTX template `move(%a_k_rmem: rmem<2x1xf32>, %a_k: gmem<2x1xf32 strides=[4096, 1]>) attrs={nc:...`
          ld.global.nc.f32 %f5, [%rd18+0];
          ld.global.nc.f32 %f6, [%rd18+16384];

          // PTX template `%b_k_rmem: rmem<2x1xf32> = alloc() attrs={}`

          // PTX template `move(%b_k_rmem: rmem<2x1xf32>, %b_k: gmem<2x1xf32 strides=[4096, 1]>) attrs={nc:...`
          ld.global.nc.f32 %f7, [%rd20+0];
          ld.global.nc.f32 %f8, [%rd20+16384];

          // PTX template `matmul(%a_k_rmem: rmem<2x1xf32>, %b_k_rmem: rmem<2x1xf32>, %c_thread_rmem: rmem<...`
          fma.rn.f32 %f0, %f5, %f7, %f0;
          fma.rn.f32 %f1, %f5, %f8, %f1;
          fma.rn.f32 %f2, %f6, %f7, %f2;
          fma.rn.f32 %f3, %f6, %f8, %f3;

          // END: `for (%k: index = 0 to 4096) {...`
          add.s32 %r16, %r16, 1;
          setp.lt.s32 %p0, %r16, 4096;
          @%p0 bra $L0;

          // PTX template `move(%c_thread: gmem<2x2xf32 strides=[4096, 1]>, %c_thread_rmem: rmem<2x2xf32>)`
          st.global.f32 [%rd16+0], %f0;
          st.global.f32 [%rd16+4], %f1;
          st.global.f32 [%rd16+16384], %f2;
          st.global.f32 [%rd16+16388], %f3;

          ret;
        }""")


def test_codegen_tma_load():
    module_str = textwrap.dedent("""\
        module {
          kernel @tma(%a: gmem<4096x4096xf32>) grid_dim=[32, 64, 1] block_dim=[128, 1, 1] {
            %cx: index = block_id.x
            %cy: index = block_id.y
            %tx: index = thread_id.x
            %ty: index = thread_id.y
            %a_block: gmem<128x64xf32 strides=[4096, 1]> = tile(%a: gmem<4096x4096xf32>, %cx: index, %cy: index)
            %a_thread: gmem<64x32xf32 strides=[4096, 1]> = tile(%a_block: gmem<128x64xf32 strides=[4096, 1]>, %tx: index, %ty: index)
            %a_smem: smem<64x32xf32 swizzle="128B"> = alloc() attrs={id: 0}
            move(%a_smem: smem<64x32xf32 swizzle="128B">, %a_thread: gmem<64x32xf32 strides=[4096, 1]>) attrs={tma: True}
          }
        }""")  # noqa: E501

    module = gpu_dialect.parse(module_str)
    kernel = make_ptx_kernel(module)
    assert kernel.name == "tma"
    assert kernel.grid_dim == (32, 64, 1)
    assert kernel.block_dim == (128, 1, 1)
    # Verify key PTX properties rather than exact string (register
    # numbering changes as passes evolve).
    assert "mbarrier.init.shared::cta.b64" in kernel.ptx
    assert "cp.async.bulk.tensor.2d" in kernel.ptx
    assert "mbarrier.try_wait.parity" in kernel.ptx
