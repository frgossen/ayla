"""Smoke test for PTX codegen: build a tiny module and check the PTX."""
# flake8: noqa: E501
import textwrap

from ayla import (
    Block,
    GmemTensorType,
    KernelOp,
    ModuleOp,
    PTXTemplateOp,
    Value,
    make_ptx_kernel,
)
from ayla.gpu_dialect import gpu_dialect, BlockIdsOp, ThreadIdsOp, F32Type


def test_codegen_cwise_add():
    n = 1 << 20
    buf_type = GmemTensorType.build((n,), F32Type())
    lhs = Value(name="lhs", type=buf_type)
    rhs = Value(name="rhs", type=buf_type)
    dst = Value(name="dst", type=buf_type)
    template = "add.f32 ${f32:s}, ${f32:a}, ${f32:b};\n"
    block_ids = BlockIdsOp.build()
    thread_ids = ThreadIdsOp.build()
    ptx_op = PTXTemplateOp.build([lhs, rhs, dst], [], template=template)
    body_block = Block(
        arguments=[lhs, rhs, dst],
        ops=[block_ids, thread_ids, ptx_op],
    )
    kernel_op = KernelOp.build(
        "cwise_add", body_block,
        grid_dim=(4, 1, 1), block_dim=(256, 1, 1),
    )
    module = ModuleOp.build(Block(ops=[kernel_op]))

    kernel = make_ptx_kernel(module)
    assert kernel.name == "cwise_add"
    assert kernel.grid_dim == (4, 1, 1)
    assert kernel.block_dim == (256, 1, 1)
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


def test_codegen_materializes_block_dim_before_lowering():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @k() grid_dim=[1, 1, 1] block_dim=[128, None, None] num_warp_groups=1 {
          }
        }"""))
    [input_kernel] = module.body.ops

    kernel = make_ptx_kernel(module)
    assert input_kernel.block_dim == (128, 1, 1)
    assert kernel.block_dim == (128, 1, 1)
    assert kernel.ptx.count(".reqntid 128, 1, 1") == 1


def test_codegen_naive_matmul_4096():
    """Codegen smoke test for the tile/alloc/load/store/matmul module."""
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
          .reg .b32  %r<23>;
          .reg .b64  %rd<22>;
          .reg .f32  %f<8>;
          .reg .pred %p<1>;

          ld.param.b64 %rd0, [a_ptr];
          cvta.to.global.u64 %rd0, %rd0;
          ld.param.b64 %rd1, [b_ptr];
          cvta.to.global.u64 %rd1, %rd1;
          ld.param.b64 %rd2, [c_ptr];
          cvta.to.global.u64 %rd2, %rd2;

          mov.u32 %r0, %ctaid.x;
          mov.u32 %r1, %ctaid.y;
          mov.u32 %r2, %ctaid.z;

          mov.u32 %r3, %tid.x;
          mov.u32 %r4, %tid.y;
          mov.u32 %r5, %tid.z;

          mov.u64 %rd3, %rd0;           // to_ptr(%rd0)
          mov.u32 %r6, 64;              // 64
          mov.u32 %r7, 1048576;         // 1048576
          shl.b32 %r8, %r0, 20;         // %r0 * 1048576
          cvt.u64.u32 %rd5, %r8;
          add.u64 %rd4, %rd3, %rd5;     // to_ptr(%rd0) + %r0 * 1048576
          mov.u64 %rd6, %rd1;           // to_ptr(%rd1)
          shl.b32 %r9, %r1, 20;         // %r1 * 1048576
          cvt.u64.u32 %rd8, %r9;
          add.u64 %rd7, %rd6, %rd8;     // to_ptr(%rd1) + %r1 * 1048576
          mov.u64 %rd9, %rd2;           // to_ptr(%rd2)
          mov.u32 %r10, 256;            // 256
          shl.b32 %r11, %r1, 8;         // %r1 * 256
          add.s32 %r12, %r8, %r11;      // %r0 * 1048576 + %r1 * 256
          cvt.u64.u32 %rd11, %r12;
          add.u64 %rd10, %rd9, %rd11;   // to_ptr(%rd2) + %r0 * 1048576 + %r1 * 256
          mov.u32 %r13, 2;              // 2
          mov.u32 %r14, 32768;          // 32768
          shl.b32 %r15, %r3, 15;        // %r3 * 32768
          cvt.u64.u32 %rd13, %r15;
          add.u64 %rd12, %rd4, %rd13;   // to_ptr(%rd0) + %r0 * 1048576 + %r3 * 32768
          shl.b32 %r16, %r4, 15;        // %r4 * 32768
          cvt.u64.u32 %rd15, %r16;
          add.u64 %rd14, %rd7, %rd15;   // to_ptr(%rd1) + %r1 * 1048576 + %r4 * 32768
          mov.u32 %r17, 8;              // 8
          shl.b32 %r18, %r4, 3;         // %r4 * 8
          add.s32 %r19, %r15, %r18;     // %r3 * 32768 + %r4 * 8
          cvt.u64.u32 %rd17, %r19;
          add.u64 %rd16, %rd10, %rd17;  // to_ptr(%rd2) + %r0 * 1048576 + %r1 * 256 + %r3 * 32768 + %r4 * 8

          // PTX template `%c_thread_rmem: rmem<2x2xf32> = alloc() attrs={id: 0, init: 0.0}`
          mov.f32 %f0, 0f00000000;
          mov.f32 %f1, 0f00000000;
          mov.f32 %f2, 0f00000000;
          mov.f32 %f3, 0f00000000;

          // BEGIN: `for (%k: index = 0 to 4096) {...`
          mov.b32 %r20, 0;
          $L0:

          mov.u32 %r21, 4;              // 4
          shl.b32 %r22, %r20, 2;        // %r20 * 4
          cvt.u64.u32 %rd19, %r22;
          add.u64 %rd18, %rd12, %rd19;  // to_ptr(%rd0) + %r0 * 1048576 + %r3 * 32768 + %r20 * 4
          cvt.u64.u32 %rd21, %r22;
          add.u64 %rd20, %rd14, %rd21;  // to_ptr(%rd1) + %r1 * 1048576 + %r4 * 32768 + %r20 * 4

          // PTX template `%a_k_rmem: rmem<2x1xf32> = alloc() attrs={id: 1}`

          // PTX template `move(%a_k_rmem: rmem<2x1xf32>, %a_k: gmem<2x1xf32 strides=[4096, 1]>) attrs={nc:...`
          ld.global.nc.f32 %f4, [%rd18+0];
          ld.global.nc.f32 %f5, [%rd18+16384];

          // PTX template `%b_k_rmem: rmem<2x1xf32> = alloc() attrs={id: 2}`

          // PTX template `move(%b_k_rmem: rmem<2x1xf32>, %b_k: gmem<2x1xf32 strides=[4096, 1]>) attrs={nc:...`
          ld.global.nc.f32 %f6, [%rd20+0];
          ld.global.nc.f32 %f7, [%rd20+16384];

          // PTX template `matmul(%a_k_rmem: rmem<2x1xf32>, %b_k_rmem: rmem<2x1xf32>, %c_thread_rmem: rmem<...`
          fma.rn.f32 %f0, %f4, %f6, %f0;
          fma.rn.f32 %f1, %f4, %f7, %f1;
          fma.rn.f32 %f2, %f5, %f6, %f2;
          fma.rn.f32 %f3, %f5, %f7, %f3;

          // END: `for (%k: index = 0 to 4096) {...`
          add.s32 %r20, %r20, 1;
          setp.lt.s32 %p0, %r20, 4096;
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
            %cx: index, %cy: index, %cz: index = block_ids()
            %tx: index, %ty: index, %tz: index = thread_ids()
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
