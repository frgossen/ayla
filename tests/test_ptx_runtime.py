"""Smoke test for the PTX runner.

Skips when CUDA isn't available. When it runs, it loads a tiny element-wise
add PTX kernel via the driver, launches it on torch tensors, and checks the
result.
"""
import textwrap

import pytest
import torch

from ayla.base.ir import Block, Value
from ayla.gpu_dialect import (
    BarrierOp,
    ClusterBlockIdsOp,
    F32Type,
    GmemTensorType,
    KernelOp,
    ModuleOp,
    PTXTemplateOp,
)
from ayla.ptx_codegen import PtxKernel, make_ptx_kernel
from ayla.ptx_runtime import make_ptx_launcher
from tests.testing_utils import requires_cuda


# c[i] = a[i] + b[i]; assumes grid_x * block_x == len(a).
SIMPLE_ADD_PTX = textwrap.dedent("""\
    .version 8.0
    .target sm_70
    .address_size 64

    .visible .entry cwise_add(
        .param .b64 a,
        .param .b64 b,
        .param .b64 c
    )
    {
        .reg .b32 %r<5>;
        .reg .b64 %rd<10>;
        .reg .f32 %f<4>;

        ld.param.u64        %rd1, [a];
        ld.param.u64        %rd2, [b];
        ld.param.u64        %rd3, [c];

        mov.u32             %r2, %ctaid.x;
        mov.u32             %r3, %ntid.x;
        mov.u32             %r4, %tid.x;
        mad.lo.s32          %r0, %r2, %r3, %r4;

        mul.wide.s32        %rd5, %r0, 4;
        cvta.to.global.u64  %rd4, %rd1;
        add.s64             %rd6, %rd4, %rd5;
        cvta.to.global.u64  %rd7, %rd2;
        add.s64             %rd8, %rd7, %rd5;
        ld.global.f32       %f1, [%rd6];
        ld.global.f32       %f2, [%rd8];
        add.f32             %f3, %f1, %f2;
        cvta.to.global.u64  %rd9, %rd3;
        add.s64             %rd0, %rd9, %rd5;
        st.global.f32       [%rd0], %f3;

        ret;
    }
""")


@requires_cuda
def test_cwise_add():
    n = 1024
    block = 256
    grid = n // block

    a = torch.arange(n, dtype=torch.float32, device="cuda")
    b = a * 2
    c = torch.empty_like(a)

    buf_type = GmemTensorType.build((n,), F32Type())
    kernel = PtxKernel(
        ptx=SIMPLE_ADD_PTX,
        name="cwise_add",
        signature=[buf_type, buf_type, buf_type],
        grid_dim=(grid, 1, 1),
        block_dim=(block, 1, 1),
    )
    ptx_launcher = make_ptx_launcher(kernel)
    ptx_launcher.ensure_loaded()
    ptx_launcher(a, b, c)

    torch.cuda.synchronize()
    assert torch.allclose(c, a + b)


@requires_cuda
def test_cluster_block_ids_and_barrier_reach_the_launch():
    output_type = GmemTensorType.build((2,), F32Type())
    output = Value("output", output_type)
    ids = ClusterBlockIdsOp.build()
    write_x = PTXTemplateOp.build(
        operands=[output, ids.x],
        result_types=[],
        template=textwrap.dedent("""\
            mov.u32 ${b32:cta}, %ctaid.x;
            mul.wide.u32 ${b64:offset}, ${b32:cta}, 4;
            add.u64 ${b64:dst}, ${b64:operand.output.ptr}, ${b64:offset};
            cvt.rn.f32.u32 ${f32:value}, ${b32:operand.x.val};
            st.global.f32 [${b64:dst}], ${f32:value};
        """),
        operand_names=["output", "x"],
        result_names=[],
        description="write cluster block x",
    )
    module = ModuleOp.build(Block(ops=[KernelOp.build(
        "cluster_block_ids",
        Block(arguments=[output], ops=[
            ids,
            BarrierOp.build("cluster"),
            write_x,
        ]),
        grid_dim=(2, 1, 1),
        block_dim=(1, 1, 1),
        cluster_dim=(2, 1, 1),
    )]))
    launcher = make_ptx_launcher(make_ptx_kernel(module))
    actual = torch.empty(2, device="cuda")

    launcher(actual)
    torch.cuda.synchronize()

    assert actual.cpu().tolist() == [0.0, 1.0]


@requires_cuda
def test_resident_grid_rejects_more_blocks_than_can_be_active():
    buf_type = GmemTensorType.build((1,), F32Type())
    kernel = PtxKernel(
        ptx=SIMPLE_ADD_PTX,
        name="cwise_add",
        signature=[buf_type, buf_type, buf_type],
        grid_dim=(1_000_000, 1, 1),
        block_dim=(256, 1, 1),
        requires_resident_grid=True,
    )

    with pytest.raises(RuntimeError, match="requiring 1000000 resident blocks"):
        make_ptx_launcher(kernel).ensure_loaded()
