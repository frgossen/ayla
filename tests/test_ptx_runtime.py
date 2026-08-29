"""Smoke test for the PTX runner.

Skips when CUDA isn't available. When it runs, it loads a tiny element-wise
add PTX kernel via the driver, launches it on torch tensors, and checks the
result.
"""
import textwrap

import pytest
import torch

from ayla import Block, KernelOp, ModuleOp, PTXTemplateOp, Value, make_ptx_kernel
from ayla.gpu_dialect import F32Type, GmemTensorType
from ayla.ptx_codegen import PtxKernel
from ayla.ptx_runtime import make_ptx_launcher
from ayla.targets import BLACKWELL

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="no CUDA"
)


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
    ptx_launcher(a, b, c)

    torch.cuda.synchronize()
    assert torch.allclose(c, a + b)


@requires_cuda
def test_generated_blackwell_memory_kernel():
    buf_type = GmemTensorType.build((1,), F32Type())
    src = Value(name="src", type=buf_type)
    dst = Value(name="dst", type=buf_type)
    copy = PTXTemplateOp.build(
        [src, dst], [],
        template=textwrap.dedent("""\
            ld.global.f32 ${f32:value}, [${b64:operand.src.ptr}];
            st.global.f32 [${b64:operand.dst.ptr}], ${f32:value};
        """),
        operand_names=["src", "dst"],
    )
    kernel_op = KernelOp.build(
        "blackwell_copy", Block(arguments=[src, dst], ops=[copy]),
        grid_dim=(1, 1, 1), block_dim=(1, 1, 1),
    )
    kernel = make_ptx_kernel(
        ModuleOp.build(Block(ops=[kernel_op]), target=BLACKWELL)
    )
    launcher = make_ptx_launcher(kernel)
    source = torch.tensor([3.25], dtype=torch.float32, device="cuda")
    destination = torch.zeros_like(source)

    launcher(source, destination)
    torch.cuda.synchronize()

    assert destination.item() == source.item()


@requires_cuda
def test_generated_blackwell_cluster_kernel():
    output_type = GmemTensorType.build((2,), F32Type())
    output = Value(name="output", type=output_type)
    write_rank = PTXTemplateOp.build(
        [output], [],
        template=textwrap.dedent("""\
            mov.u32 ${b32:cta}, %ctaid.x;
            mov.u32 ${b32:rank}, %cluster_ctarank;
            cvt.rn.f32.u32 ${f32:value}, ${b32:rank};
            mad.wide.u32 ${b64:address}, ${b32:cta}, 4, ${b64:operand.output.ptr};
            st.global.f32 [${b64:address}], ${f32:value};
        """),
        operand_names=["output"],
    )
    kernel_op = KernelOp.build(
        "blackwell_cluster", Block(arguments=[output], ops=[write_rank]),
        grid_dim=(2, 1, 1), block_dim=(1, 1, 1), cluster_dim=(2, 1, 1),
    )
    kernel = make_ptx_kernel(
        ModuleOp.build(Block(ops=[kernel_op]), target=BLACKWELL)
    )
    destination = torch.full((2,), -1.0, dtype=torch.float32, device="cuda")

    make_ptx_launcher(kernel)(destination)
    torch.cuda.synchronize()

    torch.testing.assert_close(
        destination, torch.tensor([0.0, 1.0], device="cuda")
    )


@requires_cuda
def test_ptx_jit_failure_includes_driver_logs():
    bad_ptx = SIMPLE_ADD_PTX.replace(".target sm_70", ".target sm_999")
    buf_type = GmemTensorType.build((1,), F32Type())
    launcher = make_ptx_launcher(PtxKernel(
        ptx=bad_ptx,
        name="cwise_add",
        signature=[buf_type, buf_type, buf_type],
        grid_dim=(1, 1, 1),
        block_dim=(1, 1, 1),
    ))
    tensor = torch.zeros(1, dtype=torch.float32, device="cuda")

    with pytest.raises(RuntimeError, match="CUDA JIT error log") as exc:
        launcher(tensor, tensor, tensor)
    assert "sm_999" in str(exc.value)
