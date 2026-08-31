"""Smoke test for the PTX runner.

Skips when CUDA isn't available. When it runs, it loads a tiny element-wise
add PTX kernel via the driver, launches it on torch tensors, and checks the
result.
"""
import textwrap

import pytest
import torch

from ayla.gpu_dialect import F32Type, GmemTensorType
from ayla.ptx_codegen import PtxKernel
from ayla.ptx_runtime import make_ptx_launcher

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
    ptx_launcher.ensure_loaded()
    ptx_launcher(a, b, c)

    torch.cuda.synchronize()
    assert torch.allclose(c, a + b)
