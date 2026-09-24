"""Smoke test for the PTX runner.

Skips when CUDA isn't available. When it runs, it loads a tiny element-wise
add PTX kernel via the driver, launches it on torch tensors, and checks the
result.
"""
import textwrap

import pytest
import torch

import ayla.ptx_runtime as ptx_runtime
from ayla.base.ir import Block, Value
from ayla.gpu_dialect import (
    AllocOp,
    BarrierOp,
    ClusterBlockIdsOp,
    CollectiveAxis,
    CollectiveLevel,
    F32Type,
    GmemTensorType,
    KernelOp,
    ModuleOp,
    MoveOp,
    PTXTemplateOp,
    SmemTensorType,
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


class _FakeLauncherExtension:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.occupancy_args = None

    def module_load_data(self, ptx, optimization_level):
        return 1

    def module_unload(self, module):
        pass

    def module_get_function(self, module, name):
        return 2

    def max_resident_grid_blocks(
        self, function, grid_dim, block_dim, shared_mem_bytes, cluster_dim,
    ):
        self.occupancy_args = (
            function, grid_dim, block_dim, shared_mem_bytes, cluster_dim,
        )
        return self.capacity


@pytest.mark.parametrize(
    ("cluster_dim", "capacity"),
    [(None, 8), ((2, 1, 1), 12)],
)
def test_resident_grid_rejects_unsafe_launch_before_execution(
    monkeypatch, cluster_dim, capacity,
):
    fake = _FakeLauncherExtension(capacity)
    monkeypatch.setattr(ptx_runtime, "_get_ext", lambda: fake)
    kernel = PtxKernel(
        ptx=SIMPLE_ADD_PTX,
        name="grid_sync",
        signature=[],
        grid_dim=(16, 1, 1),
        block_dim=(256, 1, 1),
        cluster_dim=cluster_dim,
        shared_mem_bytes=32768,
        requires_resident_grid=True,
    )

    with pytest.raises(
        RuntimeError,
        match=rf"grid_sync.*requires 16 resident blocks.*capacity is {capacity}",
    ):
        make_ptx_launcher(kernel).ensure_loaded()

    assert fake.occupancy_args == (
        2,
        (16, 1, 1),
        (256, 1, 1),
        32768,
        cluster_dim or (0, 0, 0),
    )


@requires_cuda
def test_real_occupancy_rejects_an_oversubscribed_resident_grid():
    torch.empty(1, device="cuda")
    kernel = PtxKernel(
        ptx=SIMPLE_ADD_PTX,
        name="cwise_add",
        signature=[],
        grid_dim=(1_000_000, 1, 1),
        block_dim=(256, 1, 1),
        requires_resident_grid=True,
    )

    with pytest.raises(
        RuntimeError, match="requires 1000000 resident blocks",
    ):
        make_ptx_launcher(kernel).ensure_loaded()


@requires_cuda
def test_real_cluster_occupancy_rejects_an_oversubscribed_resident_grid():
    torch.empty(1, device="cuda")
    sync_type = GmemTensorType.build((2,), F32Type())
    sync = Value("sync", sync_type)
    module = ModuleOp.build(Block(ops=[KernelOp.build(
        "cluster_grid_sync",
        Block(arguments=[sync], ops=[BarrierOp.build("grid", sync)]),
        grid_dim=(1_000_000, 1, 1),
        block_dim=(128, 1, 1),
        cluster_dim=(2, 1, 1),
    )]))

    with pytest.raises(
        RuntimeError, match="requires 1000000 resident blocks",
    ):
        make_ptx_launcher(make_ptx_kernel(module)).ensure_loaded()


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
def test_grid_barrier_makes_prior_block_writes_visible():
    buf_type = GmemTensorType.build((2,), F32Type())
    data = Value("data", buf_type)
    sync = Value("sync", buf_type)
    write = PTXTemplateOp.build(
        operands=[data],
        result_types=[],
        template=textwrap.dedent(r"""\
            mov.u32 ${b32:bid}, %ctaid.x;
            shl.b32 ${b32:value}, ${b32:bid}, 23;
            add.u32 ${b32:value}, ${b32:value}, 1065353216;
            mul.wide.u32 ${b64:offset}, ${b32:bid}, 4;
            add.u64 ${b64:address}, ${b64:operand.data.ptr}, ${b64:offset};
            st.release.gpu.global.u32 [${b64:address}], ${b32:value};
            """),
        operand_names=["data"],
        result_names=[],
    )
    read = PTXTemplateOp.build(
        operands=[data],
        result_types=[],
        template=textwrap.dedent(r"""\
            mov.u32 ${b32:bid}, %ctaid.x;
            setp.ne.u32 ${pred:skip}, ${b32:bid}, 0;
            @${pred:skip} bra ${label:done};
            ld.acquire.gpu.global.u32 ${b32:value}, [${b64:operand.data.ptr}+4];
            st.release.gpu.global.u32 [${b64:operand.data.ptr}], ${b32:value};
            ${label:done}:
            """),
        operand_names=["data"],
        result_names=[],
    )
    module = ModuleOp.build(Block(ops=[KernelOp.build(
        "grid_barrier",
        Block(
            arguments=[data, sync],
            ops=[write, BarrierOp.build("grid", sync), read],
        ),
        grid_dim=(2, 1, 1),
        block_dim=(1, 1, 1),
    )]))
    data_tensor = torch.zeros(2, device="cuda", dtype=torch.float32)
    sync_tensor = torch.zeros(2, device="cuda", dtype=torch.float32)

    make_ptx_launcher(make_ptx_kernel(module))(data_tensor, sync_tensor)

    torch.cuda.synchronize()
    assert torch.equal(data_tensor, torch.tensor([2.0, 2.0], device="cuda"))


@requires_cuda
def test_tma_multicast_copies_to_every_cluster_block():
    input_type = GmemTensorType.build((16, 16), F32Type())
    output_type = GmemTensorType.build((2,), F32Type())
    input_value = Value("input", input_type)
    output = Value("output", output_type)
    smem = AllocOp.build(SmemTensorType.build((16, 16), F32Type()), id=0)
    load = MoveOp.build(
        smem.result(),
        input_value,
        collective={
            axis: CollectiveLevel.CLUSTER for axis in CollectiveAxis
        },
        attrs={
            "l2_cache_hint": "evict_last",
            "tma": True,
        },
        multicast=True,
    )
    ids = ClusterBlockIdsOp.build()
    write = PTXTemplateOp.build(
        operands=[smem.result(), output, ids.x],
        result_types=[],
        template=textwrap.dedent("""\
            ld.shared.f32 ${f32:value}, [${b32:operand.smem.ptr}];
            mul.wide.u32 ${b64:offset}, ${b32:operand.rank.val}, 4;
            add.u64 ${b64:dst}, ${b64:operand.output.ptr}, ${b64:offset};
            st.global.f32 [${b64:dst}], ${f32:value};
        """),
        operand_names=["smem", "output", "rank"],
        result_names=[],
        description="write multicast result",
    )
    module = ModuleOp.build(Block(ops=[KernelOp.build(
        "tma_multicast",
        Block(arguments=[input_value, output], ops=[smem, load, ids, write]),
        grid_dim=(2, 1, 1),
        block_dim=(32, 1, 1),
        cluster_dim=(2, 1, 1),
    )]))
    launcher = make_ptx_launcher(make_ptx_kernel(module))
    input_tensor = torch.full(
        (16, 16), 7.0, dtype=torch.float32, device="cuda",
    )
    actual = torch.empty(2, dtype=torch.float32, device="cuda")

    launcher(input_tensor, actual)
    torch.cuda.synchronize()

    assert actual.cpu().tolist() == [7.0, 7.0]
