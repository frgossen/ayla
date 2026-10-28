"""Tests for packed SwiGLU matching, scheduling, and compilation."""

import torch

from ayla.base.ir import find_ops_of_type
from ayla.fx_compiler import (
    FxRegionMatch,
    fx_region_to_module,
)
from ayla.gemm_collection.swiglu import (
    compile_swiglu,
    is_swiglu_region,
    schedule_swiglu,
)
from ayla.gpu_dialect import (
    BarrierOp,
    KernelOp,
    MatmulOp,
)
from ayla.jit_compiler import PtxRoutePlan, ScratchInit
from ayla.passes.verify_gpu_dialect import verify_gpu_dialect
from ayla.ptx_codegen import make_ptx_kernel
from tests.test_fx_compiler import _swiglu_region


def test_schedules_imported_swiglu_as_one_kernel():
    match = _swiglu_region(m=1, hidden=6_656, intermediate=19_968)
    assert is_swiglu_region(match)
    module = fx_region_to_module(match, name="swiglu")

    scheduled = schedule_swiglu(module)

    verify_gpu_dialect(scheduled)
    [kernel] = scheduled.body.ops
    assert isinstance(kernel, KernelOp)
    assert kernel.grid_dim == (104, 1, 1)
    matmul_labels = {
        str(op.attrs.get("label")).split("_", 1)[0]
        for op in find_ops_of_type(scheduled, MatmulOp)
    }
    assert matmul_labels == {"gate", "up", "down"}
    assert any(
        barrier.scope == "grid"
        for barrier in find_ops_of_type(scheduled, BarrierOp)
    )
    assert [argument.name for argument in kernel.body.arguments] == [
        "lhs", "packed_weight", "down_weight", "hidden", "output", "sync",
    ]
    ptx_kernel = make_ptx_kernel(scheduled)
    assert ".entry swiglu" in ptx_kernel.ptx


def test_rejects_non_swiglu_fx_region():
    match = _swiglu_region(m=1, hidden=6_656, intermediate=19_968)
    projection = next(
        node for node in match.output.graph.nodes
        if node.target is torch.ops.aten.mm.default
    )

    assert not is_swiglu_region(FxRegionMatch(projection, match.inputs))


def test_compiles_swiglu_plan():
    small = fx_region_to_module(
        _swiglu_region(m=1, hidden=6_656, intermediate=19_968),
        name="small_swiglu",
    )
    small_plan = compile_swiglu(small)
    assert isinstance(small_plan, PtxRoutePlan)
    assert small_plan.bindings == (
        0,
        1,
        2,
        ScratchInit.EMPTY,
        3,
        ScratchInit.ZERO,
    )
