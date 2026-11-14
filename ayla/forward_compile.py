"""Opt-in capture and PTX compilation of a complete static inference graph."""

from __future__ import annotations

import itertools
import operator
from collections.abc import Callable
from typing import cast

from torch.fx import Graph

from .base.ir import find_ops_of_type
from .fx_compiler import discover_fx_graph, fx_region_to_module
from .gpu_dialect import (
    AllocOp,
    GmemTensorType,
    KernelOp,
    ModuleOp,
    MemTensorType,
)
from .jit_compiler import JITCompiler, PtxRoutePlan, PtxSequencePlan, ScratchBuffer
from .passes.schedule_scalar_graph import ScheduleScalarGraph
from .passes.schedule_persistent_graph import SchedulePersistentGraph
from .gpu_dialect import MatmulOp
from .jit_compiler import ScratchInit
import torch
from .passes.verify_gpu_dialect import verify_gpu_dialect
from .pipelines.canonicalize import Canonicalize
from .ptx_codegen import PtxKernel, make_ptx_kernel
from .torch_custom_call import OperandKind, make_torch_custom_call

_names = itertools.count()


def compile_forward_module(
    module: ModuleOp, *, reference: bool = False, tile_scheduler: str | None = None,
    tile_order: str = "source",
    ptx_optimization_level: int | None = None,
    emit_kernel: Callable[[ModuleOp], PtxKernel] | None = None,
) -> PtxSequencePlan:
    """Compile a graph; optionally schedule resident CTA tiles across phases.

    ``tile_scheduler`` selects static assignment or a dynamic work queue.
    Shapes and strides must already be specialized. The default retains the
    measured phase schedules.
    ``ptx_optimization_level`` overrides the kernel's existing PTX JIT level
    (0–4). Lower levels can reduce speculative scheduling and spills; they
    require end-to-end qualification before use in a performance profile.
    ``emit_kernel`` optionally observes scheduled IR and emits the kernel,
    allowing profiling to retain the exact lowering without global patches.
    """
    if tile_scheduler not in (None, "static", "dynamic") or tile_order not in ("source", "memory"):
        raise ValueError("invalid tile scheduler or traversal order")
    if reference and tile_scheduler is not None:
        raise ValueError("tile scheduling requires a resident tensor-core graph")
    if ptx_optimization_level is not None and (
        type(ptx_optimization_level) is not int or ptx_optimization_level not in range(5)
    ):
        raise ValueError("ptx_optimization_level must be an integer in [0, 4]")
    verify_gpu_dialect(module)
    [kernel] = module.body.ops
    assert isinstance(kernel, KernelOp)
    if ptx_optimization_level is not None:
        kernel.attrs["ptx_optimization_level"] = ptx_optimization_level
    external_arguments = len(kernel.signature)
    bindings: list[int | ScratchBuffer] = list(range(external_arguments))
    # Global allocations are host-owned scratch arguments, never PTX allocs.
    for allocation in list(find_ops_of_type(module, AllocOp)):
        typ = allocation.result().type
        if isinstance(typ, GmemTensorType):
            if allocation.parent_block is not kernel.body:
                raise ValueError("forward scratch allocations must be at kernel scope")
            kernel.body.ops.remove(allocation)
            kernel.body.append_argument(allocation.result())
            bindings.append(ScratchBuffer(typ))
    matmuls = find_ops_of_type(module, MatmulOp)
    persistent = (
        not reference
        and bool(matmuls)
        and cast(MemTensorType, matmuls[0].lhs.type).static_shape[0] >= 128
    )
    if tile_scheduler is not None and not persistent:
        raise ValueError("tile scheduling requires a resident tensor-core graph")
    if persistent:
        blocks = torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).multi_processor_count
        if tile_scheduler is None:
            scheduler = SchedulePersistentGraph(
                blocks=blocks, external_arguments=external_arguments,
            )
        else:
            scheduler = SchedulePersistentGraph(
                blocks=blocks, external_arguments=external_arguments,
                tile_scheduler=tile_scheduler, tile_order=tile_order,
            )
        module = scheduler.run(module)
        scheduled_kernel = cast(KernelOp, module.body.ops[0])
        sync_type = scheduled_kernel.body.arguments[-1].type
        assert isinstance(sync_type, GmemTensorType)
        bindings.append(
            ScratchBuffer(
                sync_type, ScratchInit.ZERO,
                reset_each_launch=tile_scheduler is not None,
            )
        )
        module = Canonicalize().run(module)
        [kernel] = module.body.ops
        assert isinstance(kernel, KernelOp)
        kept = [
            (argument, binding)
            for argument, binding in zip(kernel.body.arguments, bindings, strict=True)
            if argument.users()
        ]
        kernel.body.arguments = [argument for argument, _ in kept]
        for index, argument in enumerate(kernel.body.arguments):
            argument.result_index = index
        bindings = [binding for _, binding in kept]
    else:
        ScheduleScalarGraph().run(module)
    return PtxSequencePlan(
        [PtxRoutePlan((emit_kernel or make_ptx_kernel)(module), bindings)],
        reuse_scratch=bool(persistent),
    )


def capture_forward_graph(graph: Graph) -> None:
    """Inductor post-grad pass replacing the complete graph with one Ayla op.

    This strict, static inference path fails on unsupported live operations.
    Prefill uses tensor cores on a bounded cooperative grid. Small graphs
    and ``reference=True`` use the one-CTA scalar reference schedule.
    """
    match = discover_fx_graph(graph)
    name = f"forward_{next(_names)}"
    module = fx_region_to_module(match, name=name)
    [kernel] = module.body.ops
    assert isinstance(kernel, KernelOp)
    signature = list(kernel.signature)
    compiler = JITCompiler(
        name=name, signature=signature, compile_fn=compile_forward_module
    )
    tensors = [node.meta["val"] for node in match.inputs]
    output_types = [
        typ for typ in signature[len(match.inputs) :] if isinstance(typ, GmemTensorType)
    ]
    tensor_signature = tuple(
        (tuple(t.shape), tuple(t.stride()), t.dtype) for t in tensors
    ) + tuple(
        (typ.static_shape, tuple(typ.strides), match.outputs[i].meta["val"].dtype)
        for i, typ in enumerate(output_types)
    )
    # Compile before graph mutation, so unsupported schedules never leave a
    # partly rewritten graph behind.
    compiler.prepare_module(tensor_signature, module)
    replacement = make_torch_custom_call(
        compiler,
        [OperandKind.R] * len(match.inputs) + [OperandKind.RESULT] * len(match.outputs),
    )
    output = next(node for node in graph.nodes if node.op == "output")
    with graph.inserting_before(output):
        call = graph.call_function(replacement, args=match.inputs)
        for i, node in enumerate(match.outputs):
            result = (
                call
                if len(match.outputs) == 1
                else graph.call_function(operator.getitem, (call, i))
            )
            result.meta = dict(node.meta)
            result.meta["val"] = node.meta["val"].new_empty_strided(
                node.meta["val"].shape,
                node.meta["val"].stride(),
            )
            # Returned intermediates can also be used by other graph outputs.
            # Only the output node is rewritten; the original region is dead
            # after all outputs have been replaced.
            output.replace_input_with(node, result)
    graph.eliminate_dead_code()
    graph.lint()
