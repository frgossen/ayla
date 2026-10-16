import pytest
import torch
from torch.fx import Graph, GraphModule, Node
from torch._inductor.pattern_matcher import PatternMatcherPass

from ayla.base.ir import find_ops_of_type
from ayla.fx_compiler import (
    FxRegionMatch,
    discover_supported_fx_region,
    fx_region_to_module,
    make_fx_to_ayla_rewrite,
)
from ayla.gpu_dialect import (
    AllocOp,
    BF16Type,
    ConstantOp,
    ElementwiseOp,
    GmemTensorType,
    KernelOp,
    MatmulOp,
)
from ayla.gemm_collection.gemm import schedule_matmul
from ayla.jit_compiler import PtxRoutePlan
from ayla.passes.verify_gpu_dialect import verify_gpu_dialect
from ayla.ptx_codegen import make_ptx_kernel
from ayla.torch_custom_call import OperandKind


def _meta(node: Node, shape, dtype=torch.bfloat16, strides=None) -> Node:
    if strides is None:
        node.meta["val"] = torch.empty(shape, dtype=dtype, device="meta")
    else:
        node.meta["val"] = torch.empty_strided(
            shape, strides, dtype=dtype, device="meta",
        )
    return node


def _gemm_region(
    *, m: int = 128, n: int = 64, k: int = 64, sigmoid: bool = False,
) -> FxRegionMatch:
    graph = Graph()
    lhs = _meta(graph.placeholder("lhs"), (m, k))
    weight = _meta(graph.placeholder("weight"), (n, k))
    weight_t = _meta(graph.call_function(
        torch.ops.aten.permute.default, args=(weight, [1, 0]),
    ), (k, n), strides=(1, k))
    output = _meta(graph.call_function(
        torch.ops.aten.mm.default, args=(lhs, weight_t),
    ), (m, n))
    if sigmoid:
        output = _meta(graph.call_function(
            torch.ops.aten.sigmoid.default, args=(output,),
        ), (m, n))
    graph.output(output)
    return FxRegionMatch(output, (lhs, weight))


def test_gemm_fx_region_imports_to_high_level_gpu_ir():
    match = _gemm_region()
    module = fx_region_to_module(match, name="gemm")

    verify_gpu_dialect(module)
    [kernel] = module.body.ops
    assert isinstance(kernel, KernelOp)
    lhs, weight, output = kernel.body.arguments
    assert [lhs.name, weight.name, output.name] == [
        "lhs", "weight", match.output.name,
    ]
    zero, matmul = kernel.body.ops
    assert isinstance(zero, ConstantOp)
    assert isinstance(matmul, MatmulOp)
    assert (matmul.lhs, matmul.rhs, matmul.dst) == (lhs, weight, output)
    assert matmul.init is zero.result()


def test_gemm_epilogue_imports_through_an_intermediate_allocation():
    match = _gemm_region(sigmoid=True)
    module = fx_region_to_module(match, name="gemm_sigmoid")

    verify_gpu_dialect(module)
    [kernel] = module.body.ops
    assert isinstance(kernel, KernelOp)
    allocation, _zero, matmul, epilogue = kernel.body.ops
    assert isinstance(allocation, AllocOp)
    assert isinstance(matmul, MatmulOp)
    assert isinstance(epilogue, ElementwiseOp)
    assert matmul.dst is allocation.result()
    assert epilogue.fn == "sigmoid"
    assert epilogue.src(0) is allocation.result()
    assert epilogue.dst is kernel.body.arguments[-1]
    assert find_ops_of_type(module, MatmulOp) == [matmul]


def test_discovers_complete_gemm_region_and_ordered_inputs():
    expected = _gemm_region(sigmoid=True)
    # FX/AOT is free to declare placeholders in an order unrelated to use.
    lhs, weight = expected.inputs
    lhs.prepend(weight)

    match = discover_supported_fx_region(expected.output)

    assert match is not None
    assert match.output is expected.output
    assert match.inputs == expected.inputs
    projection = [
        node for node in expected.output.graph.nodes
        if node.target is torch.ops.aten.mm.default
    ][0]
    assert discover_supported_fx_region(projection) is None


def test_fx_region_import_rejects_intermediate_external_users():
    match = _gemm_region(sigmoid=True)
    projection = [
        node for node in match.output.graph.nodes
        if node.target is torch.ops.aten.mm.default
    ][0]
    match.output.graph.call_function(torch.ops.aten.neg.default, (projection,))

    assert discover_supported_fx_region(match.output) is None

    with pytest.raises(ValueError, match="region value has external users"):
        fx_region_to_module(match, name="gemm_sigmoid")


def test_fx_region_import_rejects_unsupported_targets():
    match = _gemm_region()
    [lhs, _weight] = match.inputs
    mm = match.output
    with mm.graph.inserting_before(mm):
        negated = _meta(mm.graph.call_function(
            torch.ops.aten.neg.default, args=(lhs,),
        ), (128, 64))
    mm.replace_input_with(lhs, negated)

    with pytest.raises(ValueError, match="unsupported FX target"):
        fx_region_to_module(match, name="gemm_neg")


def _compile_gemm(module):
    scheduled = schedule_matmul(
        module,
        output_n_tile=64,
        output_tma_inner_size=None,
    )
    return PtxRoutePlan(make_ptx_kernel(scheduled))


def test_compiled_fx_rewrite_prepares_kernel_and_registers_meta_op():
    match = _gemm_region()

    patterns = PatternMatcherPass(pass_name="compiled_fx_gemm_test")
    bf16 = BF16Type()
    make_fx_to_ayla_rewrite(
        patterns,
        root=torch.ops.aten.mm.default,
        name="compiled_fx_gemm",
        signature=[
            GmemTensorType.build([128, 64], bf16),
            GmemTensorType.build([64, 64], bf16),
            GmemTensorType.build([128, 64], bf16),
        ],
        signature_mapping=[
            OperandKind.R, OperandKind.R, OperandKind.RESULT,
        ],
        compile_region=_compile_gemm,
        namespace="ayla_compiled_fx_rewrite_test",
    )
    replacement = (
        torch.ops.ayla_compiled_fx_rewrite_test.compiled_fx_gemm.default
    )

    module = GraphModule({}, match.output.graph)
    assert patterns.apply(module) == 1

    calls = [
        node for node in module.graph.nodes
        if node.op == "call_function"
    ]
    assert len(calls) == 1
    assert calls[0].target is replacement
    result = replacement(
        torch.empty((128, 64), dtype=torch.bfloat16, device="meta"),
        torch.empty((64, 64), dtype=torch.bfloat16, device="meta"),
    )
    assert result.shape == (128, 64)
