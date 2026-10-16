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
    BlockIdOp,
    ConstantOp,
    ElementwiseOp,
    GmemTensorType,
    KernelOp,
    MatmulOp,
    ReshapeOp,
    TileOp,
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


def _addmm_region(
    *, m: int = 128, n: int = 64, k: int = 64, beta: int = 1,
) -> FxRegionMatch:
    graph = Graph()
    bias = _meta(graph.placeholder("bias"), (n,))
    lhs = _meta(graph.placeholder("lhs"), (m, k))
    weight = _meta(graph.placeholder("weight"), (n, k))
    weight_t = _meta(graph.call_function(
        torch.ops.aten.permute.default, args=(weight, [1, 0]),
    ), (k, n), strides=(1, k))
    output = _meta(graph.call_function(
        torch.ops.aten.addmm.default,
        args=(bias, lhs, weight_t),
        kwargs={} if beta == 1 else {"beta": beta},
    ), (m, n))
    graph.output(output)
    return FxRegionMatch(output, (bias, lhs, weight))


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


def test_addmm_initializes_the_matmul_from_the_broadcast_bias():
    match = _addmm_region()
    assert discover_supported_fx_region(match.output) == match
    module = fx_region_to_module(match, name="addmm")

    verify_gpu_dialect(module)
    [kernel] = module.body.ops
    assert isinstance(kernel, KernelOp)
    bias, lhs, weight, output = kernel.body.arguments
    assert isinstance(bias.type, GmemTensorType)
    assert tuple(bias.type.sizes) == (128, 64)
    assert tuple(bias.type.strides) == (0, 1)
    [matmul] = kernel.body.ops
    assert isinstance(matmul, MatmulOp)
    assert (matmul.lhs, matmul.rhs, matmul.dst) == (lhs, weight, output)
    assert matmul.init is bias


def test_scaled_addmm_is_not_imported():
    with pytest.raises(ValueError, match="scaled matmuls are unsupported"):
        fx_region_to_module(_addmm_region(beta=2), name="addmm")


def test_bmm_imports_one_matmul_per_grid_z_batch():
    graph = Graph()
    lhs = _meta(graph.placeholder("lhs"), (3, 128, 64))
    weight = _meta(graph.placeholder("weight"), (3, 32, 64))
    weight_t = _meta(graph.call_function(
        torch.ops.aten.permute.default, args=(weight, [0, 2, 1]),
    ), (3, 64, 32), strides=(2048, 1, 64))
    output = _meta(graph.call_function(
        torch.ops.aten.bmm.default, args=(lhs, weight_t),
    ), (3, 128, 32))
    graph.output(output)
    match = discover_supported_fx_region(output)
    assert match == FxRegionMatch(output, (lhs, weight))

    module = fx_region_to_module(match, name="bmm")

    verify_gpu_dialect(module)
    [kernel] = module.body.ops
    assert isinstance(kernel, KernelOp)
    assert kernel.grid_dim == (None, None, 3)
    [batch] = find_ops_of_type(module, BlockIdOp)
    tiles = find_ops_of_type(module, TileOp)
    assert [tile.arg for tile in tiles] == kernel.body.arguments
    assert all(tile.indices[0] is batch.value for tile in tiles)
    [matmul] = find_ops_of_type(module, MatmulOp)
    assert all(
        isinstance(operand.definition, ReshapeOp)
        for operand in (matmul.lhs, matmul.rhs, matmul.dst)
    )
    assert [tuple(operand.type.sizes) for operand in (
        matmul.lhs, matmul.rhs, matmul.dst,
    )] == [(128, 64), (32, 64), (128, 32)]


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
