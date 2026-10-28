import operator

import pytest
import torch
from torch.fx import Graph, GraphModule, Node
from torch._inductor.pattern_matcher import PatternMatcherPass

import ayla.jit_compiler as jit_compiler
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
    ElementwiseOp,
    F32Type,
    GmemTensorType,
    KernelOp,
    MatmulOp,
    ReshapeOp,
    TileOp,
)
from ayla.gemm_collection.gemm import schedule_matmul
from ayla.jit_compiler import PtxRoutePlan, ScratchInit
from ayla.passes.verify_gpu_dialect import verify_gpu_dialect
from ayla.ptx_codegen import PtxKernel, make_ptx_kernel
from ayla.torch_custom_call import OperandKind


def _meta(node: Node, shape, dtype=torch.bfloat16, strides=None) -> Node:
    if strides is None:
        node.meta["val"] = torch.empty(shape, dtype=dtype, device="meta")
    else:
        node.meta["val"] = torch.empty_strided(
            shape, strides, dtype=dtype, device="meta",
        )
    return node


def _swiglu_region(
    *, m: int = 4, hidden: int = 8, intermediate: int = 6,
) -> FxRegionMatch:
    graph = Graph()
    packed_intermediate = 2 * intermediate
    lhs = _meta(graph.placeholder("lhs"), (m, hidden))
    packed_weight = _meta(
        graph.placeholder("packed_weight"), (packed_intermediate, hidden),
    )
    down_weight = _meta(
        graph.placeholder("down_weight"), (hidden, intermediate),
    )
    packed_weight_t = _meta(
        graph.call_function(
            torch.ops.aten.permute.default,
            args=(packed_weight, [1, 0]),
        ),
        (hidden, packed_intermediate),
        strides=(1, hidden),
    )
    projection = _meta(graph.call_function(
        torch.ops.aten.mm.default, args=(lhs, packed_weight_t),
    ), (m, packed_intermediate))
    gate = _meta(graph.call_function(
        torch.ops.aten.slice.Tensor, args=(projection, 1, 0, intermediate),
    ), (m, intermediate), strides=(packed_intermediate, 1))
    up = _meta(graph.call_function(
        torch.ops.aten.slice.Tensor,
        args=(
            projection, 1, intermediate, 9_223_372_036_854_775_807,
        ),
    ), (m, intermediate), strides=(packed_intermediate, 1))
    gate_f32 = _meta(graph.call_function(
        torch.ops.prims.convert_element_type.default,
        args=(gate, torch.float32),
    ), (m, intermediate), dtype=torch.float32)
    neg = _meta(graph.call_function(
        torch.ops.aten.neg.default, args=(gate_f32,),
    ), (m, intermediate), dtype=torch.float32)
    exp = _meta(graph.call_function(
        torch.ops.aten.exp.default, args=(neg,),
    ), (m, intermediate), dtype=torch.float32)
    denominator = _meta(graph.call_function(
        torch.ops.aten.add.Tensor, args=(exp, 1),
    ), (m, intermediate), dtype=torch.float32)
    div = _meta(graph.call_function(
        torch.ops.aten.div.Tensor, args=(gate_f32, denominator),
    ), (m, intermediate), dtype=torch.float32)
    activation = _meta(graph.call_function(
        torch.ops.prims.convert_element_type.default,
        args=(div, torch.bfloat16),
    ), (m, intermediate))
    product = _meta(graph.call_function(
        torch.ops.aten.mul.Tensor, args=(activation, up),
    ), (m, intermediate))
    down_weight_t = _meta(graph.call_function(
        torch.ops.aten.permute.default,
        args=(down_weight, [1, 0]),
    ), (intermediate, hidden), strides=(1, intermediate))
    output = _meta(graph.call_function(
        torch.ops.aten.mm.default, args=(product, down_weight_t),
    ), (m, hidden))
    graph.output(output)
    return FxRegionMatch(output, (lhs, packed_weight, down_weight))


def test_swiglu_fx_region_imports_to_high_level_gpu_ir():
    match = _swiglu_region()
    module = fx_region_to_module(match, name="swiglu")

    verify_gpu_dialect(module)
    [kernel] = module.body.ops
    assert isinstance(kernel, KernelOp)
    assert [value.name for value in kernel.body.arguments] == [
        "lhs", "packed_weight", "down_weight", match.output.name,
    ]
    matmuls = find_ops_of_type(module, MatmulOp)
    assert len(matmuls) == 2
    assert matmuls[0].rhs.name == "packed_weight"
    assert matmuls[1].rhs.name == "down_weight"
    projection_slices = find_ops_of_type(module, TileOp)
    assert len(projection_slices) == 2
    assert [tile.indices[1] for tile in projection_slices] == [0, 1]
    assert [
        op.fn for op in find_ops_of_type(module, ElementwiseOp)
    ] == ["convert", "silu", "mul"]


def test_discovers_complete_swiglu_region_and_ordered_inputs():
    expected = _swiglu_region()
    # FX/AOT is free to declare placeholders in an order unrelated to use.
    inputs = list(expected.inputs)
    inputs[0].prepend(inputs[1])

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
    match = _swiglu_region()
    projection = [
        node for node in match.output.graph.nodes
        if node.target is torch.ops.aten.mm.default
    ][0]
    match.output.graph.call_function(torch.ops.aten.neg.default, (projection,))

    assert discover_supported_fx_region(match.output) is None

    with pytest.raises(ValueError, match="region value has external users"):
        fx_region_to_module(match, name="swiglu")


@pytest.mark.parametrize("split_kind", ["sizes", "uniform"])
def test_split_getitems_and_cat_import_as_aliases(split_kind):
    """Exercise the QKV/rotary split-and-concatenate structure at small sizes."""
    graph = Graph()
    packed = _meta(graph.placeholder("packed"), (2, 12))
    if split_kind == "sizes":
        split = graph.call_function(
            torch.ops.aten.split_with_sizes.default,
            args=(packed, [8, 2, 2], -1),
        )
        sizes = (8, 2, 2)
    else:
        split = graph.call_function(
            torch.ops.aten.split.Tensor, args=(packed, 4, -1),
        )
        sizes = (4, 4, 4)
    split.meta["val"] = tuple(
        torch.empty_strided(
            (2, size), (12, 1), dtype=torch.bfloat16, device="meta",
        )
        for size in sizes
    )

    pieces = []
    products = []
    start = 0
    for index, size in enumerate(sizes):
        piece = _meta(
            graph.call_function(operator.getitem, args=(split, index)),
            (2, size),
            strides=(12, 1),
        )
        product = _meta(
            graph.call_function(
                torch.ops.aten.mul.Tensor, args=(piece, piece),
            ),
            (2, size),
        )
        pieces.append((piece, start // size))
        products.append(product)
        start += size
    output = _meta(graph.call_function(
        torch.ops.aten.cat.default, args=(products, -1),
    ), (2, 12))
    graph.output(output)

    match = discover_supported_fx_region(output)
    assert match is not None
    assert match.inputs == (packed,)
    module = fx_region_to_module(match, name="split_cat")

    verify_gpu_dialect(module)
    [kernel] = module.body.ops
    assert isinstance(kernel, KernelOp)
    assert not find_ops_of_type(module, AllocOp)
    source_tiles = [
        tile for tile in find_ops_of_type(module, TileOp)
        if tile.arg is kernel.body.arguments[0]
    ]
    assert [tile.indices[-1] for tile in source_tiles] == [
        index for _piece, index in pieces
    ]
    elementwise = find_ops_of_type(module, ElementwiseOp)
    assert len(elementwise) == 3
    assert all(
        isinstance(op.dst.definition, TileOp)
        and op.dst.definition.arg is kernel.body.arguments[-1]
        for op in elementwise
    )


def test_singleton_view_ops_import_as_reshape_aliases():
    graph = Graph()
    source = _meta(
        graph.placeholder("source"), (1, 8), strides=(12, 1),
    )
    clone = _meta(graph.call_function(
        torch.ops.aten.clone.default,
        args=(source,),
        kwargs={"memory_format": torch.contiguous_format},
    ), (1, 8))
    viewed = _meta(graph.call_function(
        torch.ops.aten.view.default, args=(clone, [2, 4]),
    ), (2, 4))
    unsqueezed = _meta(
        graph.call_function(
            torch.ops.aten.unsqueeze.default, args=(viewed, -2),
        ),
        (2, 1, 4),
    )
    output = _meta(graph.call_function(
        torch.ops.aten.mul.Tensor, args=(unsqueezed, unsqueezed),
    ), (2, 1, 4))
    graph.output(output)

    match = discover_supported_fx_region(output)
    assert match is not None
    module = fx_region_to_module(match, name="views")

    verify_gpu_dialect(module)
    assert len(find_ops_of_type(module, ReshapeOp)) == 3


def test_clone_that_changes_layout_is_not_erased():
    graph = Graph()
    source = _meta(
        graph.placeholder("source"), (2, 8), strides=(12, 1),
    )
    clone = _meta(graph.call_function(
        torch.ops.aten.clone.default,
        args=(source,),
        kwargs={"memory_format": torch.contiguous_format},
    ), (2, 8))
    output = _meta(graph.call_function(
        torch.ops.aten.mul.Tensor, args=(clone, clone),
    ), (2, 8))
    graph.output(output)

    match = discover_supported_fx_region(output)
    assert match is not None
    with pytest.raises(ValueError, match="clone changes layout"):
        fx_region_to_module(match, name="copying_clone")


def _gemm_region() -> FxRegionMatch:
    graph = Graph()
    lhs = _meta(graph.placeholder("lhs"), (128, 64))
    weight = _meta(graph.placeholder("weight"), (64, 64))
    weight_t = _meta(graph.call_function(
        torch.ops.aten.permute.default, args=(weight, [1, 0]),
    ), (64, 64), strides=(1, 64))
    output = _meta(graph.call_function(
        torch.ops.aten.mm.default, args=(lhs, weight_t),
    ), (128, 64))
    graph.output(output)
    return FxRegionMatch(output, (lhs, weight))


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


def test_ptx_route_plan_binds_runtime_and_scratch_arguments(monkeypatch):
    launched = []
    scratch = object()
    zeroed_scratch = object()
    signature = [
        GmemTensorType.build([8], BF16Type()),
        GmemTensorType.build([2], F32Type()),
        GmemTensorType.build([1], F32Type()),
        GmemTensorType.build([8], BF16Type()),
    ]
    kernel = PtxKernel(
        ptx="", name="scratch_test", signature=signature,
        grid_dim=(1, 1, 1), block_dim=(1, 1, 1),
    )
    plan = PtxRoutePlan(
        kernel,
        bindings=[0, ScratchInit.EMPTY, ScratchInit.ZERO, 1],
    )
    monkeypatch.setattr(
        jit_compiler, "make_ptx_launcher",
        lambda _kernel: lambda *args: launched.append(args),
    )
    monkeypatch.setattr(torch, "empty", lambda *_args, **_kwargs: scratch)
    monkeypatch.setattr(
        torch, "zeros", lambda *_args, **_kwargs: zeroed_scratch,
    )

    route = plan()
    route("input", "output")  # type: ignore[arg-type]

    assert launched == [("input", scratch, zeroed_scratch, "output")]
