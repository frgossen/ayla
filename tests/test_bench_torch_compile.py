"""Reporting used by the captured-block benchmark."""

import pytest
import torch

from benchmarks.bench_torch_compile.bench_torch_compile import (
    _comparison_regions,
    capture_aten_graph,
    format_captured_graph,
    profiled_launches,
    selected_ayla_kernels,
)
from benchmarks.bench_torch_compile.corpus import build_configs


def test_captured_graph_marks_selected_nodes():
    def block(lhs, rhs):
        return torch.sigmoid(lhs @ rhs.t())

    graph = capture_aten_graph(
        block,
        [torch.randn(2, 3), torch.randn(4, 3)],
    )

    rendered = format_captured_graph(graph, {"mm", "sigmoid"})

    assert "    %t :" in rendered
    assert ">>> %mm :" in rendered
    assert ">>> %sigmoid :" in rendered


def test_captured_graph_marks_the_complete_silu_dataflow():
    def block(lhs, rhs):
        result = lhs @ rhs.t()
        return result * torch.sigmoid(result)

    graph = capture_aten_graph(
        block,
        [torch.randn(2, 3), torch.randn(4, 3)],
    )

    rendered = format_captured_graph(graph, {"mm", "sigmoid", "mul"})

    assert "    %t :" in rendered
    assert ">>> %mm :" in rendered
    assert ">>> %sigmoid :" in rendered
    assert ">>> %mul :" in rendered


def test_selected_ayla_kernels_are_read_from_generated_code():
    codes = [
        "# ayla nodes: mm,sigmoid,mul\n"
        "torch.ops.ayla.first.default(lhs, rhs)\n"
        "# ayla nodes: mm_1\n"
        "torch.ops.ayla.second.default(value)\n",
    ]

    assert selected_ayla_kernels(codes) == (
        (("mm", "sigmoid", "mul"), "first"),
        (("mm_1",), "second"),
    )


def test_profiled_launches_maps_wrapper_calls_to_source_nodes():
    codes = ["""
        # Topologically Sorted Source Nodes: [permute, mm], Original ATen: [aten.mm]
        extern_kernels.mm(lhs, rhs, out=buf0)
        # Topologically Sorted Source Nodes: [sigmoid, mul], Original ATen: [aten.sigmoid, aten.mul]
        triton_poi_fused_mul_sigmoid_0.run(buf0)
        # Topologically Sorted Source Nodes: [sigmoid, mul, mm_1], Original ATen: [aten.mm]
        extern_kernels.mm(buf0, rhs, out=buf1)
        # ayla nodes: mm_2
        torch.ops.ayla.dynamic_m_matmul_out.default(lhs, rhs, buf2)
    """]

    assert profiled_launches(codes) == (
        ("aten::mm", frozenset({"permute", "mm"})),
        ("triton_poi_fused_mul_sigmoid_0", frozenset({"sigmoid", "mul"})),
        ("aten::mm", frozenset({"mm_1"})),
        ("ayla::dynamic_m_matmul_out", frozenset({"mm_2"})),
    )


def test_comparison_regions_expand_to_complete_launch_boundaries():
    stock = (("stock_fused", frozenset({"mm", "sigmoid", "mul"})),)
    ayla = (
        ("ayla_mm_sigmoid", frozenset({"mm", "sigmoid"})),
        ("ayla_mul", frozenset({"mul"})),
    )

    regions = _comparison_regions(
        (stock, ayla), (("mm", "sigmoid"),),
        ("mm", "sigmoid", "mul"),
    )

    assert len(regions) == 1
    assert regions[0].requested_nodes == ("mm", "sigmoid")
    assert regions[0].compared_nodes == ("mm", "sigmoid", "mul")
    assert regions[0].launch_indices == ((0,), (0, 1))


def test_comparison_regions_reject_ambiguous_launch_ownership():
    duplicated = (
        ("first", frozenset({"mm"})),
        ("second", frozenset({"mm"})),
    )
    one_launch = (("ayla", frozenset({"mm"})),)

    with pytest.raises(AssertionError, match="ambiguous=\\['mm'\\]"):
        _comparison_regions(
            (duplicated, one_launch), (("mm",),), ("mm",),
        )


def test_muse_capture_uses_one_concrete_input_shape():
    configs = build_configs(muse_num_tokens=17)

    assert [config.name for config in configs] == [
        "muse_glimmer_block M=17 bf16",
    ]
