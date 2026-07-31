"""Tests for SSA-based cluster TMA multicast inference."""
import textwrap

import pytest

from ayla.base.ir import find_ops
from ayla.gpu_dialect import (
    AsyncOp,
    MoveOp,
    gpu_dialect,
    multicast_count_for_lock,
)
from ayla.passes.fuse_async import fuse_async
from ayla.passes.infer_tma_multicast import infer_tma_multicast
from ayla.passes.lower_to_async import lower_to_async


@pytest.mark.parametrize(
    "cluster_dim,expected_label",
    [((2, 1, 1), "b"), ((1, 2, 1), "a")],
)
def test_infers_only_cluster_invariant_tma_tile(cluster_dim, expected_label):
    module = gpu_dialect.parse(textwrap.dedent(f"""\
        module {{
          kernel @k(%a: gmem_tensor<128x128xbf16>, %b: gmem_tensor<128x128xbf16>) grid_dim=[2, 2, 1] block_dim=[128, 1, 1] cluster_dim={list(cluster_dim)} {{
            %bx: index, %by: index, %bz: index = block_ids()
            %at: gmem_tensor<64x128xbf16> = tile(%a: gmem_tensor<128x128xbf16>, %bx: index, 0)
            %bt: gmem_tensor<64x128xbf16> = tile(%b: gmem_tensor<128x128xbf16>, %by: index, 0)
            %sa: smem_tensor<64x128xbf16> = alloc()
            move(%sa: smem_tensor<64x128xbf16>, %at: gmem_tensor<64x128xbf16>) attrs={{label: "a", tma: True}}
            %sb: smem_tensor<64x128xbf16> = alloc()
            move(%sb: smem_tensor<64x128xbf16>, %bt: gmem_tensor<64x128xbf16>) attrs={{label: "b", tma: True}}
          }}
        }}"""))
    module = fuse_async(lower_to_async(module))
    infer_tma_multicast(module)

    [async_op] = find_ops(module, lambda op: isinstance(op, AsyncOp))
    moves = find_ops(async_op, lambda op: isinstance(op, MoveOp))
    marked = [m.attrs["label"] for m in moves if "multicast_count" in m.attrs]
    assert marked == [expected_label]
    assert async_op.lock.type.attrs == {"kind": "mbar"}
    assert multicast_count_for_lock(async_op.lock) == 2


def test_does_not_infer_partial_subcluster_multicast():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @k(%a: gmem_tensor<128x128xbf16>) grid_dim=[2, 2, 1] block_dim=[128, 1, 1] cluster_dim=[2, 2, 1] {
            %bx: index, %by: index, %bz: index = block_ids()
            %at: gmem_tensor<64x64xbf16 strides=[128, 1]> = tile(%a: gmem_tensor<128x128xbf16>, %bx: index, %by: index)
            %sa: smem_tensor<64x64xbf16> = alloc()
            move(%sa: smem_tensor<64x64xbf16>, %at: gmem_tensor<64x64xbf16 strides=[128, 1]>) attrs={tma: True}
          }
        }"""))
    infer_tma_multicast(module)
    [move] = find_ops(module, lambda op: isinstance(op, MoveOp))
    assert "multicast_count" not in move.attrs
