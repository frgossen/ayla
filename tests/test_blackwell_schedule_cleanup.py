"""Production routes keep backend dispatch after default compaction."""

import pytest

from ayla.gemm_collection.blackwell_router import make_blackwell_router_launcher


@pytest.mark.parametrize(
    "shape,workspace",
    [((1536, 512, 16448), True), ((401536, 64, 192), False)],
)
def test_compact_production_routes_build_their_measured_backend(shape, workspace):
    launch = make_blackwell_router_launcher(m=shape[0], n=shape[1], k=shape[2])
    assert (launch.reduction_kernel is not None) == workspace
    assert (launch.workspace_shape != (0,)) == workspace
