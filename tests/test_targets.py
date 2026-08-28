from ayla.targets import BLACKWELL, HOPPER, PtxTarget


def test_architecture_specific_targets_require_exact_compute_capability():
    assert HOPPER.supports_compute_capability((9, 0))
    assert not HOPPER.supports_compute_capability((10, 0))
    assert BLACKWELL.supports_compute_capability((10, 0))
    assert not BLACKWELL.supports_compute_capability((10, 1))


def test_plain_ptx_target_is_forward_compatible():
    target = PtxTarget(
        name="ampere",
        ptx_version="8.0",
        architecture="sm_80",
        compute_capability=(8, 0),
        tensor_core_kind="none",
    )

    assert target.supports_compute_capability((8, 0))
    assert target.supports_compute_capability((10, 0))
    assert not target.supports_compute_capability((7, 5))
