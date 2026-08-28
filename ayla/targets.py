"""GPU architecture configuration for PTX generation.

Targets are deliberately small and explicit: the PTX ISA version and target
directive travel together, while lowering decisions use the architecture
family rather than scattered string comparisons.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PtxTarget:
    """A PTX compilation target and its architecture-specific capabilities."""

    name: str
    ptx_version: str
    architecture: str
    compute_capability: tuple[int, int]
    tensor_core_kind: str
    address_size: int = 64

    def module_attrs(self) -> dict[str, object]:
        return {
            "version": self.ptx_version,
            "target": self.architecture,
            "address_size": self.address_size,
        }


HOPPER = PtxTarget(
    name="hopper",
    ptx_version="9.0",
    architecture="sm_90a",
    compute_capability=(9, 0),
    tensor_core_kind="wgmma",
)

BLACKWELL = PtxTarget(
    name="blackwell",
    ptx_version="9.0",
    architecture="sm_100a",
    compute_capability=(10, 0),
    tensor_core_kind="tcgen05",
)


_KNOWN_TARGETS = {
    (target.ptx_version, target.architecture, target.address_size): target
    for target in (HOPPER, BLACKWELL)
}


def target_from_attrs(
    ptx_version: str,
    architecture: str,
    address_size: int,
) -> PtxTarget:
    """Recover a known target, or preserve a generic scalar PTX target.

    Generic targets remain useful for simple PTX and legacy tests. They have no
    architecture-specific tensor-core lowering, so attempting to lower such an
    operation fails explicitly.
    """
    key = (ptx_version, architecture, address_size)
    if key in _KNOWN_TARGETS:
        return _KNOWN_TARGETS[key]
    return PtxTarget(
        name=architecture,
        ptx_version=ptx_version,
        architecture=architecture,
        compute_capability=(-1, -1),
        tensor_core_kind="none",
        address_size=address_size,
    )
