"""GPU architecture configuration for PTX generation.

Targets are deliberately small and explicit: the PTX ISA version and target
directive travel together, while lowering decisions use the architecture
family rather than scattered string comparisons.
"""
from __future__ import annotations

from dataclasses import dataclass
import re


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

    def supports_compute_capability(self, capability: tuple[int, int]) -> bool:
        """Whether PTX for this target may be loaded on ``capability``.

        NVIDIA's architecture-specific ``a`` targets expose instructions that
        are only guaranteed within that exact compute capability. Plain
        ``sm_XX`` targets retain PTX's forward-compatibility behavior.
        """
        match = re.fullmatch(r"sm_(\d+)([a-z]?)", self.architecture)
        if match is None:
            return False
        encoded = int(match.group(1))
        required = (encoded // 10, encoded % 10)
        return capability == required if match.group(2) else capability >= required


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
