"""Direct scheduled GPU IR lowering: machine instructions -> SASS -> cubin.

The experimental backend owns TMA pipelining and physical registers; it never
invokes PTX template lowering or ptxas. The assembler/ISA database is provided
by nvidia-sass-document (see docs/sass_backend.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache
from collections import OrderedDict
import os
import re
from pathlib import Path
import threading

from ..gpu_dialect import (
    KernelOp,
    ModuleOp,
)
from ..ptx_runtime import CudaLauncher

_ASSEMBLER_LOCK = threading.RLock()
_ENCODINGS: OrderedDict[str, tuple[int, int]] = OrderedDict()


def _assemble_cached(assembler, source):
    """Reuse instruction encodings, resolving branches after final layout.

    This caches instructions, not shape-specialized kernels: new dimensions
    still generate a new instruction stream, parameter ABI, and cubin.
    """
    labels = {}
    lines = []
    for line in source.splitlines():
        if line.startswith("#def_label("):
            labels[line[len("#def_label(") : -1]] = len(lines) * 16
        else:
            lines.append(line)
    for i, line in enumerate(lines):
        lines[i] = re.sub(
            r"#label\(([^)]+)\)", lambda m: str(labels[m[1]] - (i + 1) * 16), line
        )
    missing = list(dict.fromkeys(line for line in lines if line not in _ENCODINGS))
    if missing:
        encoded = assembler.assemble_flat("\n".join(missing), arch="sm90")
        _ENCODINGS.update(zip(missing, encoded, strict=True))
    result = [_ENCODINGS[line] for line in lines]
    for line in lines:
        _ENCODINGS.move_to_end(line)
    while len(_ENCODINGS) > 8192:
        _ENCODINGS.popitem(last=False)
    return result


@cache
def _assembler():
    """Load the reference assembler without modifying the global search path."""
    root = Path(
        os.environ.get(
            "AYLA_SASS_DOCUMENT",
            Path(__file__).resolve().parents[2] / "nvidia-sass-document",
        )
    )
    import importlib.util
    import sys

    name = "_ayla_sass_assembler"
    spec = importlib.util.spec_from_file_location(
        name,
        root / "assembler" / "__init__.py",
        submodule_search_locations=[str(root / "assembler")],
    )
    if spec is None or spec.loader is None or not (root / "sm90.json").exists():
        raise RuntimeError(
            "SASS backend requires nvidia-sass-document/assembler and sm90.json; "
            "run python nvidia-sass-document/tools/parse_sm90.py or set AYLA_SASS_DOCUMENT"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class SassKernel:
    source: str
    image: bytes
    name: str
    signature: list
    grid_dim: tuple[int, int, int]
    block_dim: tuple[int, int, int]
    shared_mem_bytes: int
    num_mbarriers: int
    register_counts: dict[str, int]
    lowered_ir: str
    optimization_stats: dict[str, int] = field(default_factory=dict)

    @property
    def dynamic_shared_mem_bytes(self) -> int:
        # Native shared storage is declared in the cubin.
        return 0


class LowerToSass:
    """Lower the actual scheduled operation graph into executable SASS."""

    def __init__(
        self,
        *,
        max_ctas_per_sm: int | None = None,
    ):
        if max_ctas_per_sm is not None and not 2 <= max_ctas_per_sm <= 16:
            raise ValueError(
                "Hopper register reservation supports CTA caps from 2 to 16"
            )
        self.max_ctas_per_sm = max_ctas_per_sm

    def run(self, module: ModuleOp) -> SassKernel:
        from ..base.ir import clone_op
        from ..sass_codegen import SassLoweringPipeline, SassCodegen

        if len(module.body.ops) != 1 or not isinstance(module.body.ops[0], KernelOp):
            raise ValueError("SASS lowering requires exactly one kernel")
        if module.target not in ("sm_90", "sm_90a"):
            raise ValueError("SASS machine lowering currently targets Hopper sm_90")
        scheduled = clone_op(module, {})
        lowered = SassLoweringPipeline().run(scheduled)
        if len(lowered.body.ops) != 1 or not isinstance(lowered.body.ops[0], KernelOp):
            raise ValueError("SASS lowering requires exactly one kernel")
        kernel = lowered.body.ops[0]
        from ..sass_codegen import relax_prefetched_matrix_waits

        relaxed = relax_prefetched_matrix_waits(kernel)
        codegen = SassCodegen(kernel)
        source, params, counts = codegen.run()
        codegen.optimization_stats = getattr(codegen, "optimization_stats", {})
        codegen.optimization_stats["matrix_waits_relaxed"] = relaxed
        with _ASSEMBLER_LOCK, _assembler()._using_arch("sm90"):
            assembler = _assembler()
            encoded = _assemble_cached(assembler, source)
            cb = assembler.CubinBuilder()
            cb.set_code(encoded, kernel_name=kernel.name)
            cb.set_params(params)
            votes = [
                hex(i * 16)
                for i, line in enumerate(
                    line for line in source.splitlines() if not line.startswith("#")
                )
                if "VOTEU" in line
            ]
            cb.set_pragma("INT_WARP_WIDE_OFFSETS", ",".join(votes))
            regcount = max(8, (counts["R"] + 2 + 7) // 8 * 8)
            if self.max_ctas_per_sm is not None:
                # Hopper has 65536 GPRs/SM and allocates in eight-register
                # units per thread. Reserve enough to exclude cap+1 CTAs;
                # the instructions and their physical registers stay intact.
                threads = 128  # Enforced by SassCodegen.init.
                floor = 65536 // (threads * (self.max_ctas_per_sm + 1)) + 1
                regcount = max(regcount, (floor + 7) // 8 * 8)
            if regcount > 248:
                raise ValueError("SASS register pressure requires spilling")
            cb.set_pragma("REGCOUNT", str(regcount))
            cb.set_regcount(regcount)
            cb.set_shared_mem(codegen.shared_bytes)
            cb.set_pragma("NUM_MBARRIERS", str(len(codegen.phases)))
            image = cb.build()
        return SassKernel(
            source=source,
            image=image,
            name=f"_Z{len(kernel.name)}{kernel.name}",
            signature=list(kernel.signature),
            grid_dim=tuple(d or 1 for d in kernel.grid_dim),
            block_dim=tuple(d or 1 for d in kernel.block_dim),
            shared_mem_bytes=codegen.shared_bytes,
            num_mbarriers=len(codegen.phases),
            register_counts=counts,
            lowered_ir=str(lowered),
            optimization_stats=codegen.optimization_stats,
        )


def lower_to_sass(
    module: ModuleOp,
    *,
    max_ctas_per_sm: int | None = None,
) -> SassKernel:
    """Lower a scheduled module to Hopper SASS without changing the caller IR."""
    return LowerToSass(max_ctas_per_sm=max_ctas_per_sm).run(module)


class SassLauncher(CudaLauncher):
    """Reuse the tensor-map/stream ABI, requiring the cubin's Hopper target."""

    def ensure_loaded(self) -> None:
        if self._module is None:
            import torch

            if torch.cuda.get_device_capability() != (9, 0):
                raise RuntimeError("SASS backend currently targets Hopper sm_90 only")
        super().ensure_loaded()


def make_sass_launcher(kernel: SassKernel) -> SassLauncher:
    return SassLauncher(kernel, kernel.image)
