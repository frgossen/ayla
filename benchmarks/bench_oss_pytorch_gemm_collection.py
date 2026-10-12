"""Benchmark GEMMs over the shapes real models actually run.

``benchmarks/oss_pytorch_gemm_collection.json`` holds the mm/addmm/bmm shapes
PyTorch recorded from the HuggingFace, timm and TorchBench models (see
``scripts/collect_oss_pytorch_gemm_shapes.py``). This benchmarks kernels over
those shapes instead of the single hand-picked 4096 cube in
``bench_gemm_collection.py``, weighted by how often each shape actually occurs.

Only the shapes ayla has a kernel for are benchmarked, asked for via
``ayla.gemm_collection.query``, each against a torch baseline in the same
operand layout. Kernels are compiled
per shape and none in the collection today matches a recorded shape -- they are
all 4096 cubes, which no model runs -- so at the moment this benchmarks nothing
and says so. Nothing here needs changing as kernels for real shapes land; they
simply start showing up in the table.
"""

from __future__ import annotations

import collections
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from ayla import gemm_collection

from benchmarks.common import gemm_bench_config, print_results, run, torch_mm_nt


CORPUS = Path(__file__).resolve().parent / "oss_pytorch_gemm_collection.json"

# The corpus records dtypes as torch spells them, minus the prefix.
DTYPES = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


@dataclass
class Shape:
    """One GEMM shape, pooled across every suite that recorded it."""

    m: int
    k: int
    n: int
    dtype: torch.dtype
    count: int  # how many times the models ran it


def ayla_candidates(shape: Shape) -> list[tuple[str, Callable]]:
    """(label, op) for every ayla kernel built for exactly this GEMM.

    Empty for a shape the collection has no kernel for, which is what decides
    whether the shape is benchmarked at all: there is nothing to learn from
    timing torch against itself.
    """
    return [
        (kernel.label, kernel.op)
        for kernel in gemm_collection.query(
            m=shape.m, k=shape.k, n=shape.n, dtype=shape.dtype
        )
    ]


def load_shapes() -> list[Shape]:
    """Read the corpus and pool identical shapes across suites.

    Only mm: the shared timing loop calls op(a, b), so addmm's bias and bmm's
    batch dimension have nowhere to go yet.
    """
    op = "mm"

    # A shape is the same workload whichever suite recorded it, so pool the
    # counts: "most common" means most common across all the models.
    pooled = collections.Counter()
    for entry in json.loads(CORPUS.read_text()):
        if entry["op"] == op:
            pooled[entry["m"], entry["k"], entry["n"], entry["args"][0]["dtype"]] += (
                entry["count"]
            )

    shapes = [
        Shape(m=m, k=k, n=n, dtype=DTYPES[dtype], count=count)
        for (m, k, n, dtype), count in pooled.items()
        if dtype in DTYPES
    ]
    shapes.sort(key=lambda s: (-s.count, s.m, s.k, s.n))
    return shapes


def main() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    shapes = load_shapes()
    total_calls = sum(shape.count for shape in shapes)

    # Only shapes the collection actually has a kernel for: timing the rest
    # would just be torch against itself.
    selected = [
        (shape, candidates)
        for shape in shapes
        if (candidates := ayla_candidates(shape))
    ]
    covered = sum(shape.count for shape, _ in selected)
    print(
        f"{len(selected)}/{len(shapes)} recorded mm shapes have an ayla kernel "
        f"({covered}/{total_calls} calls, {covered / total_calls * 100:.1f}%)",
        file=sys.stderr,
    )

    if not selected:
        return

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    configs = [
        gemm_bench_config(
            name=f"{label} x{shape.count}",
            m=shape.m,
            n=shape.n,
            k=shape.k,
            compute=op,
            dtype=shape.dtype,
        )
        for shape, candidates in selected
        for label, op in [("torch.mm", torch_mm_nt), *candidates]
    ]

    print_results(run(configs))


if __name__ == "__main__":
    main()
