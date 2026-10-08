#!/usr/bin/env python3
"""Profile one measured Torch or Ayla GEMM inside a CUDA profiler range."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any, Callable

import torch

from benchmarks.common import torch_addmm_nt, torch_bmm_nt, torch_mm_nt
from benchmarks.oss_gemm_corpus import (
    RAW_JSON,
    ScheduleConfig,
    Shape,
    TensorSpec,
)


def _measured_config(
    rows: list[dict[str, Any]], dimensions: tuple[int, int, int], name: str,
    *, workload: str = "mm", batch: int | None = None,
) -> ScheduleConfig:
    matches = [
        row for row in rows
        if (int(row["m"]), int(row["n"]), int(row["k"])) == dimensions
        and row.get("op", "mm") == workload
        and row.get("batch") == batch
        and row.get("implementation") == "ayla"
        and row.get("status") == "ok"
        and row.get("configuration", {}).get("name") == name
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one successful {dimensions} / {name!r} row; "
            f"found {len(matches)}"
        )
    config = matches[0]["configuration"]
    values: dict[str, object] = {}
    for field in dataclasses.fields(ScheduleConfig):
        if field.name in config:
            values[field.name] = config[field.name]
        elif field.default is not dataclasses.MISSING:
            values[field.name] = field.default
        else:
            raise ValueError(f"configuration is missing {field.name!r}")
    return ScheduleConfig(**values)  # type: ignore[arg-type]


def _profile_shape(
    dimensions: tuple[int, int, int], workload: str = "mm",
    batch: int | None = None,
) -> Shape:
    """Build the canonical contiguous-NT shape used by the benchmark path."""
    m, n, k = dimensions
    operand_batch = (batch,) if workload == "bmm" else ()
    matrix_args = (
        TensorSpec(
            shape=operand_batch + (m, k),
            stride=((m * k, k, 1) if batch is not None else (k, 1)),
            dtype="float16",
        ),
        TensorSpec(
            shape=operand_batch + (n, k),
            stride=((n * k, k, 1) if batch is not None else (k, 1)),
            dtype="float16",
        ),
    )
    return Shape(
        op=workload, batch=batch, m=m, n=n, k=k,
        args=(
            (TensorSpec(shape=(n,), stride=(1,), dtype="float16"),)
            if workload == "addmm" else ()
        ) + matrix_args,
        count=1, suites=("profile",), source_records=1,
    )


def _make_profile_ayla_compute(
    dimensions: tuple[int, int, int], schedule: ScheduleConfig, workload: str,
    batch: int | None = None,
) -> tuple[Callable[..., torch.Tensor], list[torch.Tensor]]:
    """Use the benchmark's exact builder path, including fixed grids/clusters."""
    from benchmarks.benchmark_oss_pytorch_gemm_shapes import _make_ayla_config

    bench, _, _ = _make_ayla_config(
        _profile_shape(dimensions, workload, batch), schedule,
    )
    assert bench.make_inputs is not None
    return bench.compute, bench.make_inputs()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", choices=("torch", "ayla"), required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--configuration")
    parser.add_argument("--results-json", type=Path, default=RAW_JSON)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument(
        "--workload", choices=("mm", "addmm", "bmm"), default="mm",
    )
    parser.add_argument("--batch", type=int)
    args = parser.parse_args()
    dimensions = (args.m, args.n, args.k)
    dtype = torch.float16
    if args.workload == "bmm" and args.batch is None:
        parser.error("--workload bmm requires --batch")
    if args.workload != "bmm" and args.batch is not None:
        parser.error("--batch applies only to --workload bmm")
    operand_batch = (args.batch,) if args.batch is not None else ()
    a = torch.randn(operand_batch + (args.m, args.k), dtype=dtype, device="cuda")
    bt = torch.randn(operand_batch + (args.n, args.k), dtype=dtype, device="cuda")
    inputs = [a, bt]

    compute: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    if args.implementation == "torch":
        if args.configuration is not None:
            parser.error("--configuration applies only to --implementation ayla")
        if args.workload == "bmm":
            compute = torch_bmm_nt
        elif args.workload == "addmm":
            compute = torch_addmm_nt
            inputs = [
                torch.randn(args.n, dtype=dtype, device="cuda"), a, bt,
            ]
        else:
            compute = torch_mm_nt
        print(
            f"torch.{args.workload} NT {args.m}x{args.n}x{args.k}",
            flush=True,
        )
    else:
        if args.configuration is None:
            parser.error("--implementation ayla requires --configuration")
        rows = json.loads(args.results_json.read_text())
        schedule = _measured_config(
            rows, dimensions, args.configuration,
            workload=args.workload, batch=args.batch,
        )
        print(
            f"Ayla {args.m}x{args.n}x{args.k}; "
            f"config={dataclasses.asdict(schedule)}",
            flush=True,
        )
        compute, inputs = _make_profile_ayla_compute(
            dimensions, schedule, args.workload, args.batch,
        )

    for _ in range(args.warmup):
        compute(*inputs)
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    compute(*inputs)
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
