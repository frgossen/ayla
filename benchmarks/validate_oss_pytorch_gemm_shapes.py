#!/usr/bin/env python3
"""Classify every pooled OSS GEMM shape without compiling or using a GPU."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from benchmarks.oss_gemm_corpus import (
    CORPUS,
    VALIDATION_CSV,
    VALIDATION_JSON,
    benchmark_shapes,
    load_shapes,
    validate_shapes,
    validation_row,
    write_json,
    write_rows_csv,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--csv", type=Path, default=VALIDATION_CSV)
    parser.add_argument("--json", type=Path, default=VALIDATION_JSON)
    args = parser.parse_args()

    validations = validate_shapes(load_shapes(args.corpus))
    rows = [validation_row(validation) for validation in validations]
    write_rows_csv(rows, args.csv)
    write_json(rows, args.json)

    supported = [item for item in validations if item.supported]
    workloads = benchmark_shapes(validations)
    reasons = Counter(
        code for item in validations for code in item.reason_codes
    )
    total_calls = sum(item.shape.count for item in validations)
    supported_calls = sum(item.shape.count for item in supported)
    print(
        f"supported {len(supported)}/{len(validations)} pooled signatures, "
        f"covering {supported_calls}/{total_calls} recorded calls; "
        f"{len(workloads)} canonical benchmark shapes"
    )
    print("skipped by reason: " + ", ".join(
        f"{code}={count}" for code, count in sorted(reasons.items())
    ))
    print(f"CSV:  {args.csv.resolve()}")
    print(f"JSON: {args.json.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
