#!/usr/bin/env python3
"""Select OSS GEMM changes that survive replicated paired measurements."""

from __future__ import annotations

import argparse
import copy
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from benchmarks.oss_gemm_corpus import (
    CORPUS,
    benchmark_shapes,
    load_shapes,
    require_same_timing,
    validate_shapes,
    write_json,
)


def _config_key(row: dict[str, Any]) -> str:
    config = {
        key: value
        for key, value in row["configuration"].items()
        if key != "name"
    }
    # Historical rows predate the native N-output grouping search control.
    config.setdefault("output_n_tiles_per_cta", 1)
    return json.dumps(config, sort_keys=True, separators=(",", ":"))


def _has_distinct_configuration(
    original: dict[str, Any], candidate: dict[str, Any],
) -> bool:
    return _config_key(original) != _config_key(candidate)


def _implementation_rows(
    rows: list[dict[str, Any]], implementation: str,
) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("implementation") != implementation:
            continue
        if row.get("status") != "ok" or row.get("latency_ms") is None:
            continue
        shape_id = str(row["shape_id"])
        if shape_id in selected:
            raise ValueError(
                f"multiple {implementation} rows for {shape_id} in one input"
            )
        selected[shape_id] = row
    return selected


def _read(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text())
    if not isinstance(value, list):
        raise ValueError(f"{path}: expected a JSON list")
    return value


def _representative_incumbent(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return the paired incumbent run closest to its replicated median."""
    median_pct = statistics.median(
        float(row["original"]["torch_pct"]) for row in rows
    )
    return min(
        rows,
        key=lambda row: abs(float(row["original"]["torch_pct"]) - median_pct),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument(
        "--comparison", action="append", nargs=2, default=[],
        metavar=("ORIGINAL_JSON", "CANDIDATE_JSON"),
        help="add one same-GPU replicated original/candidate result pair",
    )
    parser.add_argument(
        "--incumbent", action="append", type=Path, default=[],
        help="add one paired Torch/Ayla incumbent-only replicated result",
    )
    parser.add_argument("--minimum-speedup", type=float, default=1.01)
    parser.add_argument("--minimum-wins", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    if not args.comparison and not args.incumbent:
        parser.error("at least one --comparison or --incumbent is required")
    if args.minimum_speedup <= 1:
        parser.error("--minimum-speedup must exceed one")
    if args.minimum_wins < 1:
        parser.error("--minimum-wins must be positive")
    if args.comparison and args.minimum_wins > len(args.comparison):
        parser.error("--minimum-wins exceeds the number of comparisons")

    supported_shape_ids = {
        shape.shape_id
        for shape in benchmark_shapes(validate_shapes(load_shapes(args.corpus)))
    }
    baseline_rows = [
        row for row in _read(args.baseline)
        if str(row.get("shape_id", "")) in supported_shape_ids
    ]
    baseline_ayla = _implementation_rows(baseline_rows, "ayla")
    measurements: dict[str, list[dict[str, Any]]] = defaultdict(list)
    incumbent_measurements: dict[str, list[dict[str, Any]]] = defaultdict(list)
    candidate_keys: dict[str, str] = {}

    for original_path_text, candidate_path_text in args.comparison:
        original_path = Path(original_path_text)
        candidate_path = Path(candidate_path_text)
        original_rows = _read(original_path)
        candidate_rows = _read(candidate_path)
        require_same_timing(
            (original_path, original_rows), (candidate_path, candidate_rows),
        )
        original_ayla = _implementation_rows(original_rows, "ayla")
        original_torch = _implementation_rows(original_rows, "torch")
        candidate_ayla = _implementation_rows(candidate_rows, "ayla")
        candidate_torch = _implementation_rows(candidate_rows, "torch")
        for shape_id in sorted(original_ayla.keys() & candidate_ayla.keys()):
            if (
                shape_id not in baseline_ayla
                or shape_id not in original_torch
                or shape_id not in candidate_torch
            ):
                continue
            original = original_ayla[shape_id]
            candidate = candidate_ayla[shape_id]
            if _config_key(original) != _config_key(baseline_ayla[shape_id]):
                raise ValueError(
                    f"{original_path}: original config for {shape_id} does "
                    "not match the baseline"
                )
            incumbent_measurements[shape_id].append({
                "original": original,
                "original_torch": original_torch[shape_id],
            })
            if candidate.get("configuration", {}).get("unsafe_reuse_smem"):
                raise ValueError(f"unsafe candidate for {shape_id}")
            candidate_key = _config_key(candidate)
            if not _has_distinct_configuration(original, candidate):
                continue
            prior_key = candidate_keys.setdefault(shape_id, candidate_key)
            if candidate_key != prior_key:
                raise ValueError(
                    f"replicates use different candidates for {shape_id}"
                )
            if original.get("torch_pct") is None or candidate.get("torch_pct") is None:
                raise ValueError(f"missing paired torch percentage for {shape_id}")
            measurements[shape_id].append({
                "original": original,
                "original_torch": original_torch[shape_id],
                "candidate": candidate,
                "candidate_torch": candidate_torch[shape_id],
                "pct_speedup": (
                    float(candidate["torch_pct"]) / float(original["torch_pct"])
                ),
                "latency_speedup": (
                    float(original["latency_ms"])
                    / float(candidate["latency_ms"])
                ),
            })

    for incumbent_path in args.incumbent:
        incumbent_rows = _read(incumbent_path)
        require_same_timing(
            (args.baseline, baseline_rows), (incumbent_path, incumbent_rows),
        )
        incumbent_ayla = _implementation_rows(incumbent_rows, "ayla")
        incumbent_torch = _implementation_rows(incumbent_rows, "torch")
        for shape_id in sorted(incumbent_ayla.keys() & incumbent_torch.keys()):
            if shape_id not in baseline_ayla:
                continue
            original = incumbent_ayla[shape_id]
            if _config_key(original) != _config_key(baseline_ayla[shape_id]):
                raise ValueError(
                    f"{incumbent_path}: incumbent config for {shape_id} does "
                    "not match the baseline"
                )
            if original.get("torch_pct") is None:
                raise ValueError(f"missing paired torch percentage for {shape_id}")
            incumbent_measurements[shape_id].append({
                "original": original,
                "original_torch": incumbent_torch[shape_id],
            })

    accepted: dict[str, dict[str, Any]] = {}
    summary_rows: list[dict[str, Any]] = []
    for shape_id, rows in sorted(measurements.items()):
        pct_speedups = [float(row["pct_speedup"]) for row in rows]
        latency_speedups = [float(row["latency_speedup"]) for row in rows]
        median_pct_speedup = statistics.median(pct_speedups)
        median_latency_speedup = statistics.median(latency_speedups)
        wins = sum(speedup > 1 for speedup in pct_speedups)
        threshold_wins = sum(
            speedup >= args.minimum_speedup for speedup in pct_speedups
        )
        is_accepted = (
            median_pct_speedup >= args.minimum_speedup
            and wins >= args.minimum_wins
            and median_latency_speedup > 1
        )
        representative = min(
            rows,
            key=lambda row: abs(
                float(row["pct_speedup"]) - median_pct_speedup
            ),
        )
        if is_accepted:
            accepted[shape_id] = representative
        original = representative["original"]
        candidate = representative["candidate"]
        summary_rows.append({
            "shape_id": shape_id,
            "frequency": candidate.get("frequency"),
            "accepted": is_accepted,
            "replicates": len(rows),
            "wins": wins,
            "threshold_wins": threshold_wins,
            "median_pct_speedup": median_pct_speedup,
            "median_latency_speedup": median_latency_speedup,
            "median_original_pct": statistics.median(
                float(row["original"]["torch_pct"]) for row in rows
            ),
            "median_candidate_pct": statistics.median(
                float(row["candidate"]["torch_pct"]) for row in rows
            ),
            "median_original_latency_ms": statistics.median(
                float(row["original"]["latency_ms"]) for row in rows
            ),
            "median_candidate_latency_ms": statistics.median(
                float(row["candidate"]["latency_ms"]) for row in rows
            ),
            "median_original_tflops": statistics.median(
                float(row["original"]["tflops"]) for row in rows
            ),
            "median_candidate_tflops": statistics.median(
                float(row["candidate"]["tflops"]) for row in rows
            ),
            "original_name": original["configuration"]["name"],
            "candidate_name": candidate["configuration"]["name"],
            "original_configuration": original["configuration"],
            "candidate_configuration": candidate["configuration"],
            "pct_speedups": pct_speedups,
            "latency_speedups": latency_speedups,
        })

    output_rows: list[dict[str, Any]] = []
    for baseline_row in baseline_rows:
        shape_id = str(baseline_row["shape_id"])
        selected = accepted.get(shape_id)
        refreshed = (
            _representative_incumbent(incumbent_measurements[shape_id])
            if shape_id in incumbent_measurements else None
        )
        if selected is None and refreshed is None:
            output_rows.append(copy.deepcopy(baseline_row))
            continue
        if baseline_row.get("implementation") == "torch":
            source = (
                selected["candidate_torch"]
                if selected is not None else refreshed["original_torch"]
            )
            output_rows.append(copy.deepcopy(source))
        elif baseline_row.get("implementation") == "ayla":
            row = copy.deepcopy(
                selected["candidate"]
                if selected is not None else refreshed["original"]
            )
            row["replicated_selection_speedup"] = (
                statistics.median(
                    item["pct_speedup"] for item in measurements[shape_id]
                )
                if selected is not None else 1.0
            )
            row["replicated_selection_count"] = (
                len(measurements[shape_id])
                if selected is not None
                else len(incumbent_measurements[shape_id])
            )
            output_rows.append(row)
        else:
            output_rows.append(copy.deepcopy(baseline_row))

    write_json(output_rows, args.output)
    write_json(summary_rows, args.summary)
    print(
        f"selected {len(accepted)} replicated changes across "
        f"{len(measurements)} compared shapes and refreshed "
        f"{len(incumbent_measurements)} incumbents; ranking JSON: "
        f"{args.output.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
