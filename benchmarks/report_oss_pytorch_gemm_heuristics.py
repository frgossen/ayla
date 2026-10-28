#!/usr/bin/env python3
"""Turn raw OSS GEMM measurements into a regime-level heuristic document."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from benchmarks.oss_gemm_corpus import (
    HEURISTICS_MD,
    IMPROVEMENTS_CSV,
    IMPROVEMENTS_JSON,
    RAW_JSON,
    SEEDS_JSON,
    SHAPE_IMPROVEMENTS_CSV,
    SHAPE_IMPROVEMENTS_JSON,
    write_json,
    write_rows_csv,
)


REGIME_FIELDS = ("dtype", "layout", "m_bucket", "aspect", "alignment")
DISPLAY_KNOBS = (
    "output_tiles_per_cta", "output_cta_count", "unroll_output_tiles",
    "output_tile_fast_axis",
    "output_m_tile", "output_n_tile",
    "warpgroup_m_tile", "warpgroup_n_tile", "warpgroup_fast_axis",
    "cluster_m_ctas", "cluster_n_ctas",
    "operand_stages", "operand_k_tile", "rhs_tma_load_slices", "output_slices",
    "output_store_buffers", "output_tma_inner_size", "output_store",
    "unsafe_reuse_smem",
    "lhs_l2_cache_hint", "rhs_l2_cache_hint", "dst_l2_cache_hint",
    "lhs_l2_promotion", "ptx_optimization_level", "multicast",
)


def _weighted_geomean(values: list[tuple[float, int]]) -> float:
    total_weight = sum(weight for _, weight in values)
    return math.exp(
        sum(math.log(value) * weight for value, weight in values) / total_weight
    )


def _config_key(row: dict[str, Any]) -> str:
    return json.dumps(row["configuration"], sort_keys=True, separators=(",", ":"))


def _paired_percentage(
    baseline: dict[str, Any], candidate: dict[str, Any],
) -> float:
    value = candidate.get("torch_pct")
    return (
        float(value) if value is not None
        else float(baseline["latency_ms"]) / float(candidate["latency_ms"]) * 100
    )


def _format_knobs(config: dict[str, Any]) -> str:
    return ", ".join(f"{key}={config.get(key)!r}" for key in DISPLAY_KNOBS)


def improvement_rows(rows: list[dict[str, Any]]) -> list[dict[str, object]]:
    """Aggregate each shape's fastest Ayla result against its Torch result."""
    torch_by_shape = {
        str(row["shape_id"]): row for row in rows
        if row.get("implementation") == "torch" and row.get("status") == "ok"
    }
    ayla_by_shape: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    failures_by_regime: Counter[str] = Counter()
    for row in rows:
        if row.get("implementation") != "ayla":
            continue
        if (
            row.get("status") == "ok"
            and row.get("latency_ms") is not None
            and not bool(
                row.get("configuration", {}).get("unsafe_reuse_smem")
            )
        ):
            ayla_by_shape[str(row["shape_id"])].append(row)
        else:
            failures_by_regime[str(row["regime"])] += 1

    baselines_by_regime: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in torch_by_shape.values():
        baselines_by_regime[str(row["regime"])].append(row)

    def aggregate(
        name: str, baselines: list[dict[str, Any]], failure_count: int,
    ) -> dict[str, object]:
        paired = []
        winning_configs: Counter[str] = Counter()
        for baseline in baselines:
            shape_id = str(baseline["shape_id"])
            candidates = ayla_by_shape.get(shape_id, [])
            if not candidates:
                continue
            best = max(candidates, key=lambda row: _paired_percentage(baseline, row))
            paired.append((baseline, best))
            winning_configs[str(best["configuration"]["name"])] += int(
                baseline["frequency"]
            )

        calls = sum(int(row["frequency"]) for row in baselines)
        covered_calls = sum(int(base["frequency"]) for base, _ in paired)
        torch_ms = sum(
            float(best["latency_ms"])
            * _paired_percentage(base, best) / 100
            * int(base["frequency"])
            for base, best in paired
        )
        ayla_ms = sum(
            float(best["latency_ms"]) * int(base["frequency"])
            for base, best in paired
        )
        weighted_flops = sum(
            2 * int(base["m"]) * int(base["n"]) * int(base["k"])
            * int(base["frequency"])
            for base, _ in paired
        )
        speedup = torch_ms / ayla_ms if ayla_ms else None
        shape_percentages = [
            _paired_percentage(base, best) for base, best in paired
        ]
        winning_shapes = sum(value > 100 for value in shape_percentages)
        return {
            "regime": name,
            "shape_count": len(baselines),
            "call_count": calls,
            "covered_shapes": len(paired),
            "covered_calls": covered_calls,
            "torch_weighted_latency_ms": (
                torch_ms / covered_calls if covered_calls else None
            ),
            "ayla_weighted_latency_ms": (
                ayla_ms / covered_calls if covered_calls else None
            ),
            "torch_aggregate_tflops": (
                weighted_flops / (torch_ms / 1e3) / 1e12 if torch_ms else None
            ),
            "ayla_aggregate_tflops": (
                weighted_flops / (ayla_ms / 1e3) / 1e12 if ayla_ms else None
            ),
            "percentage_of_torch": None if speedup is None else speedup * 100,
            "shape_mean_percentage_of_torch": (
                statistics.fmean(shape_percentages)
                if shape_percentages else None
            ),
            "shape_median_percentage_of_torch": (
                statistics.median(shape_percentages)
                if shape_percentages else None
            ),
            "winning_shapes": winning_shapes,
            "speedup": speedup,
            "latency_reduction_pct": (
                None if speedup is None else (1 - 1 / speedup) * 100
            ),
            "win_rate_pct": (
                100 * winning_shapes / len(paired) if paired else None
            ),
            "winning_configs_by_frequency": [
                config for config, _ in winning_configs.most_common(3)
            ],
            "failed_configurations": failure_count,
        }

    result = [
        aggregate(name, baselines, failures_by_regime[name])
        for name, baselines in sorted(baselines_by_regime.items())
    ]
    result.append(aggregate(
        "OVERALL", list(torch_by_shape.values()),
        sum(failures_by_regime.values()),
    ))
    return result


def shape_improvement_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, object]]:
    """Pair every shape's fastest Ayla result with its canonical baseline."""
    torch_by_shape = {
        str(row["shape_id"]): row for row in rows
        if row.get("implementation") == "torch"
        and row.get("status") == "ok"
        and row.get("latency_ms") is not None
    }
    ayla_by_shape: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if (
            row.get("implementation") == "ayla"
            and row.get("status") == "ok"
            and row.get("latency_ms") is not None
            and not bool(
                row.get("configuration", {}).get("unsafe_reuse_smem")
            )
        ):
            ayla_by_shape[str(row["shape_id"])].append(row)

    result = []
    for shape_id, baseline in torch_by_shape.items():
        candidates = ayla_by_shape.get(shape_id, [])
        if not candidates:
            continue
        best = max(candidates, key=lambda row: _paired_percentage(baseline, row))
        ayla_latency = float(best["latency_ms"])
        percentage = _paired_percentage(baseline, best)
        torch_latency = ayla_latency * percentage / 100
        result.append({
            "shape_id": shape_id,
            "op": str(baseline.get("op", "mm")),
            "batch": baseline.get("batch"),
            "m": int(baseline["m"]),
            "n": int(baseline["n"]),
            "k": int(baseline["k"]),
            "dtype": str(baseline["dtype"]),
            "layout": str(baseline["layout"]),
            "frequency": int(baseline["frequency"]),
            "regime": str(baseline["regime"]),
            "torch_latency_ms": torch_latency,
            "ayla_latency_ms": ayla_latency,
            "torch_tflops": float(best["tflops"]) / (percentage / 100),
            "ayla_tflops": float(best["tflops"]),
            "percentage_of_torch": percentage,
            "speedup": percentage / 100,
            "latency_reduction_pct": (1 - ayla_latency / torch_latency) * 100,
            "beats_torch": percentage > 100,
            "configuration_name": str(best["configuration"]["name"]),
            "configuration": best["configuration"],
        })
    return sorted(
        result,
        key=lambda row: (
            -float(row["percentage_of_torch"]),
            str(row["shape_id"]),
        ),
    )


def seed_recommendations(rows: list[dict[str, Any]]) -> list[dict[str, object]]:
    """Return the three most reliable measured configurations per regime."""
    by_regime: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if (
            row.get("implementation") == "ayla"
            and not bool(
                row.get("configuration", {}).get("unsafe_reuse_smem")
            )
        ):
            by_regime[str(row["regime"])].append(row)

    recommendations = []
    for regime_name, group in sorted(by_regime.items()):
        shape_count = len({str(row["shape_id"]) for row in group})
        configs: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in group:
            configs[_config_key(row)].append(row)
        ranked = []
        for key, config_rows in configs.items():
            successful = [
                row for row in config_rows
                if row.get("status") == "ok" and row.get("torch_pct") is not None
            ]
            values = [float(row["torch_pct"]) for row in successful]
            weighted = [
                (float(row["torch_pct"]), int(row["frequency"]))
                for row in successful if float(row["torch_pct"]) > 0
            ]
            ranked.append((
                len({str(row["shape_id"]) for row in successful}),
                _weighted_geomean(weighted) if weighted else 0.0,
                key, values, len(config_rows) - len(successful),
            ))
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        for rank, (coverage, score, key, values, failures) in enumerate(
            ranked[:3], 1,
        ):
            config = json.loads(key)
            recommendations.append({
                "regime": regime_name,
                "rank": rank,
                "configuration_name": config["name"],
                "configuration": config,
                "covered_shapes": coverage,
                "regime_shapes": shape_count,
                "weighted_percentage_of_torch": score,
                "median_percentage_of_torch": (
                    statistics.median(values) if values else None
                ),
                "worst_percentage_of_torch": min(values) if values else None,
                "failed_shapes": failures,
            })
    return recommendations


def _fmt(value: object, suffix: str = "") -> str:
    return "-" if value is None else f"{float(value):.2f}{suffix}"


def render(
    rows: list[dict[str, Any]], source: Path,
    improvements: list[dict[str, object]] | None = None,
) -> str:
    ayla_rows = [row for row in rows if row.get("implementation") == "ayla"]
    if not ayla_rows:
        raise ValueError("raw data contains no Ayla rows")

    by_regime: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ayla_rows:
        by_regime[str(row["regime"])].append(row)

    lines = [
        "# OSS PyTorch GEMM heuristic study",
        "",
        f"Generated from `{source}`. Results are grouped into behavioral regimes; "
        "individual measurements remain in the raw CSV/JSON.",
        "",
        "Environment: "
        + ", ".join(sorted({
            f"{row.get('gpu')} / torch {row.get('torch_version')} / "
            f"CUDA {row.get('cuda_version')}"
            for row in rows if row.get("gpu")
        })),
        "",
        "Buckets: `xs` ≤ 128, `small` ≤ 512, `medium` ≤ 2,048, "
        "`large` ≤ 8,192, and `xl` above 8,192. Aspect ratio is `tall` or "
        "`wide` at 4:1, otherwise `balanced`. Alignment records the largest "
        "tested reduction tile dividing K.",
        "",
        "Recorded operand layouts are normalized to contiguous A/BT and do not "
        "cause rejection. Cluster shape, operand multicast, and two-dimensional "
        "workgroup tiling are recorded explicitly; the broad ablations promote "
        "them only in regimes where they beat the current best.",
        "",
    ]
    improvements = improvements or improvement_rows(rows)
    overall = next(row for row in improvements if row["regime"] == "OVERALL")
    overall_summary = []
    if int(overall["covered_shapes"]):
        overall_summary = [
            f"Across all {overall['covered_shapes']} covered shapes, "
            f"{overall['winning_shapes']} beat Torch/cuBLAS "
            f"({float(overall['win_rate_pct']):.2f}%). The unweighted median "
            f"shape reaches "
            f"{float(overall['shape_median_percentage_of_torch']):.2f}% of "
            f"Torch/cuBLAS and the unweighted mean reaches "
            f"{float(overall['shape_mean_percentage_of_torch']):.2f}%.",
            "",
        ]
    lines.extend([
        "## Improvement summary",
        "",
        "Aggregates pair each shape's fastest successful Ayla configuration "
        "with its measured Torch baseline, weighted by recorded call frequency.",
        "",
        *overall_summary,
        "| regime | shapes | calls | covered | Torch TFLOPS | Ayla TFLOPS | "
        "% Torch | median shape % | wins | speedup | latency reduction | "
        "win rate | failures |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: | ---: | ---: |",
    ])
    for row in improvements:
        lines.append(
            f"| {row['regime']} | {row['shape_count']} | {row['call_count']} | "
            f"{row['covered_shapes']}/{row['shape_count']} | "
            f"{_fmt(row['torch_aggregate_tflops'])} | "
            f"{_fmt(row['ayla_aggregate_tflops'])} | "
            f"{_fmt(row['percentage_of_torch'], '%')} | "
            f"{_fmt(row['shape_median_percentage_of_torch'], '%')} | "
            f"{row['winning_shapes']} | "
            f"{_fmt(row['speedup'], 'x')} | "
            f"{_fmt(row['latency_reduction_pct'], '%')} | "
            f"{_fmt(row['win_rate_pct'], '%')} | "
            f"{row['failed_configurations']} |"
        )
    lines.append("")

    for regime_name in sorted(by_regime):
        group = by_regime[regime_name]
        exemplar = group[0]
        shape_ids = {str(row["shape_id"]) for row in group}
        frequencies = {
            str(row["shape_id"]): int(row["frequency"]) for row in group
        }
        total_frequency = sum(frequencies.values())
        configs: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in group:
            configs[_config_key(row)].append(row)

        successful_by_shape: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in group:
            if row.get("status") == "ok" and row.get("torch_pct") is not None:
                successful_by_shape[str(row["shape_id"])].append(row)
        winners = {
            _config_key(max(shape_rows, key=lambda row: float(row["torch_pct"])))
            for shape_rows in successful_by_shape.values() if shape_rows
        }

        ranked = []
        for key, config_rows in configs.items():
            successful = [
                row for row in config_rows
                if row.get("status") == "ok" and row.get("torch_pct") is not None
            ]
            measurements = [
                (float(row["torch_pct"]), int(row["frequency"]))
                for row in successful
                if float(row["torch_pct"]) > 0
            ]
            score = _weighted_geomean(measurements) if measurements else 0.0
            ranked.append((
                len({str(row["shape_id"]) for row in successful}),
                score,
                key,
                successful,
                len(config_rows) - len(successful),
            ))
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)

        descriptor = ", ".join(
            f"{field}={exemplar[field]}" for field in REGIME_FIELDS
        )
        shape_rows = {
            str(row["shape_id"]): row for row in group
        }.values()
        n_buckets = Counter(str(row["n_bucket"]) for row in shape_rows)
        shape_rows = {
            str(row["shape_id"]): row for row in group
        }.values()
        k_buckets = Counter(str(row["k_bucket"]) for row in shape_rows)
        lines.extend([
            f"## {regime_name}",
            "",
            f"{descriptor}. {len(shape_ids)} shapes, "
            f"{total_frequency:,} recorded calls. N buckets: "
            f"{dict(sorted(n_buckets.items()))}; K buckets: "
            f"{dict(sorted(k_buckets.items()))}.",
            "",
            "| rank | seed configuration | coverage | weighted % Torch | "
            "median % Torch | worst % Torch | failures |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
        ])
        for rank, (coverage, score, key, successful, failures) in enumerate(
            ranked[:3], 1,
        ):
            values = [float(row["torch_pct"]) for row in successful]
            config = json.loads(key)
            lines.append(
                f"| {rank} | `{config['name']}` | {coverage}/{len(shape_ids)} | "
                f"{score:.1f}% | "
                f"{statistics.median(values) if values else 0:.1f}% | "
                f"{min(values) if values else 0:.1f}% | {failures} |"
            )
        lines.extend(["", "Seed details:", ""])
        for _, _, key, _, _ in ranked[:3]:
            config = json.loads(key)
            marker = " (wins at least one shape)" if key in winners else ""
            lines.append(
                f"- `{config['name']}`{marker}: {_format_knobs(config)}"
            )
        lines.append("")

        probes = [row for row in group if row.get("search_scope") == "regime_probe"]
        if probes:
            lines.extend([
                "Representative-shape knob probes:",
                "",
                "| configuration | result | % Torch |",
                "| --- | --- | ---: |",
            ])
            for row in sorted(
                probes,
                key=lambda value: (
                    value.get("status") != "ok",
                    -float(value.get("torch_pct") or 0),
                ),
            ):
                config = row["configuration"]
                result = "ok" if row.get("status") == "ok" else str(row["error"])
                lines.append(
                    f"| `{config['name']}` | {result} | "
                    f"{_fmt(row.get('torch_pct'), '%')} |"
                )
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=RAW_JSON)
    parser.add_argument("--output", type=Path, default=HEURISTICS_MD)
    parser.add_argument("--improvements-csv", type=Path, default=IMPROVEMENTS_CSV)
    parser.add_argument("--improvements-json", type=Path, default=IMPROVEMENTS_JSON)
    parser.add_argument(
        "--shape-improvements-csv", type=Path,
        default=SHAPE_IMPROVEMENTS_CSV,
    )
    parser.add_argument(
        "--shape-improvements-json", type=Path,
        default=SHAPE_IMPROVEMENTS_JSON,
    )
    parser.add_argument("--seeds-json", type=Path, default=SEEDS_JSON)
    args = parser.parse_args()
    rows = json.loads(args.input.read_text())
    if not isinstance(rows, list):
        raise ValueError(f"{args.input}: expected a JSON list")
    improvements = improvement_rows(rows)
    document = render(rows, args.input, improvements)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(document)
    write_json(improvements, args.improvements_json)
    write_rows_csv(improvements, args.improvements_csv)
    shape_improvements = shape_improvement_rows(rows)
    write_json(shape_improvements, args.shape_improvements_json)
    write_rows_csv(shape_improvements, args.shape_improvements_csv)
    write_json(seed_recommendations(rows), args.seeds_json)
    print(f"heuristic document: {args.output.resolve()}")
    print(f"improvements JSON: {args.improvements_json.resolve()}")
    print(f"improvements CSV:  {args.improvements_csv.resolve()}")
    print(f"shape improvements JSON: {args.shape_improvements_json.resolve()}")
    print(f"shape improvements CSV:  {args.shape_improvements_csv.resolve()}")
    print(f"heuristic seeds:   {args.seeds_json.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
