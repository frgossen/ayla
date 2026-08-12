#!/usr/bin/env python3
"""Plot the CSV produced by tools/power_efficiency_sweep.py."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics


COLORS = {
    "cuBLAS torch.mm": "#4C78A8",
    "Ayla best": "#F58518",
    "Triton TMA persistent": "#54A24B",
    "CUTLASS CuTe DSL": "#E45756",
    "Triton tuned @ 350 W": "#72B7B2",
    "Triton tuned @ 500 W": "#54A24B",
    "Triton current (@ 650 W)": "#2E7D32",
}

DISPLAY_NAMES = {
    "Ayla best": "Ayla k3 4-tile",
}

PAIRED_NAMES = {
    "torch_cublas_bf16": "cuBLAS torch.mm",
    "ayla_k3_4tiles_split2_dbuf2": "Ayla k3 4-tile",
    "triton_tma_persistent_bf16": "Triton TMA persistent",
    "cutlass_cute_persistent_bf16": "CUTLASS CuTe DSL",
}

NCU_TO_SWEEP_NAME = {
    "Ayla k3 4-tile": "Ayla best",
    "cuBLAS": "cuBLAS torch.mm",
    "CUTLASS CuTe DSL": "CUTLASS CuTe DSL",
}

# 2 * M * N * K, expressed in GFLOP, for M=N=K=4096.
GEMM_GFLOP = 2 * 4096**3 / 1e9


def _load(csv_path: Path) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    with csv_path.open(newline="") as source:
        for row in csv.DictReader(source):
            rows.append({
                key: value if key == "implementation" else float(value)
                for key, value in row.items()
            })
    if not rows:
        raise ValueError(f"no rows in {csv_path}")
    return rows


def _load_ncu_efficiencies(ncu_summary: Path) -> dict[str, dict[str, float]]:
    """Load the per-kernel tensor duty cycle measured by NCU.

    NCU replay makes its clock unsuitable for a power-cap plot.  Total cycles
    divided by tensor-active cycles is still useful: applied to an uninterrupted
    sweep result, it removes pipeline gaps while retaining that result's real
    random-input DVFS operating point.
    """
    efficiencies: dict[str, dict[str, float]] = {}
    with ncu_summary.open(newline="") as source:
        for row in csv.DictReader(source):
            sweep_name = NCU_TO_SWEEP_NAME.get(row["implementation"])
            if sweep_name is None:
                continue
            total_cycles = float(row["sm_cycles"])
            tensor_cycles = float(row["tensor_active_cycles"])
            efficiencies[sweep_name] = {
                "total_cycles": total_cycles,
                "tensor_cycles": tensor_cycles,
                "tensor_active_fraction": tensor_cycles / total_cycles,
            }
    if not efficiencies:
        raise ValueError(f"no recognized implementations in {ncu_summary}")
    return efficiencies


def _make_roofline(
    rows: list[dict[str, float | str]], ncu_summary: Path,
) -> list[dict[str, float | str]]:
    """Build a switching-aware compute roofline for each measured power cap.

    For every NCU-profiled implementation, project its uninterrupted random-BF16
    result from the observed tensor duty cycle to 100%.  The upper envelope is
    the least implementation-specific clock operating point demonstrated at a
    cap.  It is an optimistic compute ceiling, not a claim that 100% tensor duty
    would preserve exactly the same voltage/frequency state.
    """
    efficiencies = _load_ncu_efficiencies(ncu_summary)
    limits = sorted({int(row["power_limit_w"]) for row in rows})
    roofline: list[dict[str, float | str]] = []
    for limit in limits:
        candidates = []
        for row in rows:
            name = str(row["implementation"])
            if int(row["power_limit_w"]) != limit or name not in efficiencies:
                continue
            ncu = efficiencies[name]
            achieved_tflops = float(row["tflops"])
            roof_tflops = achieved_tflops / ncu["tensor_active_fraction"]
            effective_clock_mhz = ncu["total_cycles"] / (
                float(row["median_ms"]) * 1000
            )
            candidates.append((roof_tflops, name, achieved_tflops, effective_clock_mhz, ncu))
        if not candidates:
            continue
        roof_tflops, source_name, achieved_tflops, effective_clock_mhz, ncu = max(candidates)

        cublas = next((candidate for candidate in candidates if candidate[1] == "cuBLAS torch.mm"), None)
        cublas_roof_tflops = cublas[0] if cublas else float("nan")
        roofline.append({
            "power_limit_w": limit,
            "roofline_tflops": roof_tflops,
            "runtime_floor_ms": GEMM_GFLOP / roof_tflops,
            "source_implementation": source_name,
            "source_tflops": achieved_tflops,
            "source_effective_sm_clock_mhz": effective_clock_mhz,
            "source_tensor_active_pct": 100 * ncu["tensor_active_fraction"],
            "cublas_operating_point_tflops": cublas_roof_tflops,
            "cublas_operating_point_runtime_floor_ms": (
                GEMM_GFLOP / cublas_roof_tflops if cublas else float("nan")
            ),
        })
    return roofline


def _write_roofline(rows: list[dict[str, float | str]], output_path: Path) -> None:
    if not rows:
        return
    with output_path.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_power_sweep(
    csv_path: str | Path,
    output_dir: str | Path | None = None,
    additional_csvs: list[str | Path] | None = None,
    drop_implementations: set[str] | None = None,
    ncu_summary: str | Path | None = None,
) -> list[Path]:
    """Generate comparison plots and return their paths."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    csv_path = Path(csv_path)
    output_dir = Path(output_dir) if output_dir else csv_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _load(csv_path)
    for additional_csv in additional_csvs or []:
        rows.extend(_load(Path(additional_csv)))
    if drop_implementations:
        rows = [row for row in rows if row["implementation"] not in drop_implementations]
    # Later CSVs intentionally override earlier measurements of the same point.
    # This is used for the isolated 350 W reruns, where a fresh process avoids
    # the power governor hysteresis seen in a multi-policy process.
    rows = list({
        (str(row["implementation"]), int(row["power_limit_w"])): row
        for row in rows
    }.values())
    implementations = list(dict.fromkeys(str(row["implementation"]) for row in rows))
    limits = sorted({int(row["power_limit_w"]) for row in rows})
    by_implementation = {
        name: sorted(
            (row for row in rows if row["implementation"] == name),
            key=lambda row: row["power_limit_w"],
        )
        for name in implementations
    }
    if ncu_summary is None:
        default_ncu_summary = csv_path.parent / "ncu-350w" / "summary.csv"
        ncu_summary = default_ncu_summary if default_ncu_summary.exists() else None
    roofline = _make_roofline(rows, Path(ncu_summary)) if ncu_summary else []
    roofline_path = output_dir / "roofline.csv"
    _write_roofline(roofline, roofline_path)

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    metrics = [
        ("tflops", "Throughput", "TFLOP/s"),
        ("median_ms", "Runtime", "Milliseconds per GEMM"),
        ("joules_per_gemm", "Energy per 4096³ GEMM", "J/GEMM"),
        ("median_sm_clock_mhz", "Sustained SM clock", "MHz"),
    ]
    for axis, (metric, title, ylabel) in zip(axes.flat, metrics):
        for name, implementation_rows in by_implementation.items():
            axis.plot(
                [row["power_limit_w"] for row in implementation_rows],
                [row[metric] for row in implementation_rows],
                marker="o", linewidth=2, label=DISPLAY_NAMES.get(name, name),
                color=COLORS.get(name),
            )
        if roofline and metric in {"tflops", "median_ms"}:
            roof_metric = "roofline_tflops" if metric == "tflops" else "runtime_floor_ms"
            axis.plot(
                [row["power_limit_w"] for row in roofline],
                [row[roof_metric] for row in roofline],
                marker="D", markersize=5, linestyle="--", linewidth=2.2,
                color="#222222", label="Theoretical maximum",
                zorder=5,
            )
        axis.set(title=title, xlabel="Power limit (W)", ylabel=ylabel, xticks=limits)
    axes[0, 0].legend(frameon=True)
    summary_path = output_dir / "power_efficiency_summary.png"
    figure.savefig(summary_path, dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    width = min(0.8 / len(implementations), 0.36)
    centers = list(range(len(limits)))
    for implementation_index, name in enumerate(implementations):
        values_by_limit = {
            int(row["power_limit_w"]): float(row["tflops"])
            for row in by_implementation[name]
        }
        offset = (implementation_index - (len(implementations) - 1) / 2) * width
        bars = axis.bar(
            [center + offset for center in centers],
            [values_by_limit[limit] for limit in limits],
            width=width, label=DISPLAY_NAMES.get(name, name), color=COLORS.get(name),
        )
        axis.bar_label(bars, fmt="%.0f", padding=3, fontsize=8, rotation=90)
    axis.set(
        title="BF16 4096³ GEMM throughput by GPU power limit",
        xlabel="Power limit (W)", ylabel="TFLOP/s",
        xticks=centers, xticklabels=[str(limit) for limit in limits],
    )
    axis.legend(frameon=True)
    bar_path = output_dir / "tflops_barplot.png"
    figure.savefig(bar_path, dpi=180)
    plt.close(figure)

    paired_path = output_dir / "fixed-750mhz-paired.json"
    generated = [summary_path, bar_path]
    if roofline:
        generated.append(roofline_path)
    if paired_path.exists():
        paired = json.loads(paired_path.read_text())
        baseline = paired["baseline"]
        comparisons = paired["comparisons"]
        values = {
            baseline: statistics.median(
                row["baseline_median_us"] for row in comparisons.values()
            ),
            **{
                name: row["candidate_median_us"]
                for name, row in comparisons.items()
            },
        }
        ordered = [
            "cutlass_cute_persistent_bf16",
            "torch_cublas_bf16",
            "triton_tma_persistent_bf16",
            "ayla_k3_4tiles_split2_dbuf2",
        ]
        figure, axis = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
        bars = axis.bar(
            [PAIRED_NAMES[name] for name in ordered],
            [values[name] for name in ordered],
            color=[
                COLORS["CUTLASS CuTe DSL"],
                COLORS["cuBLAS torch.mm"],
                COLORS["Triton TMA persistent"],
                COLORS["Ayla best"],
            ],
        )
        axis.bar_label(bars, fmt="%.1f µs", padding=3)
        axis.set(
            title="Equal-clock BF16 4096³ GEMM runtime (750 MHz, paired/interleaved)",
            ylabel="Microseconds per GEMM",
        )
        axis.text(
            0.5, -0.13,
            "Lower is better. Unlike the power-cap plot, every kernel runs at the same SM clock.",
            transform=axis.transAxes, ha="center",
        )
        equal_clock_path = output_dir / "equal_clock_runtime_750mhz.png"
        figure.savefig(equal_clock_path, dpi=180)
        plt.close(figure)
        generated.append(equal_clock_path)
    return generated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--additional-csv", action="append", type=Path, default=[])
    parser.add_argument("--drop-implementation", action="append", default=[])
    parser.add_argument(
        "--ncu-summary", type=Path,
        help="NCU summary CSV used to add a power-cap-aware compute roofline",
    )
    args = parser.parse_args()
    for path in plot_power_sweep(
        args.csv,
        args.output_dir,
        additional_csvs=args.additional_csv,
        drop_implementations=set(args.drop_implementation),
        ncu_summary=args.ncu_summary,
    ):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
