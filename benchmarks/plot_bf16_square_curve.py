#!/usr/bin/env python3
"""Plot an Ayla/cuBLAS square-curve CSV."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    with args.csv.open() as input_file:
        rows = list(csv.DictReader(input_file))
    sizes = [int(row["size"]) for row in rows]
    ayla = [float(row["ayla_tflops"]) for row in rows]
    torch = [float(row["torch_tflops"]) for row in rows]
    ayla_deltas = [
        100 * (ayla_tflops / torch_tflops - 1)
        for ayla_tflops, torch_tflops in zip(ayla, torch, strict=True)
    ]
    input_policies = {row.get("input_policy", "") for row in rows}
    normalized_policies = {
        "fixed_weight_rotating"
        if policy.startswith("fixed_weight_rotating_") else policy
        for policy in input_policies
    }
    if len(normalized_policies) != 1:
        raise RuntimeError(f"mixed input policies: {sorted(input_policies)}")
    input_policy = next(iter(normalized_policies))

    fig, axis = plt.subplots(figsize=(8.2, 4.8))
    axis.plot(sizes, ayla, marker="o", linewidth=2, label="Ayla")
    axis.plot(sizes, torch, marker="o", linewidth=2, label="cuBLAS")
    for point_index, (size, ayla_tflops, torch_tflops) in enumerate(zip(
        sizes, ayla, torch, strict=True,
    )):
        advantage = ayla_deltas[point_index]
        difference = ayla_tflops - torch_tflops
        axis.annotate(
            f"Ayla {advantage:+.1f}%\n{difference:+.1f} TFLOP/s",
            (size, max(ayla_tflops, torch_tflops)),
            xytext=(0, 7 + 12 * (point_index % 2)),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
            color="#147d3f" if advantage >= 0 else "#b3261e",
        )
    axis.set_xscale("log", base=2)
    axis.set_xticks(
        sizes, [f"{size:,}" for size in sizes], rotation=25, ha="right",
    )
    axis.set_xlabel("Square matrix size")
    axis.set_ylabel("TFLOP/s")
    if "cache_cold" in input_policy:
        cache_label = "L2-evicted cache-cold"
    elif "fixed_weight_rotating" in input_policy:
        cache_label = "fixed weights, rotating activations/outputs"
    else:
        cache_label = "cache-hot"
    axis.set_title(
        f"H100 square BF16 GEMM — graph-captured, {cache_label}\n"
        "labels show throughput advantage vs cuBLAS"
    )
    axis.grid(True, alpha=.3)
    axis.legend()
    axis.margins(y=.25)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180 if args.output.suffix == ".png" else None)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
