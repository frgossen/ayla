"""Plot the selected OSS GEMM performance distribution in 1% buckets."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    rows = json.loads(args.results.read_text())
    selected = [
        row for row in rows
        if row.get("implementation") == "ayla"
        and row.get("status") == "ok"
        and row.get("torch_pct") is not None
    ]
    by_operation = {
        operation: np.asarray(
            [
                float(row["torch_pct"])
                for row in selected
                if row.get("op", "mm") == operation
            ],
            dtype=float,
        )
        for operation in ("mm", "addmm", "bmm")
    }
    plotted_operations = [
        operation for operation in ("mm", "addmm", "bmm")
        if by_operation[operation].size
    ]
    if not plotted_operations:
        raise ValueError("results contain no successful Ayla measurements")
    percentages = np.concatenate(
        [by_operation[operation] for operation in plotted_operations],
    )

    lower = math.floor(float(percentages.min()))
    upper = math.ceil(float(percentages.max()))
    bins = np.arange(lower, upper + 1, 1, dtype=float)
    geomean = math.exp(float(np.log(percentages).mean()))

    fig, ax = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
    ax.hist(
        [by_operation[operation] for operation in plotted_operations],
        bins=bins,
        stacked=True,
        color=[
            {
                "mm": "#4c78a8",
                "addmm": "#f58518",
                "bmm": "#54a24b",
            }[operation]
            for operation in plotted_operations
        ],
        edgecolor="white",
        linewidth=0.7,
        label=[
            f"{operation} (n={by_operation[operation].size})"
            for operation in plotted_operations
        ],
    )
    ax.axvline(100, color="#d62728", linewidth=2.2, label="baseline (100%)")
    ax.axvline(
        geomean,
        color="#1f4e79",
        linestyle="--",
        linewidth=1.8,
        label=f"geomean ({geomean:.2f}%)",
    )
    ax.set_xlim(lower, upper)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.set_xlabel("Ayla performance relative to Torch/cuBLAS (%) — 1% buckets")
    ax.set_ylabel("Kernel shapes")
    ax.set_title(f"OSS GEMM kernel performance distribution (n={len(percentages)})")
    ax.grid(axis="y", color="#dddddd", linewidth=0.8)
    ax.legend(frameon=False)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
