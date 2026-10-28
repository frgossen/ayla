#!/usr/bin/env python3
"""Keep each shape's best distinct schedule families from inference-v2 results.

Inference timing ranks a shape's configurations at full clock, and that
ranking does not carry over exactly to full graphs. So instead of one winner
per shape, the schedule table keeps up to ``--families`` performant families
for a full graph to choose from. A family is what a CTA and its warpgroups
compute: the output, warpgroup and K tiles, the orientation, the tiles grouped
per CTA, a fixed or one-tile-per-CTA grid, and the cluster. Stages, output
slices, cache hints, L2 promotion, PTX level and issue order are variants
within a family; only each family's best variant is kept.

Example::

    pixi run python -m benchmarks.select_gemm_schedule_families \
        --update-table .benchmarks/better_benchmark_gemm/*/gpu*.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from benchmarks.oss_gemm_corpus import ScheduleConfig, timing_protocol

TABLE = Path(__file__).resolve().parents[1] / "ayla/gemm_schedule_table.py"
PROTOCOL = "inference-v2"

Shape = tuple[str, str, int, int, int, int]

_DEFAULTS = {
    field.name: field.default
    for field in dataclasses.fields(ScheduleConfig)
    if field.default is not dataclasses.MISSING
} | {"pipeline_output_waves": False}
# Every table entry lists these; the rest are omitted at their defaults.
_REQUIRED = tuple(
    field.name for field in dataclasses.fields(ScheduleConfig)
    if field.default is dataclasses.MISSING and field.name != "name"
)
# The table builds through gemm_jit, which supports only these values.
_UNSUPPORTED = {
    "wgmma_group_size": (None,),
    "wgmma_pending_groups": (0,),
    "wgmma_epilogue_pipeline": (False,),
    "unsafe_reuse_smem": (False,),
    "split_k_parts": (None, 1),
    "output_physical_order": (None,),
    "output_physical_block_width": (None,),
}


def family(config: dict[str, Any]) -> tuple[object, ...]:
    """The tile geometry, grid and cluster that a variant tunes around."""
    c = _DEFAULTS | config
    return (
        c["transpose_output"],
        c["output_m_tile"], c["output_n_tile"],
        c["warpgroup_m_tile"] or c["output_m_tile"], c["warpgroup_n_tile"],
        c["operand_k_tile"],
        c["output_tiles_per_cta"], c["output_n_tiles_per_cta"],
        c["output_cta_count"] is not None,
        c["multicast"], c["cluster_m_ctas"], c["cluster_n_ctas"],
    )


def table_entry(config: dict[str, Any]) -> dict[str, object] | None:
    """``config`` in the table's form, or None if gemm_jit cannot build it."""
    c = _DEFAULTS | config
    if any(c.get(key, allowed[0]) not in allowed
           for key, allowed in _UNSUPPORTED.items()):
        return None
    # The sweep sets a fixed grid's cluster traversal outside the builder.
    if c["output_cta_count"] is not None and c["multicast"]:
        return None
    entry = {key: c[key] for key in _REQUIRED} | {
        key: c[key] for key, default in _DEFAULTS.items()
        if key not in _UNSUPPORTED and c[key] != default
    }
    return dict(sorted(entry.items()))


def ranked_candidates(
    paths: list[Path],
) -> dict[Shape, list[tuple[float, dict[str, object]]]]:
    """Each shape's distinct buildable configurations, best median first."""
    observed: dict[Shape, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list),
    )
    for path in paths:
        rows = json.loads(path.read_text())
        if timing_protocol(rows, path) != PROTOCOL:
            raise ValueError(f"{path}: not timed under {PROTOCOL}")
        for row in rows:
            pct = row.get("torch_pct")
            if (
                row.get("implementation") != "ayla" or row.get("status") != "ok"
                or not isinstance(pct, (int, float)) or pct <= 0
            ):
                continue
            entry = table_entry(row["configuration"])
            if entry is None:
                continue
            shape = (
                row["op"], row["dtype"], row.get("batch") or 1,
                row["m"], row["n"], row["k"],
            )
            observed[shape][json.dumps(entry)].append(float(pct))
    return {
        shape: sorted(
            ((statistics.median(pcts), json.loads(key))
             for key, pcts in configs.items()),
            key=lambda item: -item[0],
        )
        for shape, configs in observed.items()
    }


def select_families(
    installed: list[dict[str, object]],
    ranked: list[tuple[float, dict[str, object]]],
    *, families: int, band: float,
) -> list[dict[str, object]]:
    """Installed seeds, then each new family's best variant within the band."""
    seeds = list(installed)
    seen = {family(config) for config in installed}
    for pct, config in ranked:
        if len(seen) >= families or pct < band * ranked[0][0]:
            break
        if family(config) not in seen:
            seeds.append(config)
            seen.add(family(config))
    return seeds


def _format_table(
    docstring: str,
    configs: list[dict[str, object]],
    schedules: dict[tuple[str, str], dict[tuple[int, int, int, int], tuple[int, ...]]],
) -> str:
    lines = [
        docstring.rstrip("\n"),
        "# fmt: off",
        "CONFIGS: tuple[dict[str, object], ...] = (",
        *(
            "    dict(" + ", ".join(f"{k}={v!r}" for k, v in c.items()) + "),"
            for c in configs
        ),
        ")",
        "",
        "# (operation, dtype) -> (batch, m, n, k) -> indices into CONFIGS, best first.",
        "SCHEDULES: dict[tuple[str, str], dict[tuple[int, int, int, int], tuple[int, ...]]] = {",
    ]
    for group, table in schedules.items():
        lines.append(f"    {group!r}: {{")
        lines += [f"        {shape!r}: {seeds!r}," for shape, seeds in table.items()]
        lines.append("    },")
    return "\n".join([*lines, "}", ""])


def update_table(
    candidates: dict[Shape, list[tuple[float, dict[str, object]]]],
    *, families: int, band: float,
) -> tuple[int, int]:
    """Widen the measured shapes' table entries; return (shapes, seeds) added."""
    from ayla.gemm_schedule_table import CONFIGS, SCHEDULES

    configs = list(CONFIGS)
    index = {json.dumps(table_entry(dict(c))): i for i, c in reversed(list(enumerate(configs)))}
    schedules = {group: dict(table) for group, table in SCHEDULES.items()}
    widened = added = 0
    for (op, dtype, batch, m, n, k), ranked in candidates.items():
        table = schedules.get((op, dtype), {})
        shape = (batch, m, n, k)
        if shape not in table:
            continue
        installed = [dict(configs[i]) for i in table[shape]]
        seeds = list(table[shape])
        for config in select_families(
            installed, ranked, families=families, band=band,
        )[len(installed):]:
            key = json.dumps(config)
            if key not in index:
                index[key] = len(configs)
                configs.append(config)
            seeds.append(index[key])
        widened += len(seeds) > len(table[shape])
        added += len(seeds) - len(table[shape])
        table[shape] = tuple(seeds)
    text = TABLE.read_text()
    docstring = text[:text.index("# fmt: off")]
    TABLE.write_text(_format_table(docstring, configs, schedules))
    return widened, added


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="+",
                        help=f"raw benchmark JSON timed under {PROTOCOL}")
    parser.add_argument("--families", type=int, default=10,
                        help="families per shape, counting the installed seeds'")
    parser.add_argument("--band", type=float, default=0.95,
                        help="keep families whose best variant reaches this "
                        "fraction of the shape's best inference score")
    parser.add_argument("--update-table", action="store_true",
                        help=f"add the families to {TABLE.name}'s measured shapes")
    args = parser.parse_args()
    candidates = ranked_candidates(args.results)
    if args.update_table:
        widened, added = update_table(
            candidates, families=args.families, band=args.band,
        )
        print(f"added {added} seeds to {widened} shapes in {TABLE}")
        return 0
    for shape, ranked in sorted(candidates.items()):
        seeds = select_families([], ranked, families=args.families, band=args.band)
        print(shape, len(seeds), "families")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
