"""Merge full GEMM candidate tables, preserving the target's preferred order."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import runpy

from benchmarks.select_gemm_schedule_families import DEFAULT_MAX_CANDIDATES, TABLE


def merge_tables(target: dict, source: dict, *, max_candidates: int = DEFAULT_MAX_CANDIDATES) -> tuple[list[dict], dict]:
    """Deduplicate by configuration, append source candidates, reject overflow."""
    if max_candidates < 1:
        raise ValueError("max_candidates must be positive")
    configs = list(target["CONFIGS"])
    keys = [json.dumps(c, sort_keys=True) for c in configs]
    indices = {}
    for i, key in enumerate(keys):
        indices.setdefault(key, i)
    schedules = {}
    for table in (target, source):
        for group, shapes in table["SCHEDULES"].items():
            merged = schedules.setdefault(group, {})
            for shape, candidates in shapes.items():
                seeds = list(merged.get(shape, ()))
                seen = {keys[i] for i in seeds}
                for candidate in candidates:
                    config = table["CONFIGS"][candidate]
                    key = json.dumps(config, sort_keys=True)
                    if key in seen:
                        continue
                    if key not in indices:
                        indices[key] = len(configs)
                        configs.append(config)
                        keys.append(key)
                    seeds.append(indices[key])
                    seen.add(key)
                if len(seeds) > max_candidates:
                    raise ValueError(f"{group} {shape}: {len(seeds)} candidates exceed limit {max_candidates}")
                merged[shape] = tuple(seeds)
    return configs, schedules


def replace_tables(text: str, configs: list[dict], schedules: dict) -> str:
    """Replace only CONFIGS/SCHEDULES; preserve routing and admission metadata."""
    replacements = {}
    for node in ast.parse(text).body:
        if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name):
            continue
        name = node.target.id
        if name == "CONFIGS":
            lines = ["CONFIGS: tuple[dict[str, object], ...] = ("]
            lines += ["    dict(" + ", ".join(f"{k}={v!r}" for k, v in c.items()) + ")," for c in configs]
            lines.append(")")
        elif name == "SCHEDULES":
            lines = ["SCHEDULES: dict[tuple[str, str], dict[tuple[int, int, int, int], tuple[int, ...]]] = {"]
            for group, shapes in schedules.items():
                lines.append(f"    {group!r}: {{")
                lines += [f"        {shape!r}: {seeds!r}," for shape, seeds in shapes.items()]
                lines.append("    },")
            lines.append("}")
        else:
            continue
        replacements[node.lineno - 1] = (node.end_lineno, "\n".join(lines) + "\n")
    if len(replacements) != 2:
        raise ValueError("expected CONFIGS and SCHEDULES declarations")
    original = text.splitlines(keepends=True)
    for start, (end, replacement) in sorted(replacements.items(), reverse=True):
        original[start:end] = [replacement]
    return "".join(original)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--target", type=Path, default=TABLE)
    parser.add_argument("--max-candidates", type=int, default=DEFAULT_MAX_CANDIDATES)
    parser.add_argument("--write", action="store_true", help="write the merged target table")
    args = parser.parse_args()
    target = runpy.run_path(str(args.target), run_name="ayla._merge_target")
    source = runpy.run_path(str(args.source), run_name="ayla._merge_source")
    configs, schedules = merge_tables(target, source, max_candidates=args.max_candidates)
    added = sum(len(ids) for shapes in schedules.values() for ids in shapes.values()) - sum(
        len(ids) for shapes in target["SCHEDULES"].values() for ids in shapes.values())
    changed = configs != list(target["CONFIGS"]) or schedules != target["SCHEDULES"]
    if args.write and changed:
        args.target.write_text(replace_tables(args.target.read_text(), configs, schedules))
    print(json.dumps(dict(added_candidates=added, added_configs=len(configs)-len(target["CONFIGS"]),
                         shape_keys=sum(map(len, schedules.values())),
                         largest_list=max((len(ids) for shapes in schedules.values() for ids in shapes.values()), default=0),
                         max_candidates=args.max_candidates, written=args.write and changed), indent=2))


if __name__ == "__main__":
    main()
