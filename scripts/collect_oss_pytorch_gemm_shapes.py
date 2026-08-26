#!/usr/bin/env python3
"""Collect OSS PyTorch GEMM shapes from its recorded operator_inp_logs.

PyTorch records every operator invocation of the HuggingFace, timm and
TorchBench models under a ``TorchDispatchMode`` and checks the result into its
own tree, one text file per model, as
``benchmarks/dynamo/microbenchmarks/operator_inp_logs/*/<Model>_training.txt``::

    Operator: aten.mm.default
    cnt: 48, ((T([1024, 1024], f16), T([1024, 1024], f16)), {})

This script reuses PyTorch's ``OperatorInputsLoader`` to read those logs, keeps
the mm/addmm/bmm entries, and writes them to a JSON corpus so ayla can benchmark
against shapes real models produce instead of invented ones.

The logs ship with a PyTorch *source* checkout, not the installed wheel, so a
checkout is required. Only run this when refreshing the corpus -- the benchmarks
read the committed ``benchmarks/oss_pytorch_gemm_collection.json``::

    scripts/collect_oss_pytorch_gemm_shapes.py --pytorch-root ~/workspace-phoenix/pytorch
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
from torch._subclasses.fake_tensor import FakeTensorMode


AYLA_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = AYLA_ROOT / "benchmarks" / "oss_pytorch_gemm_collection.json"

MICROBENCHMARK_SUBDIR = Path("benchmarks") / "dynamo" / "microbenchmarks"

# Suite name -> the OperatorInputsLoader factory that unions its log directory.
SUITE_LOADERS = {
    "huggingface": "get_huggingface_loader",
    "timm": "get_timm_loader",
    "torchbench": "get_torchbench_loader",
}

# Short name -> operator as it appears in the ``Operator:`` headers.
OPERATORS = {
    "mm": "aten.mm.default",
    "addmm": "aten.addmm.default",
    "bmm": "aten.bmm.default",
}


def resolve_pytorch_root(explicit: str | None) -> Path:
    """Locate a PyTorch source checkout, or exit with a usable message."""
    candidates = []
    if explicit is not None:
        candidates.append(Path(explicit).expanduser())
    if env_root := os.environ.get("PYTORCH_ROOT"):
        candidates.append(Path(env_root).expanduser())
    # Editable/source installs put torch/ directly under the checkout root.
    candidates.append(Path(torch.__file__).resolve().parents[1])

    for candidate in candidates:
        if (candidate / MICROBENCHMARK_SUBDIR / "operator_inp_utils.py").is_file():
            return candidate.resolve()

    sys.exit(
        "No PyTorch source checkout found (looked for "
        f"{MICROBENCHMARK_SUBDIR / 'operator_inp_utils.py'} in: "
        + ", ".join(str(candidate) for candidate in candidates)
        + "). Pass --pytorch-root or set PYTORCH_ROOT."
    )


def import_operator_inp_utils(pytorch_root: Path):
    sys.path.insert(0, str(pytorch_root / MICROBENCHMARK_SUBDIR))
    import operator_inp_utils

    return operator_inp_utils


def encode_tensor(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "stride": list(tensor.stride()),
    }


def gemm_dims(op_name: str, args: tuple[torch.Tensor, ...]) -> dict[str, int]:
    """The B/M/K/N of one recorded call. addmm's leading arg is the bias."""
    left, right = (args[1], args[2]) if op_name == "addmm" else (args[0], args[1])
    if op_name == "bmm":
        batch, m, k = left.shape
        return {"b": int(batch), "m": int(m), "k": int(k), "n": int(right.shape[2])}
    m, k = left.shape
    return {"m": int(m), "k": int(k), "n": int(right.shape[1])}


def collect(operator_inp_utils) -> list[dict[str, Any]]:
    """One entry per unique (suite, operator, argument set)."""
    entries = []

    # deserialize_args materializes every recorded shape, so stay fake.
    with FakeTensorMode():
        for suite_name, factory in SUITE_LOADERS.items():
            loader = getattr(operator_inp_utils.OperatorInputsLoader, factory)()

            for op_name, qualified_name in OPERATORS.items():
                records = loader.operator_db.get(qualified_name, {})
                for serialized_inputs, count in records.items():
                    args, kwargs = operator_inp_utils.deserialize_args(
                        serialized_inputs
                    )
                    if kwargs or not all(isinstance(a, torch.Tensor) for a in args):
                        raise ValueError(
                            f"{suite_name}/{op_name}: expected tensor-only args, "
                            f"got {args!r}, {kwargs!r}"
                        )
                    entries.append(
                        {
                            "suite": suite_name,
                            "op": op_name,
                            "count": count,
                            **gemm_dims(op_name, args),
                            "args": [encode_tensor(arg) for arg in args],
                        }
                    )

                total = sum(records.values())
                print(
                    f"{suite_name}/{op_name}: {len(records)} unique inputs, "
                    f"{total} recorded calls",
                    file=sys.stderr,
                )

    # Most-used shapes first, with a total order so the committed file does not
    # churn between runs.
    entries.sort(key=lambda e: (e["suite"], e["op"], -e["count"], json.dumps(e)))
    return entries


def dump_entries(entries: list[dict[str, Any]]) -> str:
    """One entry per line: valid JSON, but diffable.

    ``json.dumps(indent=2)`` triples the file size by putting every shape and
    stride element on its own line; fully compact output is one 400 KB line that
    no diff tool can read. One entry per line splits the difference.
    """
    lines = [
        json.dumps(entry, sort_keys=True, separators=(",", ":")) for entry in entries
    ]
    return "[\n" + ",\n".join(lines) + "\n]\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--pytorch-root",
        default=None,
        help="Path to a PyTorch source checkout (default: $PYTORCH_ROOT).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Where to write the corpus (default: {DEFAULT_OUTPUT}).",
    )
    args = parser.parse_args()

    operator_inp_utils = import_operator_inp_utils(
        resolve_pytorch_root(args.pytorch_root)
    )
    entries = collect(operator_inp_utils)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(dump_entries(entries))
    print(f"wrote {len(entries)} entries to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
