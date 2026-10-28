"""Freeze the requested priorities and donor schedules before GPU work."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from ayla.gemm_jit import CONFIGS, SCHEDULES, _DEFAULTS, select_gemm_schedule  # noqa: E402 -- script path setup precedes local imports
from ayla.gemm_heuristics import expand_gemm_heuristic  # noqa: E402 -- script path setup precedes local imports
from benchmarks.oss_gemm_corpus import load_shapes, validate_shapes, benchmark_shapes  # noqa: E402 -- script path setup precedes local imports

attachment = Path(
    "/home/arseniivanov/.codex/attachments/f2908b88-4d0e-4419-8d24-f28abdf612d0/pasted-text-1.txt"
)
shapes = benchmark_shapes(validate_shapes(load_shapes(HERE / "corpus.json")))
priorities = []
for line in attachment.read_text().splitlines():
    fields = [x.strip() for x in line.split("│")][1:-1]
    if not fields or not fields[0].isdigit():
        continue
    rank, op, m, n, k, calls, torch_us, ayla_us, pct, share, cumulative = fields
    m, n, k = map(int, (m, n, k))
    matches = [
        s for s in shapes if (s.op, s.dtype, s.m, s.n, s.k) == (op, "bfloat16", m, n, k)
    ]
    assert len(matches) == 1, (fields, len(matches))
    shape = matches[0]
    seed = select_gemm_schedule(op, "bfloat16", m, n, k)
    assert seed and seed.builder_config
    priorities.append(
        dict(
            rank=int(rank),
            target=shape.shape_id,
            op=op,
            dtype="bfloat16",
            m=m,
            n=n,
            k=k,
            calls=int(calls),
            historical_torch_us=float(torch_us),
            historical_ayla_us=float(ayla_us),
            historical_pct=float(pct.rstrip("%")),
            share_pct=float(share.rstrip("%")),
            configuration=seed.captured_config,
            selected_source=seed.source_shape,
        )
    )
assert len(priorities) == 35
(HERE / "priorities.json").write_text(json.dumps(priorities, indent=2) + "\n")
donors = []
for (op, dtype), table in SCHEDULES.items():
    if op not in ("mm", "addmm"):
        continue
    for (batch, m, n, k), indices in table.items():
        if batch != 1 or m < 1000 or n < 128 or k < 128:
            continue
        cfg = {**_DEFAULTS, **CONFIGS[indices[0]]}
        donors.append(
            dict(
                backend="current_jit",
                op=op,
                dtype=dtype,
                m=m,
                n=n,
                k=k,
                configuration=cfg,
                table_index=indices[0],
            )
        )
legacy_path = Path(
    "/home/arseniivanov/ayla-pr-verification/ayla/oss_gemm_heuristic.json"
)
legacy = expand_gemm_heuristic(json.loads(legacy_path.read_text()))
for row in legacy["exact"]:
    if row["operation"] not in ("mm", "addmm") or row["m"] < 4096 or row["n"] < 128:
        continue
    score = row["seed"]["source_percentage_of_torch"]
    if score is None or score < 102:
        continue
    donors.append(
        dict(
            backend="legacy_native_hypothesis",
            op=row["operation"],
            dtype=row["dtype"],
            m=row["m"],
            n=row["n"],
            k=row["k"],
            donor_pct=score,
            configuration=row["seed"]["configuration"],
        )
    )
(HERE / "donors.json").write_text(json.dumps(donors, indent=2) + "\n")
meta = dict(
    head=subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip(),
    branch=subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=ROOT, text=True
    ).strip(),
    priorities=len(priorities),
    donors=len(donors),
    priority_attachment_sha256=hashlib.sha256(attachment.read_bytes()).hexdigest(),
    legacy_donor_sha256=hashlib.sha256(legacy_path.read_bytes()).hexdigest(),
    sources={
        p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
        for p in (
            "ayla/gemm_jit.py",
            "ayla/gemm_heuristics.py",
            "ayla/gemm_schedule_table.py",
            "benchmarks/common.py",
            "ayla/gemm_collection/gemm.py",
        )
    },
)
(HERE / "source.json").write_text(json.dumps(meta, indent=2) + "\n")
jobs = [
    dict(
        target=p["target"],
        entries=[
            dict(kind="control", label="incumbent", configuration=p["configuration"])
        ],
    )
    for p in priorities
]
(HERE / "baseline_jobs.json").write_text(json.dumps(jobs, indent=2) + "\n")
print(json.dumps(meta, indent=2))
