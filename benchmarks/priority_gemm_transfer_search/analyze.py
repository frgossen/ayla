"""Rank transfers by both raw Ayla time and paired Torch-normalized gain."""

import argparse
import json
import math
from pathlib import Path
import statistics as st

HERE = Path(__file__).resolve().parent
p = argparse.ArgumentParser()
p.add_argument("--phases", nargs="+", default=["transfer"])
a = p.parse_args()
priorities = json.loads((HERE / "priorities.json").read_text())
baseline = {
    r["target"]: r
    for path in (HERE / "baseline").glob("[0-9][0-9][0-9][0-9].json")
    for r in json.loads(path.read_text())
    if r["status"] == "ok"
}
all_rows = []
ranked = {}
for phase in a.phases:
    for path in (HERE / phase).glob("[0-9][0-9][0-9][0-9].json"):
        rows = json.loads(path.read_text())
        all_rows.extend(rows)
        controls = [
            (i, r)
            for i, r in enumerate(rows)
            if r["status"] == "ok" and r["entry"]["kind"] == "control"
        ]
        if not controls:
            continue
        for i, r in enumerate(rows):
            if r["status"] != "ok" or r["entry"]["kind"] == "control":
                continue
            nearby = sorted(controls, key=lambda ir: abs(ir[0] - i))[:2]
            control_ms = st.median(x["ayla_ms"] for _, x in nearby)
            control_pct = st.median(x["torch_pct"] for _, x in nearby)
            gain = min(control_ms / r["ayla_ms"], r["torch_pct"] / control_pct)
            if any(r["code_fingerprint"] == x["code_fingerprint"] for _, x in controls):
                gain = 1
            r = dict(
                r,
                conservative_gain=gain,
                control_ms=control_ms,
                control_pct=control_pct,
                screen_phase=phase,
                screen_file=str(path.relative_to(HERE)),
            )
            ranked.setdefault(r["target"], []).append(r)
leaders = []
for target, rows in ranked.items():
    seen = set()
    for r in sorted(rows, key=lambda r: r["conservative_gain"], reverse=True):
        if r["code_fingerprint"] in seen:
            continue
        seen.add(r["code_fingerprint"])
        if len(seen) <= 3:
            leaders.append(r)
(HERE / "leaders.json").write_text(json.dumps(leaders, indent=2) + "\n")
summary = []
for pr in priorities:
    rows = [r for r in leaders if r["target"] == pr["target"]]
    best = max(rows, key=lambda r: r["conservative_gain"]) if rows else None
    summary.append(
        dict(
            rank=pr["rank"],
            target=pr["target"],
            baseline_pct=baseline[pr["target"]]["torch_pct"],
            best_pct=best["torch_pct"] if best else None,
            gain=best["conservative_gain"] if best else None,
            label=best["entry"]["label"] if best else None,
            trials=sum(
                r["target"] == pr["target"] and r["entry"]["kind"] != "control"
                for r in all_rows
            ),
        )
    )
(HERE / "screen_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def gm(xs):
    return math.exp(st.mean(math.log(x) for x in xs))


print("BASELINE 35-shape geomean", gm([r["torch_pct"] for r in baseline.values()]))
print(
    "BASELINE 35-shape call-weighted speed",
    100
    * sum(pr["calls"] * baseline[pr["target"]]["torch_ms"] for pr in priorities)
    / sum(pr["calls"] * baseline[pr["target"]]["ayla_ms"] for pr in priorities),
)
print("trials", len(all_rows), "errors", sum(r["status"] != "ok" for r in all_rows))
for r in summary:
    print(
        r["rank"],
        r["target"],
        f"baseline {r['baseline_pct']:.2f}",
        f"gain {100 * (r['gain'] - 1):.2f}% best {r['best_pct']:.2f}"
        if r["gain"]
        else "",
        r["trials"],
        r["label"] or "",
    )
