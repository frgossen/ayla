"""Transfer this pass's successes to remaining deficits; bounded concept checks."""

import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from ayla.gemm_jit import _seed, _DEFAULTS  # noqa: E402 -- script path setup precedes local imports

ps = json.loads((HERE / "priorities.json").read_text())
leaders = json.loads((HERE / "leaders.json").read_text())
summaries = {
    r["rank"]: r for r in json.loads((HERE / "screen_summary.json").read_text())
}
tested = {
    j["target"]: {json.dumps(e["configuration"], sort_keys=True) for e in j["entries"]}
    for j in json.loads((HERE / "transfer_jobs.json").read_text())
}
jobs = []
for rank in (2, 4, 5, 7, 19, 23, 32):
    pr = next(p for p in ps if p["rank"] == rank)
    target = pr["target"]
    inc = pr["configuration"]
    best = next(r for r in leaders if r["target"] == target)
    base = best["configuration"] if best["conservative_gain"] > 1 else inc
    candidates = []
    seen = tested[target].copy()

    def add(c, label, source=None):
        c = {**_DEFAULTS, **c}
        key = json.dumps(c, sort_keys=True)
        if key in seen:
            return
        try:
            seed = _seed(
                c,
                pr["op"],
                pr["m"],
                pr["n"],
                pr["k"],
                (1, pr["m"], pr["n"], pr["k"]),
                exact=False,
            )
        except (AssertionError, KeyError, ValueError, TypeError):
            return
        if seed is None:
            return
        seen.add(key)
        candidates.append(
            dict(kind="followup_transfer", label=label, configuration=c, donor=source)
        )

    if rank in (4, 5, 7, 32):
        donor = next(r for r in leaders if "-m25216-n2304-" in r["target"])
        c = dict(donor["configuration"])
        c.update(unroll_output_tiles=False)
        add(c, "new-25216-winner:rolled", donor["target"])
        for group in (4, 8):
            c = dict(donor["configuration"])
            c.update(
                output_cta_count=None,
                output_tiles_per_cta=group,
                output_n_tiles_per_cta=1,
                unroll_output_tiles=True,
            )
            add(c, f"new-25216-winner:static-M{group}", donor["target"])
        # Same-N/K successes at moderate M suggest different output grouping.
        donor = next(
            r
            for r in leaders
            if ("-m32768-n768-k3072-" if rank == 5 else "-m32768-n3072-k768-")
            in r["target"]
        )
        for group in (4, 8):
            c = dict(donor["configuration"])
            c.update(output_tiles_per_cta=group)
            add(c, f"new-32768-winner:M{group}", donor["target"])
        for c, label in (
            ({**inc, "cyclic_operand_pipeline": True}, "incumbent:cyclic"),
            ({**inc, "output_store_buffers": 2}, "incumbent:double-store"),
            (
                {**inc, "operand_stages": 2, "operand_k_tile": 128},
                "incumbent:K128-two-stage",
            ),
            (
                {**inc, "output_m_tile": 64, "operand_stages": 4},
                "incumbent:M64-four-stage",
            ),
        ):
            add(c, label)
    if rank == 2:
        for donor_rank in (1, 12):
            donor = next(p for p in ps if p["rank"] == donor_rank)
            for mt, nt, wg, stages in (
                (64, 256, 256, 4),
                (256, 128, 128, 3),
                (128, 256, 128, 3),
            ):
                c = dict(donor["configuration"])
                c.update(
                    output_m_tile=mt,
                    output_n_tile=nt,
                    warpgroup_n_tile=wg,
                    operand_stages=stages,
                    output_tiles_per_cta=2,
                    output_n_tiles_per_cta=1,
                    output_slices=2,
                )
                add(c, f"wide-r{donor_rank}:M{mt}N{nt}WG{wg}", donor["target"])
        for group in (2, 4):
            for stages in (3, 4):
                c = dict(inc)
                c.update(
                    output_tiles_per_cta=group,
                    operand_stages=stages,
                    cyclic_operand_pipeline=True,
                    output_store_buffers=2,
                )
                add(c, f"wide-cyclic-double-store:Mgroup{group}:s{stages}")
    if rank == 19:
        # N is only 16-aligned. Use linear-layout donors and retain legal TMA strides.
        for mt, nt, stages, ng, mg in (
            (128, 208, 2, 2, 1),
            (128, 224, 2, 2, 1),
            (128, 240, 2, 2, 1),
            (128, 192, 3, 2, 1),
            (192, 144, 2, 2, 1),
            (256, 144, 2, 1, 1),
            (128, 160, 2, 4, 1),
            (128, 208, 3, 4, 1),
            (128, 224, 3, 2, 1),
            (128, 144, 2, 4, 1),
            (128, 144, 3, 2, 2),
            (192, 192, 2, 2, 1),
        ):
            c = dict(inc)
            c.update(
                output_m_tile=mt,
                output_n_tile=nt,
                warpgroup_n_tile=nt,
                operand_stages=stages,
                output_n_tiles_per_cta=ng,
                output_tiles_per_cta=mg,
                output_slices=2,
                output_tma_inner_size=None,
                cyclic_operand_pipeline=False,
            )
            add(c, f"linear-largeN:M{mt}N{nt}:s{stages}:g{mg}x{ng}")
    if rank == 23:
        donor = next(r for r in leaders if r["target"] == target)
        for stages in (2, 4):
            c = {**base, "operand_stages": stages}
            add(c, f"M256-donor:stage{stages}", donor["target"])
        for c, label in (
            ({**base, "output_store_buffers": 2}, "M256-donor:double-store"),
            ({**base, "cyclic_operand_pipeline": True}, "M256-donor:cyclic"),
            ({**base, "output_slices": 2}, "M256-donor:two-slices"),
            (
                {**base, "output_cta_count": None, "output_tiles_per_cta": 4},
                "M256-donor:static-M4",
            ),
            (
                {**base, "output_cta_count": None, "output_tiles_per_cta": 2},
                "M256-donor:static-M2",
            ),
        ):
            add(c, label, donor["target"])
        for donor_rank in (9, 10, 12):
            donor = next(p for p in ps if p["rank"] == donor_rank)
            c = dict(donor["configuration"])
            c.update(
                output_cta_count=None,
                output_tiles_per_cta=4,
                output_n_tiles_per_cta=1,
                unroll_output_tiles=True,
            )
            add(c, f"largeM-family:r{donor_rank}:static4", donor["target"])
    entries = [dict(kind="control", label="incumbent-start", configuration=inc)]
    for i, c in enumerate(candidates):
        entries.append(c)
        if (i + 1) % 6 == 0:
            entries.append(
                dict(kind="control", label=f"incumbent-{i}", configuration=inc)
            )
    entries.append(dict(kind="control", label="incumbent-end", configuration=inc))
    jobs.append(dict(target=target, entries=entries))
(HERE / "followup_jobs.json").write_text(json.dumps(jobs, indent=2) + "\n")
print(
    "follow-up",
    [(j["target"], sum(e["kind"] != "control" for e in j["entries"])) for j in jobs],
)
