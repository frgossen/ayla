"""A/B the drain-shadow and store-pipeline PTX rewrites against the baseline.

Methodology: all launchers are built and ramped *before* any timing, then the
arms are interleaved at single-batch granularity (A,B,A,B,...) and compared
pairwise, so slow drift (thermal, clock, or power controller) cancels instead
of landing on whichever arm happened to run first.  Reports the median paired
delta with a bootstrap CI and a sign test.

Run it in both regimes -- the rewrite must win in both to count:

    sudo nvidia-smi -i 0 -lgc 1350       # high-SNR fixed-clock test
    sudo nvidia-smi -i 0 -rgc            # shipping regime, 500 W power-capped

    CUDA_VISIBLE_DEVICES=0 .pixi/envs/default/bin/python \
        tools/ptx_experiments/ab_shadow.py [variant ...]
"""
import dataclasses
import random
import statistics
import sys

import torch

sys.path.insert(0, "/home/arseniivanov/ayla/tools/ptx_experiments")
from shadow_fill import transform as shadow_xform  # noqa: E402
from store_pipeline import transform as store_xform  # noqa: E402

from ayla.gemm_collection.auto_matmul_4096_wgmma256_wg_tma_store import (  # noqa: E402
    auto_matmul_4096_wgmma256_wg_tma_store_module as M,
)
from ayla.ptx_codegen import make_ptx_kernel  # noqa: E402
from ayla.ptx_runtime import make_ptx_launcher  # noqa: E402

BEST = dict(
    output_tiles_per_cta=4, operand_stages=3, output_slices=2,
    output_tma_inner_padding=16, output_store_buffers=2,
    operand_move_attrs={"lhs": {"l2_cache_hint": "evict_first"}},
)

VARIANTS = {
    "mbar": lambda p: shadow_xform(p, "mbar"),
    "mbar+desc": lambda p: shadow_xform(p, "mbar+desc"),
    "store": lambda p: store_xform(p),
    "mbar+store": lambda p: store_xform(shadow_xform(p, "mbar")[0]),
}

FLOP = 2 * 4096 ** 3
BATCH = 20        # launches per timed batch
BATCHES = 60      # paired batches per arm
RAMP = 400        # launches before any timing, to reach steady state


def build(patch=None):
    kernel = make_ptx_kernel(M(**BEST))
    if patch is not None:
        ptx, n = patch(kernel.ptx)
        assert n > 0, "patch matched nothing"
        kernel = dataclasses.replace(kernel, ptx=ptx)
    return make_ptx_launcher(kernel)


def batch(launcher, a, b, c):
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(BATCH):
        launcher(a, b, c)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / BATCH * 1e3  # us


def bootstrap_ci(deltas, reps=4000, lo=2.5, hi=97.5):
    rng = random.Random(0)
    n = len(deltas)
    meds = sorted(
        statistics.median(rng.choices(deltas, k=n)) for _ in range(reps)
    )
    return meds[int(reps * lo / 100)], meds[int(reps * hi / 100)]


def main():
    names = sys.argv[1:] or list(VARIANTS)
    torch.manual_seed(0)
    a = torch.randn(4096, 4096, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(4096, 4096, dtype=torch.bfloat16, device="cuda")
    ref = (a.float() @ b.float().t()).bfloat16()

    base = build()
    arms = {n: build(VARIANTS[n]) for n in names}

    cb = torch.empty_like(ref)
    base(a, b, cb)
    torch.cuda.synchronize()
    print(f"baseline maxerr vs fp32 ref: "
          f"{(cb.float() - ref.float()).abs().max().item()}")
    for name, arm in arms.items():
        cv = torch.empty_like(ref)
        arm(a, b, cv)
        torch.cuda.synchronize()
        n_mismatch = (cb != cv).sum().item()
        status = "bitwise identical" if n_mismatch == 0 else f"{n_mismatch} MISMATCHES"
        print(f"  {name:12s} vs baseline: {status}")

    c = torch.empty_like(ref)
    for name, arm in [("base", base), *arms.items()]:
        for _ in range(RAMP // len(arms) // 2):
            arm(a, b, c)
    torch.cuda.synchronize()

    for name, arm in arms.items():
        tb, tv = [], []
        for i in range(BATCHES):
            # Alternate which arm leads so any within-pair bias cancels too.
            if i % 2 == 0:
                tb.append(batch(base, a, b, c))
                tv.append(batch(arm, a, b, c))
            else:
                tv.append(batch(arm, a, b, c))
                tb.append(batch(base, a, b, c))
        deltas = [100 * (v - x) / x for x, v in zip(tb, tv)]
        med = statistics.median(deltas)
        lo, hi = bootstrap_ci(deltas)
        wins = sum(1 for d in deltas if d < 0)
        mb, mv = statistics.median(tb), statistics.median(tv)
        print(f"\n{name}")
        print(f"  baseline  {mb:8.3f} us  sd {statistics.stdev(tb):5.3f}  "
              f"{FLOP / (mb * 1e-6) / 1e12:6.1f} TFLOP/s")
        print(f"  variant   {mv:8.3f} us  sd {statistics.stdev(tv):5.3f}  "
              f"{FLOP / (mv * 1e-6) / 1e12:6.1f} TFLOP/s")
        print(f"  paired delta {med:+.3f}%  95% CI [{lo:+.3f}%, {hi:+.3f}%]  "
              f"variant faster in {wins}/{len(deltas)} pairs")


if __name__ == "__main__":
    main()
