"""A/B the CuTe/Triton idioms that survived static screening, against Ayla.

Paired per-batch interleaving so drift cancels; see ab_shadow.py for the
methodology.  Run in both regimes:

    sudo nvidia-smi -i 0 -lgc 1350     # then again after -rgc
    CUDA_VISIBLE_DEVICES=0 .pixi/envs/default/bin/python \
        tools/ptx_experiments/ab_compiler_transfer.py [variant ...]
"""
import dataclasses
import random
import statistics
import sys

import torch

sys.path.insert(0, "/home/arseniivanov/ayla/tools/ptx_experiments")
from compiler_transfer import transform as xform  # noqa: E402

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
NAMES = ["prefetch", "storehint", "prefetch+storehint", "rhshint"]
FLOP = 2 * 4096 ** 3
BATCH, BATCHES, RAMP = 20, 60, 400


def build(spec=None):
    config = BEST
    if spec in ("rhs_last", "storehint+rhs_last"):
        config = {
            **BEST,
            "operand_move_attrs": {
                "lhs": {"l2_cache_hint": "evict_first"},
                "rhs": {"l2_cache_hint": "evict_last"},
                **({"dst": {"l2_cache_hint": "evict_first"}}
                   if spec == "storehint+rhs_last" else {}),
            },
        }
    k = make_ptx_kernel(M(**config))
    if spec in ("rhs_last", "storehint+rhs_last"):
        return make_ptx_launcher(k)
    if spec:
        ptx, n = xform(k.ptx, spec)
        assert n > 0, f"{spec} matched nothing"
        k = dataclasses.replace(k, ptx=ptx)
    return make_ptx_launcher(k)


def batch(f, a, b, c):
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(BATCH):
        f(a, b, c)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / BATCH * 1e3


def ci(d, reps=4000):
    rng = random.Random(0)
    m = sorted(statistics.median(rng.choices(d, k=len(d))) for _ in range(reps))
    return m[reps // 40], m[reps - reps // 40]


def main():
    names = sys.argv[1:] or NAMES
    torch.manual_seed(0)
    a = torch.randn(4096, 4096, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(4096, 4096, dtype=torch.bfloat16, device="cuda")
    ref = (a.float() @ b.float().t()).bfloat16()
    base = build()
    arms = {n: build(n) for n in names}

    cb = torch.empty_like(ref)
    base(a, b, cb)
    torch.cuda.synchronize()
    print(f"baseline maxerr {(cb.float()-ref.float()).abs().max().item()}")
    for n, arm in arms.items():
        cv = torch.empty_like(ref)
        arm(a, b, cv)
        torch.cuda.synchronize()
        mm = (cb != cv).sum().item()
        print(f"  {n:20s} {'bitwise identical' if mm == 0 else f'{mm} MISMATCHES'}")

    c = torch.empty_like(ref)
    for _, arm in [("b", base), *arms.items()]:
        for _ in range(RAMP // (len(arms) + 1)):
            arm(a, b, c)
    torch.cuda.synchronize()

    for n, arm in arms.items():
        tb, tv = [], []
        for i in range(BATCHES):
            if i % 2 == 0:
                tb.append(batch(base, a, b, c)); tv.append(batch(arm, a, b, c))
            else:
                tv.append(batch(arm, a, b, c)); tb.append(batch(base, a, b, c))
        d = [100 * (v - x) / x for x, v in zip(tb, tv)]
        lo, hi = ci(d)
        mb, mv = statistics.median(tb), statistics.median(tv)
        print(f"\n{n}\n  baseline {mb:8.3f} us  variant {mv:8.3f} us  "
              f"({FLOP/(mv*1e-6)/1e12:.1f} TFLOP/s)")
        print(f"  paired {statistics.median(d):+.3f}%  95% CI "
              f"[{lo:+.3f}%, {hi:+.3f}%]  faster in "
              f"{sum(1 for x in d if x < 0)}/{len(d)}")


if __name__ == "__main__":
    main()
