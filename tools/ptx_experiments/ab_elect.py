"""A/B: baseline PTX vs elect.sync-rewritten PTX, correctness + interleaved timing."""
import dataclasses, statistics, sys, torch
sys.path.insert(0, "/tmp")
from elect_sync import transform as elect_xform
from ayla.gemm_collection.auto_matmul_4096_wgmma256_wg_tma_store import (
    auto_matmul_4096_wgmma256_wg_tma_store_module as M)
from ayla.ptx_codegen import make_ptx_kernel
from ayla.ptx_runtime import make_ptx_launcher

CFG = {
 "k3_2t":     dict(output_tiles_per_cta=2, operand_stages=3),
 "k3_split2": dict(output_tiles_per_cta=2, operand_stages=3, output_slices=2,
                   output_tma_inner_padding=16, output_store_buffers=2),
 "k4_split2": dict(output_tiles_per_cta=2, operand_stages=4, output_slices=2),
}

def build(cfg, patch):
    k = make_ptx_kernel(M(**cfg))
    if patch:
        ptx, n = elect_xform(k.ptx)
        assert n > 0, "no guards rewritten"
        k = dataclasses.replace(k, ptx=ptx)
    return make_ptx_launcher(k)

def run(launcher, a, b):
    c = torch.empty(4096, 4096, dtype=torch.bfloat16, device="cuda")
    launcher(a, b, c); return c

def time_it(launcher, a, b, c, rounds=40, iters=20):
    for _ in range(5): launcher(a, b, c)
    torch.cuda.synchronize()
    ts = []
    for _ in range(rounds):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(iters): launcher(a, b, c)
        e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) / iters)
    return ts

torch.manual_seed(0)
a = torch.randn(4096, 4096, dtype=torch.bfloat16, device="cuda")
b = torch.randn(4096, 4096, dtype=torch.bfloat16, device="cuda")
ref = (a.float() @ b.float().t()).bfloat16()

print(f"{'config':14s} {'variant':9s} {'median us':>10s} {'min us':>9s} {'stdev':>7s}  {'maxerr':>9s}")
for name, cfg in CFG.items():
    base = build(cfg, False); ele = build(cfg, True)
    cb, ce = torch.empty_like(ref), torch.empty_like(ref)
    base(a, b, cb); ele(a, b, ce); torch.cuda.synchronize()
    eb = (cb.float()-ref.float()).abs().max().item()
    ee = (ce.float()-ref.float()).abs().max().item()
    mism = (cb != ce).sum().item()
    # interleaved A/B/B/A
    tb, te = [], []
    for i in range(4):
        if i % 2 == 0:
            tb += time_it(base, a, b, cb); te += time_it(ele, a, b, ce)
        else:
            te += time_it(ele, a, b, ce); tb += time_it(base, a, b, cb)
    for tag, t, err in (("baseline", tb, eb), ("elect", te, ee)):
        print(f"{name:14s} {tag:9s} {statistics.median(t)*1000:10.2f} {min(t)*1000:9.2f} "
              f"{statistics.stdev(t)*1000:7.2f}  {err:9.4f}")
    d = (statistics.median(tb)-statistics.median(te))/statistics.median(tb)*100
    print(f"{'':14s} {'-> delta':9s} {d:+9.2f}%   bitwise-mismatch elems: {mism}")
