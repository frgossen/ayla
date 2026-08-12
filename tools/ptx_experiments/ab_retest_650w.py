"""Re-run the previously power-masked PTX experiments at the raised power cap.

At 500 W the kernel was 97% power-capped and every schedule change measured
within +-0.5%.  At 650 W it is only ~22% capped, so instruction scheduling can
matter again.  This re-runs the same candidates against the same baseline:

  tailfuse   the retained `kk` tail unroll+fuse (vs pre-fix PTX on disk)
  storewait  bulk-store drains moved ahead of the stmatrix they guard, read 1
  wgdepth1   wgmma.wait_group 1 in the mainloop (INCORRECT; timing bound only)
"""
import dataclasses, re, statistics, sys, torch

from ayla.gemm_collection.auto_matmul_4096_wgmma256_wg_tma_store import (
    auto_matmul_4096_wgmma256_wg_tma_store_module as M)
from ayla.ptx_codegen import make_ptx_kernel
from ayla.ptx_runtime import make_ptx_launcher

CFG = dict(output_tiles_per_cta=4, operand_stages=3, output_slices=2,
           output_tma_inner_padding=16, output_store_buffers=2,
           operand_move_attrs={"lhs": {"l2_cache_hint": "evict_first"}})

_WAIT0 = re.compile(r"^\s*cp\.async\.bulk\.wait_group\.read 0;\s*$")
_STMATRIX_HDR = re.compile(r"^\s*// PTX template `stmatrix_store\(")
_BLOCK_BAR_HDR = re.compile(r"^\s*// PTX template `barrier\(\) attrs=\{scope: \"block\"\}`")
_BAR_SYNC0 = re.compile(r"^\s*bar\.sync 0;\s*$")
_AWAIT_HDR = re.compile(r"^\s*// PTX template `await\(%0: lock<bulk>\)")
_DRAIN = re.compile(r"^(\s*)wgmma\.wait_group\.sync\.aligned 0;\s*$")
_CONVERT_HDR = re.compile(r"^\s*// PTX template `convert\(")


def storewait(ptx: str) -> str:
    lines = ptx.split("\n")
    wait_idx = [i for i, l in enumerate(lines) if _WAIT0.match(l)]
    stmat_idx = [i for i, l in enumerate(lines) if _STMATRIX_HDR.match(l)]
    drop = set()
    for i in wait_idx[:-1]:
        drop.add(i)
        for j in (i - 1, i - 2):
            if j >= 0 and (_AWAIT_HDR.match(lines[j]) or not lines[j].strip()):
                drop.add(j)
    insert: dict[int, list[str]] = {}
    for i in stmat_idx[2:]:
        anchor = i
        for j in range(i - 1, max(i - 6, -1), -1):
            if _BAR_SYNC0.match(lines[j]):
                anchor = j - 1 if _BLOCK_BAR_HDR.match(lines[j - 1]) else j
                break
        else:
            insert.setdefault(anchor, []).append("  bar.sync 0;")
        insert.setdefault(anchor, []).insert(0, "  cp.async.bulk.wait_group.read 1;")
    out = []
    for i, line in enumerate(lines):
        out.extend(insert.get(i, ()))
        if i not in drop:
            out.append(line)
    return "\n".join(out)


def wgdepth(ptx: str, depth: int) -> str:
    lines = ptx.split("\n")
    for i, line in enumerate(lines):
        m = _DRAIN.match(line)
        if m and not any(_CONVERT_HDR.match(lines[j])
                         for j in range(i + 1, min(i + 6, len(lines)))):
            lines[i] = f"{m.group(1)}wgmma.wait_group.sync.aligned {depth};"
    return "\n".join(lines)


def _burst(launcher, a, b, c, iters):
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        launcher(a, b, c)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def paired(base, new, a, b, cb, cn, rounds=140, iters=20, warmup=3000):
    for _ in range(warmup):
        base(a, b, cb); new(a, b, cn)
    torch.cuda.synchronize()
    tb, tn, ratio = [], [], []
    for i in range(rounds):
        if i % 2 == 0:
            x = _burst(base, a, b, cb, iters); y = _burst(new, a, b, cn, iters)
        else:
            y = _burst(new, a, b, cn, iters); x = _burst(base, a, b, cb, iters)
        tb.append(x); tn.append(y); ratio.append(y / x)
    return tb, tn, ratio


torch.manual_seed(0)
a = torch.randn(4096, 4096, dtype=torch.bfloat16, device="cuda")
b = torch.randn(4096, 4096, dtype=torch.bfloat16, device="cuda")
ref = (a.float() @ b.float().t()).bfloat16()

kernel = make_ptx_kernel(M(**CFG))
base = make_ptx_launcher(kernel)
cb = torch.empty_like(ref)
base(a, b, cb); torch.cuda.synchronize()

CASES = [("storewait", storewait(kernel.ptx))]

print(f"{'case':15s} {'median us':>10s} {'base us':>9s} {'delta':>8s} {'90% band':>18s} {'mismatch':>9s}")
for name, ptx in CASES:
    new = make_ptx_launcher(dataclasses.replace(kernel, ptx=ptx))
    cn = torch.empty_like(ref)
    new(a, b, cn); torch.cuda.synchronize()
    mism = (cb != cn).sum().item()
    tb, tn, ratio = paired(base, new, a, b, cb, cn)
    rs = sorted(ratio)
    med, lo, hi = statistics.median(rs), rs[int(0.05 * len(rs))], rs[int(0.95 * len(rs)) - 1]
    print(f"{name:15s} {statistics.median(tn)*1000:10.3f} {statistics.median(tb)*1000:9.3f} "
          f"{(1-med)*100:+7.2f}% {(1-hi)*100:+7.2f}..{(1-lo)*100:+.2f}% {mism:9d}")
