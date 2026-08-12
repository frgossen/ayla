"""Measure time, sustained SM clock, power and energy per launch for each GEMM.

The H100 in this box runs this workload pinned at its 500 W cap with
`sw_power_cap` active and SM clock held around 1.2 GHz (boost is 1.98 GHz), so
throughput is set by the power budget rather than by the schedule.  Under that
regime the figure of merit is energy per launch: any joule not spent moving
data is a joule available to raise the clock.

Samples nvidia-smi in a helper thread while each variant runs back to back.
"""
from __future__ import annotations

import statistics
import subprocess
import sys
import threading
import time

import torch

import ayla.gemm_collection  # noqa: F401 — registers torch.ops.ayla.*

VARIANTS = [
    "k3", "k3_2tiles", "k3_2tiles_split2_dbuf2", "k3_4tiles_split2_dbuf2",
    "k4_2tiles_split2", "k2_ktile128_2tiles_split2", "k2_n384_split3",
    "k3_m192_split2",
]
PREFIX = "auto_matmul_4096_wgmma256_wg_tma_store_"
FLOP = 2 * 4096 ** 3
SECONDS = float(sys.argv[1]) if len(sys.argv) > 1 else 12.0

_QUERY = ("clocks.sm,power.draw,temperature.gpu,"
          "clocks_throttle_reasons.sw_power_cap")


class Sampler(threading.Thread):
    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.stop = threading.Event()
        self.clk: list[float] = []
        self.pwr: list[float] = []
        self.temp: list[float] = []
        self.capped = 0
        self.n = 0

    def run(self) -> None:
        while not self.stop.is_set():
            out = subprocess.run(
                ["nvidia-smi", "-i", "0", f"--query-gpu={_QUERY}",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True).stdout.strip()
            if out:
                c, p, t, cap = [f.strip() for f in out.split(",")]
                self.clk.append(float(c))
                self.pwr.append(float(p))
                self.temp.append(float(t))
                self.capped += cap == "Active"
                self.n += 1
            time.sleep(0.25)


def measure(op, a, b, seconds: float) -> dict:
    for _ in range(50):
        op(a, b)
    torch.cuda.synchronize()
    s = Sampler()
    s.start()
    time.sleep(1.0)  # let the sampler see steady state before timing
    launches, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        for _ in range(100):
            op(a, b)
        torch.cuda.synchronize()
        launches += 100
    dt = time.perf_counter() - t0
    s.stop.set()
    s.join()
    per = dt / launches
    pwr = statistics.median(s.pwr)
    return dict(us=per * 1e6, tflops=FLOP / per / 1e12, clk=statistics.median(s.clk),
                pwr=pwr, temp=max(s.temp), cap=s.capped / max(s.n, 1),
                mj=pwr * per * 1e3)


torch.manual_seed(0)
a = torch.randn(4096, 4096, dtype=torch.bfloat16, device="cuda")
b = torch.randn(4096, 4096, dtype=torch.bfloat16, device="cuda")

rows = []
for name in VARIANTS:
    op = getattr(torch.ops.ayla, PREFIX + name)
    rows.append((name, measure(lambda x, y: op(x, y), a, b, SECONDS)))
rows.append(("cublas torch.mm", measure(lambda x, y: torch.mm(x, y.t()), a, b, SECONDS)))

print(f"{'variant':30s} {'us':>8s} {'TFLOP/s':>8s} {'SM MHz':>7s} {'W':>7s} "
      f"{'mJ/launch':>10s} {'degC':>5s} {'capped':>7s}")
for name, r in rows:
    print(f"{name:30s} {r['us']:8.1f} {r['tflops']:8.1f} {r['clk']:7.0f} "
          f"{r['pwr']:7.1f} {r['mj']:10.2f} {r['temp']:5.0f} {r['cap']*100:6.0f}%")
