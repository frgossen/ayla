# Triton power-specific autotuning and sweep

GPU 3 (NVIDIA H100) was idle before the experiment. Each native Triton
autotune used its own initially empty `TRITON_CACHE_DIR`, the official Hopper
TMA-persistent tutorial search space (48 configurations), and BF16
`4096 x 4096 x 4096` GEMM. The GPU was restored to its 650 W default afterward.

## Native autotune results

| Tune cap | Winner (BM x BN x BK) | Stages | Warps | Split epilogue | Autotuner median |
|---:|---:|---:|---:|:---:|---:|
| 350 W | 128 x 256 x 64 | 3 | 8 | yes | 254.05 us |
| 500 W | 128 x 256 x 64 | 4 | 8 | yes | 216.61 us |
| 650 W | 128 x 256 x 64 | 4 | 8 | yes | 193.02 us |

The core tile and warp count match the previously pinned Triton kernel. The
differences are epilogue subtiling at every tune cap and one extra pipeline
stage at 500/650 W. The fresh 650 W tune selected the same policy as 500 W.

The plotted `Triton current (@ 650 W)` line intentionally remains the existing
pinned policy (3 stages, 8 warps, no split epilogue), so it is distinct from
the freshly tuned policies.

## Sustained sweep (TFLOP/s)

| Policy | 350 W | 450 W | 550 W | 650 W |
|---|---:|---:|---:|---:|
| Triton tuned @ 350 W | 351.4 | 532.5 | 593.4 | 617.6 |
| Triton tuned @ 500 W | 338.2 | 533.4 | 596.8 | 625.9 |
| Triton current (@ 650 W) | 334.4 | 534.6 | 591.1 | 618.6 |

The 450/550/650 W points use two alternating forward/reverse passes, 8 seconds
per measurement. Because 350 W showed strong power-governor hysteresis, its
final plotted values use one isolated process per policy, a 5-second ramp, and
a 20-second measurement. The shared-process 350 W values remain in the main
CSV as diagnostic evidence; the isolated CSVs are under `isolated-350/`.

The updated plots combine these Triton measurements with the prior cuBLAS,
Ayla, and CuTe DSL sweep from 2026-08-10. Those other frameworks were not
rerun simultaneously with this policy comparison.

Raw autotuner results and clean caches are in
`generated_ptx/triton_power_autotune/`. The policy sweep is in
`power_efficiency.csv`; final plots are in the adjacent
`power-sweep-h100-gpu3-2026-08-10/` directory.
