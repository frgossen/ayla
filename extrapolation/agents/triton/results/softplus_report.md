# Triton Softplus result

## Candidate

- Module: `agents/triton/submissions/softplus_fused.py`
- Kernel: persistent Hopper TMA GEMM with a fused FP32 stable Softplus epilogue
- Tile: 128x128x64, 8 warps, 6 stages, group-M 8
- Grid/block: 132 CTAs x 256 threads (one persistent CTA per SM)
- Epilogue: `tau * (max(z, 0) + log(1 + exp(-abs(z))))`, `z = x/tau`
- Output: BF16, stored directly through a TMA tensor descriptor

The symmetric formulation never evaluates a positive exponential. For
`z > 20`, `exp(-z)` is too small to change FP32 `1.0`, so the result rounds to
`x` exactly and matches the canonical thresholded Softplus branch. The
canonical `tau=1.0` value is specialized at compile time. Compilation and the
first launch occur in `prepare()`.

## Correctness

Canonical seed-42 comparison against PyTorch:

- max absolute error: 0.0625
- mean absolute error: 0.0001761650
- exact BF16 fraction: 0.5876860

The candidate passes `benchmark/problems.py` (`atol=2.0`, `rtol=0.02`).

## Paired benchmark

Both final runs used 300 warm-ups, 20 launches per event sample, 60 alternating
pairs, and 4000 bootstrap repetitions on idle physical GPU 6. Negative delta
means the candidate is faster than `references/torch_softplus.py`.

| Regime | Baseline | Candidate | Paired delta (95% CI) | Wins |
| --- | ---: | ---: | ---: | ---: |
| Free clock | 452.951 us | 257.818 us | -43.108% ([-43.382%, -42.903%]) | 60/60 |
| Fixed 1050 MHz | 582.035 us | 332.525 us | -42.983% ([-43.108%, -42.732%]) | 60/60 |

The fixed-clock run drew 313.62 W under the 500 W limit. A restoration trap was
installed before locking clocks, clocks were explicitly reset afterward, and
the post-reset observed SM clock was 1965 MHz.

## Resource use and NCU

Nsight Compute profiled one launch at free clocks:

- duration: 283.072 us (profiling replay measurement)
- registers: 98 per thread
- local spilling requests: 0
- dynamic shared memory: 229.424 KB per CTA (230.448 KB allocated including driver shared memory)
- compute throughput: 73.61% of peak
- memory throughput: 55.85% of peak
- DRAM throughput: 29.52% of peak
- L2 hit rate: 81.81%
- issued warps per scheduler: 0.42; active warps per scheduler: 2.00

The actual SASS contains `HGMMA.64x128x16.F32.BF16`, twelve `UTMALDG.2D`
instructions, one `UTMASTG.2D`, 64 `MUFU.EX2` instructions, warp-group arrive
and dependency barriers, and no spill traffic. The TMA store is asynchronous
with respect to subsequent persistent-loop work.

## Artifacts

- `softplus_free_clock.json`
- `softplus_1050mhz.json`
- `softplus_profile.ncu-rep`
- `softplus_sass.txt`
- `softplus_tune_*.json` and `softplus_epilogue_*.json`
