# Ayla fused Softplus final report

## Kernel

- Candidate: `submissions/softplus.py` (`ayla_fused_softplus_i4`)
- Launch: grid `(8, 16, 1)`, block `(256, 1, 1)`; 128 CTAs
- GEMM: four 128x256 output tiles per CTA, three 64-wide operand stages,
  two output slices and two padded TMA store buffers
- Epilogue: stable-threshold FP32 Softplus in WGMMA accumulator registers;
  `ex2` is predicated off when `x / tau > 20`, then results are packed to
  BF16 and written through the existing shared-memory/TMA store pipeline
- Intermediate HBM traffic and epilogue launches: zero

## Correctness

Common `benchmark/run.py` check against seed-42 PyTorch reference:

- max absolute error: `0.0625`
- mean absolute error: `0.0001761427`
- exact BF16 fraction: `0.5877150`

## Paired timing

Both final runs used 300 warm-ups, 20 launches per event sample, 60 paired
samples, and 4,000 bootstrap resamples.

| Regime | PyTorch median | Ayla median | Paired delta (95% CI) | Wins |
| --- | ---: | ---: | ---: | ---: |
| Free clock | 447.232 us | 216.751 us | -51.520% (-51.712%, -51.205%) | 60/60 |
| Fixed 1050 MHz | 588.002 us | 286.901 us | -51.273% (-51.513%, -51.049%) | 60/60 |

The fixed-clock run drew 297 W under the unchanged 500 W limit. GPU clocks
were explicitly restored after the run.

## PTXAS and SASS

- registers/thread: `154`
- spills: `0` loads, `0` stores; stack frame `0` bytes
- static shared memory: `221,336` bytes
- barriers: `6` PTXAS resources
- SASS includes `HGMMA.64x256x16.F32.BF16`, predicated `MUFU.EX2` and
  `MUFU.LG2`, `UTMALDG`, `STAS`, and `UTMASTG`

## Nsight Compute

Profiler duration is replay-instrumented and therefore differs from the paired
CUDA-event timing.

- duration: `248.10 us`
- Compute (SM) throughput: `84.71%`
- memory throughput: `54.29%`; DRAM throughput: `16.81%`
- L2 hit rate: `80.52%`; measured memory traffic: `411.12 GB/s`
- active / eligible warps per scheduler: `2.00 / 0.12`
- issued warps per scheduler: `0.11`; no-eligible cycles: `88.59%`
- waves per SM: `0.97`; local-memory spill requests: `0`

## Artifacts

- `softplus_final_free.json`, `softplus_final_fixed1050.json`
- `softplus_final.ptx`, `softplus_final.cubin`, `softplus_final.sass`
- `softplus_final_ptxas.log`
- `softplus_final_profile.ncu-rep`, `softplus_final_profile_details.txt`
- `softplus_interleave_sweep.json`, `softplus_pipeline_sweep.json`
