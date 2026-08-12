# CuTe DSL fused Softplus result

Candidate: `agents/cutlass/submissions/softplus_cute.py`

The selected Hopper configuration is a persistent 128x256 CTA tile, 1x1
cluster, swizzle 1, and M raster. The FP32 accumulator fragment is transformed
in registers with stable Softplus and converted once to BF16 before the staged
TMA store. The exponential input is capped at the threshold before evaluation.

## Correctness

- Maximum absolute error: 0.0625
- Mean absolute error: 0.0001761427
- Exact BF16 fraction: 0.5877150

## Paired benchmark

All runs used 300 warm-ups, 20 launches per sample, 60 paired samples, and
4,000 bootstrap repetitions against `references/torch_softplus.py`.

| Regime | Candidate | Baseline | Paired delta | 95% CI | Wins |
| --- | ---: | ---: | ---: | ---: | ---: |
| Free clock A | 218.010 us | 452.582 us | -51.558% | [-51.713%, -51.369%] | 60/60 |
| Free clock B | 218.967 us | 455.462 us | -51.623% | [-51.781%, -51.438%] | 60/60 |
| 1050 MHz A | 284.978 us | 587.666 us | -51.569% | [-51.690%, -51.402%] | 60/60 |
| 1050 MHz B | 284.577 us | 587.313 us | -51.619% | [-51.686%, -51.428%] | 60/60 |

Fixed-clock measured power was 321.82--333.71 W under the 500 W limit. GPU
clocks were restored explicitly after each run.

## Resource and instruction inspection

- Launch: 132 blocks x 384 threads, one 1x1 cluster per SM, one wave/SM
- Registers: 168 per thread
- Shared memory: 230.40 KiB dynamic + 1.02 KiB driver allocation
- Spills: zero local spilling requests; cubin reports `LOCAL:0`, `STACK:0`
- SASS contains `HGMMA.64x256x16.F32.BF16`, `UTMALDG.3D`, `UTMASTG.3D`,
  `MUFU.EX2`, and `MUFU.LG2`

NCU measured 246.43 us under profiling, 85.43% compute throughput, 55.22%
memory throughput, 33.82% DRAM throughput, 75.04% L2 hit rate, 2.25 active
warps/scheduler, and 0.16 eligible warps/scheduler. Barrier stalls were 46.4%
of average issue latency.

Artifacts: `softplus_profile.ncu-rep`, `softplus_profile.txt`,
`softplus_profile.csv`, and `softplus_sass.txt` in this directory. Retained PTX
and cubins are in `agents/cutlass/build-agent/softplus-dump/`.
