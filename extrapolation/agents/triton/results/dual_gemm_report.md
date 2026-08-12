# Triton fused dual GEMM

Candidate: `agents/triton/submissions/dual_gemm_fused.py`

## Selected implementation

- Tile: `BM=64`, `BN=128`, `BK=128`
- Launch: 128 CTAs x 128 threads (4 warps)
- Pipeline: 4 stages, TMA tensor descriptors
- Scheduler: two N partitions per M tile
- Intermediate: FP32 first-GEMM accumulator rounded to BF16, then retained on
  chip and consumed by the second GEMM; it is never written to global memory
- Scheduler tradeoff: the first GEMM is recomputed for each of two N
  partitions, so the kernel performs 1.5x the canonical arithmetic in exchange
  for filling 128 of the H100's 132 SMs

All correctness checks used seed 42 and the canonical physical layouts.  The
candidate output was bitwise identical to the two-cuBLAS reference in every
saved final run (`exact_fraction=1.0`, max and mean absolute error zero).

## Paired results

Each result used 300 warm-ups, 20 launches per event sample, 60 paired samples,
and 4000 bootstrap resamples.  Negative delta means the Triton candidate is
faster.

| Regime | cuBLAS us | Triton us | Paired delta | Bootstrap 95% CI | Wins |
| --- | ---: | ---: | ---: | ---: | ---: |
| Free clock A | 43.7768 | 42.8176 | -2.4681% | [-3.2869%, -0.9505%] | 56/60 |
| Free clock B | 43.6144 | 42.9056 | -1.7461% | [-2.8575%, -0.6479%] | 47/60 |
| 1050 MHz A | 55.9224 | 51.8008 | -7.8183% | [-8.5513%, -6.2849%] | 60/60 |
| 1050 MHz B | 55.7808 | 51.6000 | -7.5471% | [-8.3124%, -6.5158%] | 60/60 |

Raw results are in `dual_gemm_free_clock.json`,
`dual_gemm_free_clock_repeat.json`, `dual_gemm_1050mhz.json`, and
`dual_gemm_1050mhz_repeat.json`.  GPU 4 was idle before each successful run.
The fixed clock was installed only after a shell restoration trap and was
explicitly reset; the restored maximum SM clock was verified as 1980 MHz.

## NCU and binary inspection

The final Nsight Compute report is `dual_gemm_profile.ncu-rep`.

- Profiled duration: 39.33 us (profiler run, 1.2 GHz SM clock)
- Grid/block: 128 x 128 threads, 0.97 waves/SM
- Registers: 122/thread
- Dynamic shared memory: 196.64 KiB/block
- Local-memory spill requests: 0; cubin local memory and stack: 0
- Achieved/theoretical occupancy: 6.33% / 6.25%
- Compute throughput: 50.78%
- Memory throughput: 46.55%; DRAM throughput: 44.74%
- L1/TEX throughput: 54.01%; L2 throughput: 52.83%
- L2 hit rate: 67.94%; measured memory throughput: 1.09 TB/s
- Main sampled stalls per issued instruction: long scoreboard 3.12, barrier 2.07

`cuobjdump --dump-sass` on the final cubin shows the intended Hopper path:

```text
UTMALDG.2D
HGMMA.64x128x16.F32.BF16
STSM.16.M88.4
HGMMA.64x128x16.F32.BF16
UTMASTG.2D
```

The corresponding Triton GPU IR uses `nvidia_mma` version 3 with
`warpsPerCTA=[4,1]`; the cubin resource report gives 122 registers, zero local
memory, and zero stack bytes.  The HGMMA mnemonic is Hopper's warp-group MMA
instruction in SASS.
