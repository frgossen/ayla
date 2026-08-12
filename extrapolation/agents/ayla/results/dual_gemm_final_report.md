# Ayla Dual GEMM final report

Candidate: `submissions/dual_gemm.py`

## Design

- One fused Hopper kernel; the BF16 `A @ B1.T` result is kept in a 128-byte-swizzled, block-major shared-memory tile and never materialized in HBM.
- Grid `(64, 2, 1)`, block `(128, 1, 1)`. Each CTA computes 64 rows and 2048 output columns as eight 64x256 output tiles.
- First GEMM: K64, four-stage TMA pipeline. Second GEMM: K128 B2 TMA load and eight WGMMA operations per output tile. Final output uses two padded TMA-store buffers.

## Correctness and runtime

All runs used physical GPU 4 (NVIDIA H100), seed 42, exact benchmark semantics, and no `--allow-busy`.

- Correctness: exact fraction `1.0`, maximum and mean absolute error `0.0`.
- Normal clocks, required 300 warm-ups / 20 launches per sample / 60 paired samples:
  - candidate `48.6624 us`, reference `44.7416 us`, paired median delta `+8.58%` (95% CI `+7.52%` to `+10.78%`).
  - fresh process: candidate `49.1248 us`, reference `44.7008 us`, paired median delta `+9.50%` (95% CI `+8.60%` to `+11.12%`).
- Fixed 1050 MHz graphics clock, same full harness: candidate `72.8256 us`, reference `56.4856 us`, paired median delta `+28.04%` (95% CI `+27.55%` to `+29.11%`). The lock was protected by a shell trap and explicitly reset; the 500 W power limit was unchanged.

## Resource and SASS audit

PTXAS for `sm_90a` reports 154 registers/thread, 218,120 bytes static shared memory, three barriers, zero stack, and zero spill loads/stores. Launch geometry is 128 CTAs x 128 threads. The final cubin contains 40 static `HGMMA` sites, 17 `UTMALDG` sites, and two `UTMASTG` sites.

Nsight Compute (profiling/replay duration `62.66 us`) reports:

- 0.97 waves/SM across 132 SMs; 6.25% theoretical and 6.18% achieved occupancy, limited to one CTA/SM by shared memory.
- 31.61% SM throughput, 29.38% DRAM throughput, 718.79 GB/s memory throughput, and 68.79% L2 hit rate.
- One active warp and 0.07 eligible warps per scheduler; 93.43% of scheduler cycles have no eligible warp.
- Barrier stalls are about 35.2% and long-scoreboard stalls about 34.1% of the 15.2 cycles per issued instruction.

Artifacts: `dual_gemm_final_free.json`, `dual_gemm_final_free_repeat.json`, `dual_gemm_final_fixed1050.json`, `dual_gemm_final.ptx`, `dual_gemm_final.cubin`, `dual_gemm_final.sass`, `dual_gemm_final_ptxas.log`, `dual_gemm_final_profile.ncu-rep`, `dual_gemm_final_profile.txt`, and `dual_gemm_final_pytest.log`.
