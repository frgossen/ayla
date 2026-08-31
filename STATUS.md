# Ayla B200 port status

Snapshot: 2026-08-31, source commit `716b098` (`Record band-concurrency audit`).
`B200_PORT.md` remains the canonical experiment log; this file is the compact
performance summary requested for the current state.

## Executive summary

Ayla now has a correct, explicit Blackwell path (`PTX 9.0`, `sm_100a`, TMA,
`tcgen05`, tensor-memory allocation/deallocation, cluster launch metadata, and
CUDA JIT diagnostics) while retaining the Hopper path. The full current suite
passes on B200 (`244 passed, 38 target-aware skipped`), and the production
square curve passes three correctness seeds at every configured size.

Performance is not yet at the acceptance gates. In the latest complete curve,
Ayla beats `torch.mm`/cuBLAS by 5.13% at 4096, trails by 7.28% at 8192 and 4.30%
at 16384, and reaches 86.7% of cuBLAS by geometric mean over 512-16384. The
4096 result is still marginal: three independent gate runs were +3.4%, +5.6%,
and +4.8%, so it has not met the required +5% in all three runs.

Environment: NVIDIA B200 (CC 10.0), driver 580.82.07, PyTorch 2.13.0+cu130 /
CUDA runtime 13.0. CUDA 13.0.3 is installed at `/usr/local/cuda-13.0`, while
the default `/usr/local/cuda` symlink still selects CUDA 12.8.2. Ayla emits
PTX 9.0 for `sm_100a`; CUDA 13's `ptxas` accepts 9.0 and rejects 9.1 as newer
than its supported ISA, so the retained path is using the latest locally
available PTX version. Reported peak percentages use NVIDIA's published 2.25
PFLOP/s dense BF16 B200 peak.

## Current best square-curve performance

All timings below are from the same graph-captured NT BF16 curve run on idle
physical GPU 6, graph depth 32, with identical operands and three correctness
seeds. The cuBLAS configuration is the `nvjet` kernel selected by `torch.mm`;
it was not overridden by an Ayla-specific benchmark choice.

| Shape M=N=K | Best Ayla configuration | Ayla ms / TFLOP/s (% peak) | cuBLAS selected kernel | cuBLAS ms / TFLOP/s (% peak) | Ayla vs cuBLAS |
| ---: | --- | ---: | --- | ---: | ---: |
| 512 | M64 N32 K512, early TMA, O2 | 0.002826 / 94.99 (4.2%) | `nvjet_sm100_tst_64x32_64x16_2x2_2cta_h_bz_TNT` | 0.002246 / 119.52 (5.3%) | -20.52% |
| 1024 | M64 N64 K256, two-bank readiness overlap, paired epilogue, O2 | 0.005650 / 380.09 (16.9%) | `nvjet_sm100_tst_128x64_64x10_2x2_2cta_h_bz_TNT` | 0.003915 / 548.53 (24.4%) | -30.71% |
| 2048 | M128 N128 K128, prefetch, communication-free adjacent-M Y2 cluster, O3 | 0.019466 / 882.54 (39.2%) | `nvjet_sm100_tst_128x256_64x6_2x2_2cta_h_bz_TNT` | 0.016140 / 1064.43 (47.3%) | -17.09% |
| 4096 | M128 N232/N224 K128, two-N persistence, reverse-K LHS reuse, 148-CTA phase split, O1 | 0.128221 / 1071.89 (47.6%) | `nvjet_sm100_tst_128x256_64x6_2x2_2cta_h_bz_TNT` | 0.134797 / 1019.60 (45.3%) | **+5.13%** |
| 8192 | M128 N232/N224 K128, 2x2 persistence, B-major adjacent-M traversal and RHS reuse, O1 | 1.170575 / 939.29 (41.7%) | `nvjet_sm100_tst_128x256_64x6_2x1_2cta_v_bz_TNT` | 1.085408 / 1012.99 (45.0%) | -7.28% |
| 16384 | M128 N256 K128, six serial B-major M bands `(23,23,23,23,18,18)`, 28 waves, O1 | 8.828296 / 996.35 (44.3%) | `nvjet_sm100_tst_256x256_64x4_2x1_2cta_v_bz_TNT` | 8.449115 / 1041.07 (46.3%) | -4.30% |

Raw timing evidence: [complete curve](.benchmarks/2026-08-31-bf16-square-curve-2048-cluster-y2-gpu6-forward.txt),
[CSV](.benchmarks/2026-08-31-bf16-square-curve-2048-cluster-y2-gpu6-forward.csv),
and independent 4096 gate [run 1](.benchmarks/2026-08-31-bf16-square-4096-current-gate-run1-gpu6-forward.txt),
[run 2](.benchmarks/2026-08-31-bf16-square-4096-current-gate-run2-gpu6-reverse.txt),
and [run 3](.benchmarks/2026-08-31-bf16-square-4096-current-gate-run3-gpu6-forward.txt).

## Profiler comparison

Values are NCU replay measurements from the closest retained kernel reports,
not timed benchmark samples. `Tensor active` is
`sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed`; `DRAM` is
percent of sustained peak bandwidth. The 16384 Ayla row is one representative
M23 band of the six-launch production operation. Captures were taken on B200s
under low contention, but not all pairs were captured on the same day/GPU, so
the table is diagnostic rather than a second timing comparison.

| Shape | Tensor active, Ayla / cuBLAS | L2 hit, Ayla / cuBLAS | DRAM, Ayla / cuBLAS | No eligible, Ayla / cuBLAS | Registers/thread, Ayla / cuBLAS | Shared KiB/CTA, Ayla / cuBLAS |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 4.84% / 2.25% | 67.72% / 59.08% | 1.89% / 1.68% | 93.77% / 92.28% | 42 / 255 | 98.33 / 180.74 |
| 1024 | 21.82% / 12.92% | 80.64% / 62.58% | 4.69% / 4.94% | 91.68% / 93.01% | 31 / 255 | 66.61 / 221.56 |
| 2048 | 39.83% / 48.32% | 82.88% / 63.26% | 7.28% / 8.12% | 91.63% / 95.52% | 55 / 255 | 82.94 / 213.28 |
| 4096 | 79.32% / 77.64% | 89.91% / 74.08% | 8.55% / 7.79% | 92.32% / 96.23% | 54 / 255 | 109.57 / 213.28 |
| 8192 | 86.68% / 94.93% | 85.18% / 72.38% | 12.79% / 15.21% | 92.41% / 96.97% | 78 / 255 | 109.57 / 213.28 |
| 16384* | 92.41% / 96.52% | 84.98% / 65.88% | 12.34% / 18.33% | 92.51% / 98.33% | 50 / 255 | 115.71 / 213.22 |

The profiler evidence shows that Ayla is not DRAM-bandwidth-bound: even at
16384 it uses about 12% of peak DRAM bandwidth. The dominant recurring signal
is 91-94% scheduler cycles with no eligible warp, with source samples clustered
around TMA readiness and `tcgen05` completion polling. At 8192 and 16384,
cuBLAS sustains materially higher tensor-pipe activity (about 95-97% versus
87-92%), which accounts for most of the remaining large-shape gap despite
Ayla's higher L2 hit rate and much lower register/shared-memory allocation.

The cuBLAS `nvjet` kernels use two- or four-CTA clusters, about 181-222 KiB of
dynamic shared memory, 255 registers per thread, and a persistent
producer/consumer split (`tid.x < 128`). That warp-specialized structure is
explicitly prohibited for this Ayla experiment, so it is useful as a hardware
efficiency reference but cannot be copied directly.

### Holistic local-minimum audit

The latest source-correlated comparison does not point to occupancy, spills,
or DRAM bandwidth as a hidden cap. At 8192, Ayla has 7.96 active warps/SM
versus cuBLAS's 6.00, an 85.18% versus 72.38% L2 hit rate, and zero spilling,
but executes 44.93M scheduler-level instructions versus 16.33M while reaching
86.68% versus 94.93% tensor activity. About 83% of Ayla's sampled non-issued
cycles are long-scoreboard stalls; the corresponding cuBLAS share is about
93%, but cuBLAS's two-CTA MMA and warp-specialized producer/consumer pipeline
perform more tensor work per control instruction.

This excess is not primarily caused by the retained four-output shape policy.
A fresh exact-N256 single-output control executes 43.26M instructions and
reaches 88.00% compute throughput—essentially the same profile as the retained
44.93M/88.17% kernel. Focused graph timing sometimes favored exact N256, but
paired medians were only +0.98% to +1.27% with 4.6-5.1% dispersion; it is not a
stable replacement. The older A-major/LHS-reuse 2x2 schedule was consistently
about 3.5% slower, while removing split readiness and prefetching cost 14-22%.
Thus the retained topology is not masking a faster simple baseline; the main
remaining cap is the uniform pipeline's per-K128 readiness/completion protocol.

The dynamic SASS mix makes that distinction concrete. In the retained 8192
capture, Ayla executes about 1.18M group-1 `UTCHMMA` instructions, 3.82M
`SYNCS...TRYWAIT` instructions, and 6.83M branches; cuBLAS executes about
0.52M `UTCHMMA.2CTA`, 3.12M `TRYWAIT`, and 3.50M branches. The lower cuBLAS
counts come from doing roughly twice the tensor work per collective MMA and
amortizing readiness over a six-stage producer/consumer pipeline—not from a
newer PTX encoding that Ayla has failed to request.

A direct fast-path experiment that tested each barrier before entering the
blocking wait loop was also neutral (best paired result +0.10% forward and
+0.05% reverse). The next useful work must change the amount of useful tensor
work hidden behind those dependencies, not merely reshape their polling code.

A follow-up implemented the other legal dense 2-SM shape, M128-total/N256,
using current CUTLASS's 2x2 TMEM mapping. It is exact, but even its K256 form
reached only 747 TFLOP/s versus 1,222 for retained Ayla and 1,340 for cuBLAS in
the same run. Also, cuBLAS's `128x256 ... 2cta` name denotes its per-CTA slice:
its M256-total six-stage collective is the topology Ayla had already tested at
roughly 654 TFLOP/s without warp specialization. Thus neither an older shape
choice nor a missing legal `tcgen05` geometry explains the remaining gap.

The conspicuous cuBLAS epilogue sequence—wide tensor-memory loads, `STSM`, then
TMA output—was also reproduced with PTX 9.0 on an exact M128/N256 control while
reusing dead operand buffers, so shared-memory occupancy did not change. It was
bit-exact for three seeds and reduced executed instructions from 43.26M to
39.84M, but lost 3.8-4.9% in both timing orders; NCU showed compute throughput
falling from 88.00% to 85.69% and no-eligible cycles rising from 92.77% to
93.70%. Double-buffering the TMA output did not recover the loss. The opcode
difference is therefore coupled to cuBLAS's prohibited producer/consumer and
two-CTA topology, rather than a standalone epilogue optimization Ayla omitted.

Two further controls rule out hidden issuer and collector wins. Rotating the
legal TMA issuer among warps was exact but 1.7% slower; moving tensor issue away
from the layout-owning warp trapped as an illegal instruction. A two-output
M128/N128 path using PTX 9.0 B-operand `fill`/`lastuse` collectors was also
bit-exact and improved its non-collector control by 0.7-1.2%, but still ran
18-20% behind exact N256. Thus fixed issue ownership and the absence of a B
collector are not the optimizations capping the retained wide kernel.

Likewise, narrowing to N192 to fit a fourth LHS buffer made it possible to
prefetch both A banks for the following K128 group while preserving two-CTA
residency. That exact control cut static TMA and wait sites by roughly 43%, but
was 8.3% slower than the matching three-LHS-buffer N192 kernel: extra TMA work
during the tensor interval delayed the critical RHS refill. Deeper lookahead
within the same uniform pipeline is therefore not automatically beneficial;
the overlap engine is already contended where the retained schedule places it.

The six-launch 16384 M-band policy was also checked as a possible scheduling
local minimum. Running the bands in concurrent pairs produced a tempting 3.7%
gain at graph depth 8, but at the production depth it lost in both preparation
orders (999/1009 versus 1011/1024 TFLOP/s for serial). Serial cache-sized
banding is therefore retained; the short-run concurrency result was not a
stable source of parallelism or cross-band reuse. Splitting every band in half
while keeping the same combined A footprint also lost 4.0%, ruling out hidden
grid concurrency independently of the original band size. A narrower
programmatic-dependent-launch prototype, which allowed only the prior band's
tail/epilogue to overlap the next band, changed from +0.86% forward to -0.17%
reverse and was likewise rejected as order noise.

Finally, replacing either current tensor-completion acquire barrier with PTX
9.0's relaxed wait plus `tcgen05.fence::after_thread_sync` passed three isolated
correctness launches but faulted during the first sustained graph replay. The
acquire waits therefore encode real cross-thread buffer/tensor-memory lifetime
ordering, not merely conservative syntax that can be removed for speed.
Polling either completion bank from one lane and publishing through a CTA
barrier was safe but 1.7-4.0% slower. Separating the retained joined A0/B0 TMA
completion into two shorter waits was also neutral in both orders, so neither
poll participation nor the earlier barrier-coalescing choice is the hidden cap.

## What has worked

- Explicit Blackwell target configuration and native `tcgen05`/tensor-memory
  lowering made the port correct and testable without weakening Hopper support.
- TMA staging, split readiness barriers, A0/B0 lookahead, paired/coalesced
  tensor-memory writeback, and overwrite-first accumulation produced the main
  single-kernel gain.
- Shape-specific scheduling matters: two-N persistence and phase splitting win
  at 4096; four-output adjacent-M/RHS reuse is best at 8192; cache-sized B-major
  M bands and exact wave partitioning are best at 16384.
- At 1024, moving the remaining readiness wait behind eight already-ready MMAs
  improved throughput by about 5%. At 2048, communication-free adjacent-M
  cluster placement improved paired benchmark throughput by about 5.3% without
  adding multicast or barriers.
- Output-store cache bypass (`L1::no_allocate`, `L2::evict_first`) is important
  at 4096; removing L1 bypass costs about 5%.

## What has not worked

Extra buffering/stages, CTA-group-2 MMA, operand multicast, explicit L2 access
windows, `cp.async`/TMA hybrids, multi-warp TMA enqueue, finer wait splitting,
larger persistent dependency chains, and most cache/synchronization micro-tunes
either regressed or stayed below run noise. Narrower K stages increase
TMA/barrier/completion frequency; wider exact-N tiles reduce CTA count but have
longer readiness/tensor paths. The common failure mode is added synchronization
or dependency depth, not excess DRAM traffic.

## Remaining gates and next direction

- Stabilize the 4096 advantage above +5% in three independent runs.
- Reach at least +5% at 8192 and 16384, and a positive geometric-mean advantage
  over the complete 512-16384 curve.
- Re-run the full collection and final clean verification once the performance
  policy is final.
- Continue with dependency-changing experiments aimed at TMA readiness and
  tensor completion latency; cache hints, direct versus wide/TMA output,
  simple barrier removal, and a nonblocking-before-blocking wait fast path have
  been exhausted for the current kernels.

Profiler evidence: [512 Ayla](.benchmarks/2026-08-30-tcgen05-512-early-tma-gpu6.ncu-rep),
[1024 Ayla](.benchmarks/2026-08-31-tcgen05-1024-bank2-overlap-gpu6.ncu-rep),
[2048 Ayla](.benchmarks/2026-08-31-tcgen05-2048-cluster-y2-gpu6.ncu-rep),
[4096 Ayla](.benchmarks/2026-08-31-tcgen05-4096-current-gpu6.ncu-rep),
[8192 Ayla](.benchmarks/2026-08-31-tcgen05-8192-current-gpu6.ncu-rep), and
[16384 representative Ayla band](.benchmarks/2026-08-31-tcgen05-16384-m-band23-gpu6.ncu-rep).
