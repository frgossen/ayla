# Priority BF16 GEMM transfer search

## Current results

The 35 BF16 signatures supplied in the priority list were measured again on
`jit-gemm-heuristics`, starting at `45c9ded`. All eight H100 GPUs were idle when
this search began. These are fresh production-JIT measurements, with the same
normal compiler and inference-v2 benchmark used before and after.

| Scope | Before, % of Torch | Installed, % of Torch |
| --- | ---: | ---: |
| Equal-weight geomean, all 35 requested shapes | 102.498 | 104.428 |
| Call-weighted speed, the same 35 shapes | 102.685 | 103.883 |

The geomean gain is 1.883% relative.
Eleven kernels changed; 24 kernels are identical before and after. Holding
unchanged shapes neutral gives a 1.926%
geomean gain in the final run. The independent eight-GPU confirmations give
1.956% for that same
change-only calculation. The median normalized ratio on unchanged kernels is
0.99993; small before/after differences
on those rows are measurement variation.

This is the requested 35-shape scope, which accounts for about 75.7% of the
historical claimed Torch GEMM time. It is not a full-corpus score. The full
MM/BF16 corpus and the legacy native backend were not remeasured, and these
numbers must not be combined with their historical geomeans. The 108% target
has not been reached in this 35-shape scope.

## Confirmed updates

Each row below is the median of eight independent GPU replicas, measured with
fixed candidate schedules and the original incumbent. This table has its own
sampling runs; the 35-shape table below comes from the final installed-route run.

| Priority | Operation M×N×K | Before % | Candidate % | Conservative gain | Decision |
| ---: | --- | ---: | ---: | ---: | --- |
| 3 | addmm 4096×4096×16384 | 100.124 | 101.117 | 0.990% | retained |
| 11 | addmm 4096×8192×2048 | 99.801 | 103.658 | 3.905% | updated |
| 16 | addmm 8192×1024×1024 | 100.024 | 101.803 | 1.697% | updated |
| 20 | addmm 32768×3072×768 | 96.689 | 100.808 | 4.227% | updated |
| 21 | mm 1000×10240×2560 | 99.528 | 112.234 | 12.596% | updated |
| 23 | addmm 16384×4096×1024 | 93.508 | 98.496 | 5.099% | updated |
| 24 | addmm 32768×768×3072 | 100.519 | 107.308 | 6.783% | updated |
| 27 | addmm 25344×3072×768 | 97.710 | 102.831 | 5.191% | updated |
| 28 | addmm 25216×3072×768 | 97.662 | 102.767 | 5.164% | updated |
| 29 | addmm 32768×2304×768 | 99.289 | 102.948 | 3.627% | updated |
| 34 | addmm 25216×2304×768 | 90.866 | 100.626 | 10.681% | updated |
| 35 | addmm 25344×2304×768 | 90.734 | 100.591 | 10.732% | updated |

The deep-K 4096×4096×16384 candidate gained 0.990%, just below the predeclared
1% admission threshold, and was retained. Every installed winner had positive
conservative gains on all eight GPUs, stable Torch references on all eight,
and a different compiled executable from its incumbent.

## What was searched

- Read and froze all 35 priorities before GPU work. A fresh baseline identified
  18 shapes below 100.5% of Torch, including near-ties.
- First pass: 366 distinct configurations, 18–24 per shape. Donors came from
  successful same-N/K shapes at other M values, nearby BF16 MM/addmm shapes,
  FP16 schedules, and legacy native measurements used as hypotheses.
- Second pass: 67 configurations over seven remaining deficits. Newly successful
  25216-M and 32768-M schedules were transferred back to the large-M family;
  fixed-grid donors were tested with their existing rolled-loop option.
- Last pass: ten configurations. Six rolled-grid trials followed the NCU
  large-N observation; four checked legal buffer/slice and resource combinations.
- Total: 443 distinct candidate configurations, 107 interspersed controls,
  423 successful candidate trials, 20 explicit compiler/resource/layout
  rejections, and 370 distinct normalized PTX/launch/optimization signatures.
  This last count describes generated code; final executable distinctness is
  checked separately using SASS, resources, and launch signatures.
- Diversity: M tiles 64/128/192/256; 13 N tile sizes from 80 through 256;
  K tiles 64/128; 2/3/4/5/6/8 operand stages; M/N output grouping;
  fixed unrolled and rolled grids; warpgroup geometry; output slices and
  buffers; cyclic pipelines; clusters; cache hints/promotions; PTX optimization
  levels; normal async rescheduling. Successful measured donors all used
  stmatrix output stores, so this pass did not enumerate other store mechanisms.

The successful transfers were:

- 175360×3072/2304×768 schedules to 25216/25344/32768-M shapes: the same
  existing JIT configuration now covers four additional targets.
- The 25216/25344×3072×768 geometry to their 2304-wide siblings: about 10.7%.
- MM schedules to addmm at 16384×4096×1024 and 32768×768×3072.
- A nearby 4096×6144×1536 MM geometry to 4096×8192×2048 addmm.
- A 4096×2560×2560 BF16 MM donor to 1000×10240×2560 MM: about 12.6%.
- A 4096-M same-N/K donor to 8192×1024×1024 addmm.

Only two new JIT configuration rows were needed. Nine updates reuse existing
rows. This pass did not repeat the earlier 18,352-tuple Cartesian sweep.

## NCU outcomes

Five kernel-replay profiles were collected using normal Torch/Ayla kernels,
cold-cache counter passes, and unchanged clocks. NCU durations are diagnostics
and are never used to select or admit configurations.

| Shape | Observation | Outcome |
| --- | --- | --- |
| addmm 16384×4096×1024 | Incumbent SM throughput 76.69%; transferred M256 geometry 81.72%; Torch 80.66%. Registers/thread fall from 177 to 107; threads/CTA rise from 256 to 512. | The transfer passed independent benchmark confirmation. |
| mm 8192×128112×1024 | Ayla uses 56,960 CTAs and 82.30% SM throughput; Torch uses 132 persistent CTAs and 85.51%. | Six normal rolled-grid trials were slower: 3.44–9.25 ms versus roughly 3.21–3.24 ms for the incumbent. Retained incumbent. |

Remaining deficits include the large-N MM, wide 4096×16384×4096 addmm,
the 175360-M family, and the improved 16384×4096×1024 addmm. Transfers
from smaller-M winners frequently lost performance at 175360 M. No compiler
changes were made to pursue these remaining gaps.

## Measurement and correctness

The ordinary production `build_gemm_module` and `GemmScheduleSeed` path built
all candidates. No compiler passes or shared-memory lifetime rules were added
by the benchmark. `unsafe_reuse_smem` was not enabled. The compiler, runtime,
JIT builder, heuristic selector, and `benchmarks/common.py` match their
pre-search source hashes; only heuristic data changed in the library tree.

Screen ranking uses the smaller of the raw Ayla speedup and the paired
Torch-percentage speedup against nearby incumbent controls. Identical compiled
programs receive no gain. A candidate's own unusually slow Torch reference
therefore cannot establish a win.

Twelve fixed candidates were confirmed across eight independent GPUs each,
with varied row order and repeated incumbent controls: 288 normal Torch/Ayla
pairs. Admission requires a 1% median conservative gain, a 1% raw-latency gain,
at least six positive replicas, at least six stable references, a median
Torch-reference ratio within 2%, and changed SASS/resources/launch signature.
All eleven installed winners exceeded these gates.

Each confirmation used three correctness seeds, BF16 maximum absolute error
32, relative RMS error 0.002, and explicit finite-error checks. The final run
timed the actual installed `gemm_op` routes for all 35 shapes against their
frozen pre-search schedules, and checked that installed compiled programs
match the confirmed candidates. All eleven installed routes also passed nine
CUDA graph checks each, using independent weight/bias sets and subsequently
changed inputs.

Sampling options are applied in the worker process, avoiding the earlier spawn
inheritance bug. Screening used a 0.5-second adaptive budget and 200-round cap;
the final ten concept trials used 0.75 seconds and a 300-round cap;
confirmations used 2 seconds and a 1,000-round cap; final installed-route
comparisons used 1.5 seconds and a 750-round cap. Actual rounds and layer counts
are stored per result; caps are not claimed as actual sample counts.

## Final installed-route results for all 35 priorities

| Priority | Operation M×N×K | Before % of Torch | Installed % of Torch | Candidate trials | Config changed |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | addmm 4096×4096×4096 | 105.891 | 106.187 | 0 | no |
| 2 | addmm 4096×16384×4096 | 99.174 | 99.113 | 37 | no |
| 3 | addmm 4096×4096×16384 | 100.111 | 100.013 | 24 | no |
| 4 | addmm 175360×3072×768 | 100.095 | 98.673 | 33 | no |
| 5 | addmm 175360×768×3072 | 99.825 | 99.887 | 33 | no |
| 6 | mm 1000×14336×4096 | 115.606 | 115.444 | 0 | no |
| 7 | addmm 175360×2304×768 | 99.340 | 99.507 | 33 | no |
| 8 | addmm 16384×3072×768 | 100.795 | 100.769 | 0 | no |
| 9 | addmm 16384×768×3072 | 107.907 | 108.320 | 0 | no |
| 10 | addmm 16384×768×768 | 112.679 | 112.932 | 0 | no |
| 11 | addmm 4096×8192×2048 | 99.884 | 103.482 | 24 | yes |
| 12 | addmm 8192×4096×1024 | 104.181 | 102.903 | 0 | no |
| 13 | addmm 4096×2048×8192 | 102.567 | 102.627 | 0 | no |
| 14 | addmm 8192×1024×4096 | 108.679 | 108.356 | 0 | no |
| 15 | mm 4096×2048×2048 | 101.620 | 101.721 | 0 | no |
| 16 | addmm 8192×1024×1024 | 100.200 | 101.764 | 18 | yes |
| 17 | addmm 16384×1024×1024 | 102.284 | 102.354 | 0 | no |
| 18 | addmm 32768×128×512 | 133.269 | 133.530 | 0 | no |
| 19 | mm 8192×128112×1024 | 96.375 | 96.179 | 42 | no |
| 20 | addmm 32768×3072×768 | 96.821 | 100.407 | 18 | yes |
| 21 | mm 1000×10240×2560 | 99.832 | 112.035 | 18 | yes |
| 22 | addmm 32768×512×128 | 115.232 | 115.997 | 0 | no |
| 23 | addmm 16384×4096×1024 | 93.518 | 98.467 | 28 | yes |
| 24 | addmm 32768×768×3072 | 100.608 | 107.466 | 18 | yes |
| 25 | mm 1000×4096×14336 | 103.557 | 103.571 | 0 | no |
| 26 | addmm 16384×1024×4096 | 106.006 | 105.796 | 0 | no |
| 27 | addmm 25344×3072×768 | 97.798 | 103.010 | 18 | yes |
| 28 | addmm 25216×3072×768 | 97.588 | 102.675 | 18 | yes |
| 29 | addmm 32768×2304×768 | 99.274 | 103.029 | 18 | yes |
| 30 | addmm 25216×768×3072 | 105.601 | 105.887 | 0 | no |
| 31 | addmm 25344×768×3072 | 105.673 | 105.575 | 0 | no |
| 32 | addmm 175360×768×768 | 97.186 | 97.126 | 27 | no |
| 33 | addmm 25088×2048×512 | 106.269 | 106.136 | 0 | no |
| 34 | addmm 25216×2304×768 | 90.775 | 100.615 | 18 | yes |
| 35 | addmm 25344×2304×768 | 90.657 | 100.677 | 18 | yes |

## Integration and evidence

The eleven winners are merged into the existing branches
`oss-gemm-benchmark-heuristics` and `jit-gemm-heuristics`.
Both existing JSONs have eleven additional typed BF16 exact rows. All 755
pre-existing exact rows, including all 749 FP16 rows and the six previous BF16
winners, are preserved. New BF16 groups retain the old tree/fallback for other
shapes. The JIT table updates eleven routes; all original seeds remain as
fallbacks, and all unrelated routes and original configuration rows are preserved.

JIT CPU tests: 21 passed. OSS heuristic CPU tests: 24 passed. GPU correctness,
installed-route equivalence, and changed-input graph checks passed as described
above. These fresh GPU results are for the JIT branch. The OSS branch carries
the same source-confirmed data; its separate static backend was not GPU timed.
That branch's minimal static builder does not implement every captured control
(for example fixed output grids and explicit M warpgroup tiling), so these
source-JIT timings are not a claim that its static seed runner reproduces them.
The reproduction runners require the JIT branch. No builder APIs were added
to the OSS branch in this data update.

`priority_gemm_transfer_search/evidence.json.gz` stores the frozen priorities,
source hashes, manifests, raw results, confirmations, graph checks, and NCU
summaries. Full PTX/cubin/SASS and NCU reports remain in
`.benchmarks/priority_transfer_20261014/` in the JIT worktree. The benchmark-only
runners are included alongside the evidence with reproduction instructions.
