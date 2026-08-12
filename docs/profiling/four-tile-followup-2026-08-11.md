# Four-tile GEMM follow-up (2026-08-11)

Target: `auto_matmul_4096_wgmma256_wg_tma_store_k3_4tiles_split2_dbuf2`
on H100.  The starting policy marked A loads evict-first; B loads and C stores
used the default L2 policy.

## Retained: evict-first output TMA stores

C is write-only, so retaining its cache lines can only displace reusable A/B.
The output promotion now receives the configured `dst` move attributes and the
registered four-tile kernel marks both A loads and C stores evict-first.

Paired, order-reversed CUDA-event measurements (20 launches per batch) found:

| Regime | Paired median | 95% bootstrap interval |
|---|---:|---:|
| 500 W GPU | **-0.833%** | -0.938% to -0.730% |
| 650 W GPU, run 1 | **-0.765%** | -0.871% to -0.623% |
| 650 W GPU, run 2 | **-0.945%** | -1.097% to -0.825% |

Negative is faster.  Every variant was bitwise identical to the original PTX,
whose maximum BF16 error against the FP32-accumulated reference was 1.

The fresh NCU run retained 154 registers/thread, 221.34 KiB shared memory,
zero spills, and 0.97 waves/SM.
NCU elapsed cycles are essentially unchanged from the earlier A-only report
(283,273 versus 282,947), while measured memory throughput fell from 420 to
384 GB/s.  This is consistent with a cache/power-side gain rather than a new
compute schedule; the reports were not a paired timing experiment.

## Scheduling and cache variants rejected

| Experiment | 650 W paired median | Result |
|---|---:|---|
| Hoist next-stage mbarrier poll above WGMMA drain | +0.052% | neutral |
| Sink output-store waits across the following K loop | +0.085% | slower |
| Both transforms | +0.098% | slower |
| Issue RHS TMA before LHS TMA | -0.033% | neutral |
| RHS evict-last | -0.017% | neutral |
| RHS evict-first | +1.527% | substantially slower |

A separate output-policy fraction sweep was monotonic: 25%, 50%, 75%, and
100% evict-first improved by 0.145%, 0.402%, 0.633%, and 0.929%, respectively.
The full-strength policy is retained.  Tensormap prefetch by itself was
neutral.  The transforms and A/B driver live in
`tools/ptx_experiments/{compiler_transfer,ab_compiler_transfer}.py`.

The remaining latency gap does not appear recoverable by simply moving scalar
work into the WGMMA drain: the legal hoist was neutral, and the previously
tested two-group WGMMA pipeline was 5.6% slower.  Likewise, 128 CTAs each have
four indivisible output tiles, so adding four ordinary CTAs does not shorten
the four-tile critical path.  Exploiting the four idle SMs would require a
real Stream-K split of the fourth output tile plus FP32 partial reduction, not
auxiliary prefetch work; the added global synchronization/reduction makes that
a larger follow-up rather than a local PTX rewrite.

Validation: the PTX structure check and all three GPU correctness seeds for
the registered four-tile kernel pass.
