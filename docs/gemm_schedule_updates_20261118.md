# GEMM schedule admission and measured updates

Schedules are rejected when realized output tile-slot efficiency is **below
98%**, including measured exact-shape and graph routes. The selector tries the
next legal table candidate. Direct construction also checks this rule. Tile
edge padding is a separate metric; a partially populated edge tile is not an
empty output iteration. Fixed grids override grouping, and grouped schedules
use the builder's shortened loop bounds rather than the requested group size.

Before a production GEMM specialization is cached or launched, CUDA reports the
loaded kernel's local-memory allocation and resident CTA capacity. Generated
GEMMs do not declare local arrays or use a call stack, so nonzero compiled local
memory is rejected as spill backing. The physical register count alone does
not determine whether a kernel spills. A zero-residency kernel is rejected.

Launch-wave utilization is `CTAs / (ceil(CTAs / capacity) * capacity)`, using
CUDA's capacity on the current device, including cluster residency. Values
**below 70% warn**; they do not reject. Output-loop tails are checked separately
and are not multiplied into launch-wave utilization a second time. SMEM,
register utilization, padding, and input reuse remain diagnostic metrics;
their improvement has not reliably predicted lower complete GEMM latency.

The November 17 tuning campaign and ten additional confirmed recommendations
promote **139 distinct FP16 shapes** relative to the previous committed table.
Earlier seeds are retained except for the previously audited twelve spilling
fallback seeds. The campaign covers wave quantization, SMEM, spills, padding,
output tails, register occupancy, output handling, reduction parallelism, input
reuse, and address-generation variants. See the accompanying
[promotion manifest](gemm_schedule_updates_20261118.json).

The saved full-corpus acquisition before the ten final recommendations reports
111.2854% geometric-mean throughput relative to paired Torch/cuBLAS across 749
FP16 shapes. Measuring both choices in that acquisition gives 111.306% with
the ten replacements. These figures are not a paired comparison against the
previous committed table. Individual promotions were independently confirmed
with production JIT, correctness checks and 100 timing rounds on multiple
H100s under `inference-v5-isolated-gemm`. Some cleaner padding/tail/spill choices
were admitted at equivalent latency; their measured ratios remain in the
manifest.

Validation of the staged commit: 41 focused GEMM JIT, JIT compiler, runtime and
admission tests passed, including actual CUDA spill detection. All ten newly
installed recommendations execute through production JIT, match their
confirmed configurations, pass Torch numerical checks and report zero local
memory. Their compiled resources and errors are saved in the
[verification records](gemm_schedule_updates_20261118_verification.json).
