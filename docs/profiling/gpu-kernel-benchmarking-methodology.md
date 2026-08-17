# GPU kernel benchmarking methodology

This document describes the benchmarking discipline used to hill-climb Ayla's
H100 GEMM kernels.  It focuses on resolving small changes—often below 0.2%—in
the presence of GPU clock ramping, temperature, power-controller history, L2
residency, JIT compilation, and ordinary timing noise.

The most important rule is that there is no single universal benchmark.  A
continuously hot throughput test, a cooled first-burst test, a fixed-clock
test, and a power-limited steady-state test answer different questions.  A
change is only a speedup for the protocol in which it wins.

## The protocols and the questions they answer

| Protocol | Question | Typical use |
|---|---|---|
| Hot paired/interleaved | Is the candidate faster once clocks and the pipeline are in steady state? | Primary hill-climbing screen and confirmation |
| Cooled, rotated inputs | Is the candidate faster when every arm begins from a comparable idle/DVFS state with reduced repeated-input L2 residency? | Confirmation of small compiler and cache-sensitive changes |
| Fixed clock | Does the candidate require fewer cycles at the same frequency? | Separate code efficiency from power/DVFS behavior |
| Isolated sustained load | What operating point does this kernel establish under a power cap? | Power efficiency and application-like sustained behavior |
| Standard repository benchmark | What does the normal user-facing benchmark report? | Final integration check, not the sole evidence for a sub-percent claim |
| Nsight Compute | Why did performance change? | Counters, traffic, instruction mix, stalls, and cycle diagnosis |

Results from different rows must not be pooled into one confidence interval.
If they disagree, report the change as protocol-dependent and investigate
clock, power, cache, and code-placement effects.

## Preflight: make the comparison valid before timing

### Use the same device and record its state

Record at least:

- physical GPU index and model;
- driver and CUDA/PTX toolchain versions;
- power limit;
- application clocks or whether clocks are free;
- initial SM clock and temperature;
- JIT optimization settings;
- kernel resources: registers, shared memory, local memory, spills, block size,
  grid size, and cluster size.

`CUDA_VISIBLE_DEVICES=0` remaps logical device zero.  The `nvidia-smi --id`
argument still needs the physical GPU index.  Mixing these up invalidates the
cooldown monitor.

Check that no unrelated process is using the selected GPU.  Another workload
can alter temperature, clock state, L2 contents, memory traffic, and the power
governor even when its average utilization looks small.

### Compile and load every arm before timing

JIT compilation and module loading must happen outside timed regions.  Force
each launcher to load, query its function attributes, and execute a correctness
launch before warmup or timing.

For PTX work, record the exact Driver JIT options.  O1 and the default JIT can
produce materially different SASS from identical PTX.  Do not compare a source
change while silently changing optimization level, line-info generation,
register limits, or cache mode.

Compilation order can expose cache or code-placement effects.  If a result is
suspiciously small, repeat with the candidate loaded first and with a fresh
CUDA compute-cache directory.  Disassemble the actual cubins when possible;
PTX source equality does not imply SASS equality, and instruction count alone
does not imply speed.

### Preallocate tensors and outputs

Allocate all A, B, and output tensors before timing.  Timed launches must not
include allocation, random-number generation, descriptor construction on the
host, compilation, or correctness checks.

Use a separate preallocated output for each arm so the two implementations do
not create an artificial dependency through the same destination.  Reusing an
arm's own output is fine for a write-only GEMM output.

### Prove correctness first

At minimum:

1. Execute both arms on identical inputs.
2. Require bitwise equality between baseline and candidate when the
   transformation should preserve operation order.
3. Compare against a trusted reference and report maximum error.
4. Test more than one random seed before retaining structural scheduling,
   barrier, tail, or address changes.

A fast but occasionally incorrect kernel is not a benchmark candidate.

## Timing mechanics

### Use CUDA events, not host wall-clock time

Record CUDA events on the same stream around a batch of launches:

```python
start.record()
for _ in range(batch):
    kernel(...)
end.record()
torch.cuda.synchronize()
microseconds = start.elapsed_time(end) * 1e3 / batch
```

This measures GPU execution and excludes most Python and launch-loop overhead.
Wall-clock timing is useful for end-to-end application latency, but it is too
noisy for sub-percent kernel hill-climbing.

### Average multiple launches per sample

One launch is too sensitive to event resolution, clock transitions, and
interrupts.  A timing sample should average a batch of launches.

Useful regimes from this investigation are:

- quick screen: 20–25 launches per arm;
- ordinary paired confirmation: 20 launches, 100–200 pairs;
- low-noise micro-optimization confirmation: 100 launches, 40–50 pairs;
- general benchmark: choose enough launches for roughly 10–50 ms of GPU work
  per sample.

Larger batches reduce timing noise but increasingly measure sustained thermal
and power behavior.  Keep batch size identical between both arms.

### Alternate AB and BA order

Never time all baseline samples followed by all candidate samples.  Clocks and
temperature drift over the run.

Use paired samples and reverse the order every pair:

```text
pair 0: baseline, candidate
pair 1: candidate, baseline
pair 2: baseline, candidate
...
```

Compute the delta within each pair:

```text
delta_percent = 100 * (candidate_time / baseline_time - 1)
```

Negative means the candidate is faster.  Pairing removes much of the slow
common-mode drift that would contaminate a comparison of two independent
medians.

### Warm up both arms symmetrically

For a hot protocol, alternate the two kernels during warmup.  This gives them a
shared clock, temperature, cache, and power-controller history.

The JIT sweep uses 1,000 alternating warmup launches by default.  Longer
cross-library comparisons have used 2,000–5,000 alternating launches.  Warmup
must be long enough for clocks to settle, but should not be silently counted as
timing data.

## Thermal and DVFS control

### Temperature is not enough

The H100 can report a similar temperature while having a different current SM
clock and power-controller history.  During this work the GPU sometimes needed
close to a minute to return from boost to its 345 MHz idle clock.  Sleeping for
an arbitrary five or eight seconds was not sufficient.

For a true cooled protocol:

1. Read the idle SM clock before allocation, JIT, correctness, or warmup work.
2. Before each timed arm, sleep for a minimum interval.
3. Poll `clocks.current.sm` until it returns within a small tolerance of the
   captured idle clock.
4. Use a timeout and report if it expires.

The current harness uses a minimum eight-second pause, an idle tolerance of
30 MHz, and a 60-second timeout.

Capturing the idle reference after correctness launches is a serious bug.  We
observed an incorrect 1,980 MHz "idle" reference that caused the cooldown loop
to accept a still-boosted GPU and reversed the apparent descriptor result.

### Hot and cooled protocols are intentionally different

The hot protocol keeps batches back-to-back so clocks do not fall between
samples.  It measures steady-state throughput and is the primary score for a
continuously used GEMM.

The cooled protocol lets each arm start from a comparable idle state.  It also
includes clock ramp behavior in the measured burst.  It is useful for short
workloads and for detecting interactions that the hot protocol hides.

Do not mix their samples.  If a variant wins cooled but is neutral hot, say so.
That can be a real difference rather than measurement failure.

### Fixed-clock timing separates cycles from power behavior

Free-running clocks are part of real performance, but a kernel can run faster
because it draws less power and receives a higher clock rather than because it
needs fewer cycles.

Repeat important comparisons at a locked SM clock when possible.  If the
fixed-clock result is neutral but the free-clock result wins, the likely cause
is DVFS or power efficiency.  If the fixed-clock result wins, the candidate has
a per-cycle advantage.

Always restore clock and power settings after the run.

### Low-power steady state needs isolation

At low power limits, controller state persists long enough that rapid AB/BA
interleaving may measure a shared governor history rather than either kernel's
natural operating point.

For power-efficiency studies, use separate processes, comparable starting
temperatures, a long ramp, and an uninterrupted measurement window.  Sample
power, clock, temperature, and power-cap state throughout the run.  These
isolated results answer a different question from equal-clock or short paired
timing.

## Cache and input control

### Fixed inputs and rotated inputs measure different reuse regimes

Repeatedly launching on one A/B pair allows later launches to benefit from the
previous launch's L2 residency.  This may be representative of repeated use of
the same weights, but it can hide sensitivity to input traffic and scheduling.

The cooled comparison allocates eight A/B pairs and rotates them across
launches.  The input offset continues across arms so neither implementation is
systematically assigned the earliest or latest copies.

Input policy is independent of thermal policy: a hot benchmark may use either
fixed or rotated inputs.  The JIT sweep uses one fixed pair, while the K3 PTX
variant harness rotates eight pairs even when cooldown is disabled.  State
which policy was used and confirm important cache-sensitive changes under the
other one.  Do not call a fixed-input cache-hot result a general cold-input
result.

Rotating inputs does not fully clear L2, and explicit L2 clearing introduces
its own workload.  If L2-cleared timing is needed, report it as a separate
protocol.

### Keep data distribution identical

Power and clock behavior depend on switching activity.  Use the same dtype,
shape, layout, seed, and value distribution for both arms.  Random-normal BF16
inputs were used for the current 4096-cubed GEMM work.

## Statistics and acceptance criteria

### Report paired statistics

For each comparison, report:

- median baseline and candidate latency;
- median paired percentage delta;
- bootstrap 95% confidence interval of paired deltas;
- number of pairs and launches per arm;
- optionally, the number of pairs won by the candidate.

The bootstrap resamples paired deltas, not the two arms independently.  A
fixed random seed makes the reported interval reproducible.

Use medians for the main result.  Means, standard deviations, minimums, and
maximums are still useful diagnostics, but selecting the minimum exaggerates
noise and boost transients.

### Hill-climbing decision rule

For a performance claim:

1. The candidate must be correct.
2. Kernel resources and SASS must be recorded.
3. A quick paired screen must show a plausible improvement.
4. A longer paired run must have a confidence interval below zero in the
   protocol being optimized.
5. Repeat the entire run when the expected gain is below about 0.1%.
6. Confirm that reversing compilation/load order does not erase the result.
7. Run at least one second protocol—cooled, fixed-clock, or rotated-input—to
   identify whether the win is universal or regime-specific.

For changes that materially reduce PTX/SASS complexity with unchanged
resources, it can be reasonable to retain a performance-neutral cleanup.  In
that case, call it neutral and report the code-size reduction; do not describe
it as a speedup.

If the confidence interval crosses zero, the honest conclusion is neutral or
inconclusive.  More samples may narrow the interval, but repeatedly rerunning
until one interval happens to exclude zero is p-hacking.

## Nsight Compute is diagnostic, not the timing authority

NCU replays and perturbs kernels.  Its reported duration, clock, and sometimes
even apparent pipeline utilization are not interchangeable with uninterrupted
CUDA-event timing.

Use matched NCU reports to compare:

- elapsed and active cycles;
- tensor-pipeline activity;
- instruction counts and issue rate;
- DRAM/L2 traffic and hit rates;
- shared-memory conflicts;
- occupancy and eligible warps;
- stall categories;
- registers, shared memory, barriers, and spills.

Treat an NCU-only speedup as a hypothesis.  The ping-pong experiment showed
higher tensor activity under profiling, but locked-clock CUDA-event timing did
not reproduce a cycle reduction, so it was correctly rejected.

## Recommended workflow

### 1. Static and correctness screen

- Generate PTX and compile/disassemble SASS.
- Record instruction/opcode counts and resources.
- Reject spills, correctness failures, and obvious SASS explosions.

### 2. Fast hot paired screen

Use 20–40 pairs of 20–25 launches with alternating AB/BA order.  This quickly
filters large regressions and identifies candidates worth more GPU time.

### 3. Low-noise hot confirmation

For a sub-0.2% candidate, use approximately 40–50 pairs of 100 launches.  The
larger batch was necessary to resolve the direct warpgroup-ID change at about
0.09%.

### 4. Cooled rotated-input confirmation

Use three to five rounds, 25 launches per arm, eight input copies, reversed arm
order, and clock-based cooldown before every arm.  This protocol is slow
because a true return to idle can take nearly a minute.

Three rounds are useful directional evidence, but produce a coarse bootstrap
interval.  Prefer five or more rounds for a retained sub-percent claim when
machine time permits.

### 5. Fixed-clock or power diagnosis

Use fixed-clock timing if free-clock and cooled results disagree.  Use isolated
sustained loads only when the application question includes long-term power
and DVFS equilibrium.

### 6. Integration benchmark

Run the normal repository benchmark and compare against the relevant external
baseline such as `torch.mm`.  This is a final sanity check, not a replacement
for paired variant timing.

## Lessons from this investigation

- Absolute latency drifted by several microseconds as the board heated, while
  paired deltas were often much more stable.  Optimize the paired delta, not
  the prettiest absolute number.
- Fewer PTX or SASS instructions did not reliably mean faster execution.
  Scheduling, dependency chains, uniform-register placement, and code layout
  mattered more than raw count in several experiments.
- Individually positive changes were not necessarily additive.  Always time
  the final combined kernel against the original retained baseline.
- Compiler hints changed value after another transformation.  `.reqntid` was
  unattractive in isolation but useful after A-descriptor precomputation.
- A result can be real but protocol-specific.  Descriptor precomputation won
  the cooled test while the complete combined form was neutral in one long hot
  end-to-end comparison.
- Profiler counters explain behavior; ordinary CUDA-event timing decides
  whether users actually receive a latency improvement.

## Repository harnesses

- [`bench_gemm_collection.py`](../../benchmarks/bench_gemm_collection.py) is
  the standard steady-state user-facing benchmark.
- [`compare_k3_ptx_variants.py`](../../benchmarks/compare_k3_ptx_variants.py)
  performs paired K3 variant comparisons with rotated inputs and optional
  clock-based cooldown.
- [`sweep_ptx_jit_options.py`](../../benchmarks/sweep_ptx_jit_options.py)
  compares Driver JIT options in alternating hot bursts.
- [`power-sweep-h100-gpu3-2026-08-10`](power-sweep-h100-gpu3-2026-08-10/README.md)
  documents the isolated power/DVFS methodology and why it differs from the
  normal paired ranking.

## Current commands

Standard user-facing benchmark:

```bash
CUDA_VISIBLE_DEVICES=6 pixi run python benchmarks/bench_gemm_collection.py \
  -k 'torch.mm.bf16|full output store'
```

Hot paired PTX comparison: disable clock waiting and use larger batches:

```bash
CUDA_VISIBLE_DEVICES=6 pixi run python benchmarks/compare_k3_ptx_variants.py \
  --physical-gpu 6 \
  --baseline recurrence \
  --candidate precompute-lhs-req256-direct-wgid \
  --rounds 50 --launches 100 --input-copies 8 \
  --cooldown-s 0 --cooldown-timeout-s 0
```

Cooled, rotated-input comparison:

```bash
CUDA_VISIBLE_DEVICES=6 pixi run python benchmarks/compare_k3_ptx_variants.py \
  --physical-gpu 6 \
  --baseline recurrence \
  --candidate precompute-lhs \
  --rounds 5 --launches 25 --input-copies 8 \
  --cooldown-s 8 --cooldown-timeout-s 60
```

Driver JIT option sweep:

```bash
CUDA_VISIBLE_DEVICES=6 pixi run python benchmarks/sweep_ptx_jit_options.py \
  --configs O1 O2 O3 O4 --warmup 1000 --pairs 100 --batch 20
```

## Reporting checklist

Every retained optimization should leave behind enough information to
reproduce the decision:

- date, host, physical GPU, driver, CUDA toolkit, power limit, and clock mode;
- exact baseline and candidate source/configuration;
- PTX JIT options and entry directives;
- shape, dtype, layout, seed, and input rotation policy;
- warmup, rounds/pairs, launches per sample, and arm ordering;
- cooldown rule or fixed clock;
- latency, TFLOP/s, paired delta, confidence interval, and win count;
- correctness result;
- registers, shared memory, local memory, spills, barriers, and block/grid;
- SASS instruction and relevant opcode counts;
- whether the result is hot, cooled, fixed-clock, or isolated sustained load;
- any disagreement between protocols.

This discipline does not eliminate all variance.  It makes the remaining
uncertainty visible, prevents thermal and ordering artifacts from becoming
"optimizations," and gives us enough confidence to hill-climb changes whose
real effect is only a few hundredths of a percent.
