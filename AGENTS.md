# Ayla B200 port: operating instructions

## Mission

Port Ayla from Hopper/H100 to Blackwell/B200 and make its BF16 GEMM kernels
correct and decisively faster than the `torch.mm`/cuBLAS baseline. Ayla is a
custom IR that lowers directly to PTX; architecture support belongs in the IR,
passes, PTX lowering, runtime launch metadata, and tests rather than in a
one-off benchmark workaround.

This is a long-running optimization task. Do not conclude the task, declare
success, or give a final answer merely because the code compiles, one kernel
runs, one benchmark improves, or progress becomes difficult. Keep diagnosing,
implementing, testing, profiling, and tuning until every completion gate below
is satisfied. When an approach fails, record what was learned and continue with
the next evidence-based experiment. If temporarily blocked by a busy GPU, wait
and retry instead of ending the task.

## Non-negotiable design constraint: no warp specialization

Do not implement producer/consumer warp specialization. In particular, do not
reserve persistent warps or warp groups solely for TMA loads, tensor-core work,
stores, barrier management, or other pipeline roles. The experiment is to use
software pipelines whose stages, ordering, buffering, and shapes remain
searchable and reconfigurable, with the participating warps following the same
high-level pipeline.

Blackwell async mechanisms, `tcgen05`, TMA, mbarriers, CTA clusters, multicast,
and instructions that require an elected issuing thread are allowed. A brief
single-thread or single-lane issue sequence required by an instruction is not
itself warp specialization, provided the other warps are not assigned durable
producer/consumer roles. Document any such choice in code and preserve the
shared software-pipeline structure.

## Non-negotiable compatibility constraint: retain H100 support

The B200 port must not replace or regress the existing H100 path. Branch on
target architecture or CUDA compute capability wherever hardware behavior,
PTX syntax, tensor-core instructions, accumulator storage, synchronization, or
launch requirements differ. Keep those branches at reasonable abstraction
boundaries—prefer target capabilities selected by IR/module configuration and
consumed by lowering/runtime code over scattered checks for device names or
literal `sm_90a`/`sm_100a` strings.

Share architecture-independent IR and transforms. Use explicit Hopper and
Blackwell implementations only where the hardware actually diverges, such as
`wgmma` versus `tcgen05`, tensor memory, cluster launch attributes, or supported
PTX forms. Add tests for both target paths, retain H100 expected PTX coverage,
and never make runtime GPU detection silently change serialized IR semantics.

## Work continuously and autonomously

- Begin by making a concrete plan, then keep exactly one next experiment in
  progress. Update the plan as evidence changes.
- Do not stop at a status report. After reporting an intermediate result,
  proceed to the next useful action.
- Do not ask the user to choose ordinary implementation details. Inspect the
  code, PTX/SASS, profiler data, and NVIDIA tooling, make a reasoned choice, and
  test it.
- Exhaust local diagnoses and alternative implementations before calling
  anything blocked. A failed experiment, compiler error, performance
  regression, unavailable GPU, or incomplete feature is not a terminal state.
- Preserve useful work in small, reviewable changes. `B200_PORT.md` is the one
  canonical progress document; keep it sparse and current rather than creating
  competing diaries or status files. Each entry should be a 2-5 sentence note
  or one compact table row covering the hypothesis/change, correctness and
  benchmark result, key profiler finding, and next decision. Link raw artifacts
  under `.benchmarks/` instead of copying large output into the document.
- Document architecture decisions and non-obvious PTX constraints in the code
  they affect, and summarize consequential decisions in `B200_PORT.md`.
- Prefer one single-purpose git commit per coherent change once its focused
  tests pass. Use meaningful imperative commit messages that explain the change,
  such as `Add Blackwell PTX target configuration` or `Lower BF16 GEMM through
  tcgen05`. Stage only files belonging to that change, never rewrite unrelated
  user commits, and do not push unless explicitly asked. Record the commit hash
  in the corresponding `B200_PORT.md` entry when available.
- Re-read this file and `B200_PORT.md` after context compaction or a restart,
  then resume from the best verified result instead of restarting the port.
- Read `BLOCKERS.md` at startup and before declaring completion. It is separate
  from the progress log and contains only external blockers that require human,
  permission, credential, network, hardware, or administrative action.
- Never weaken tests, tolerances, sample counts, synchronization, or baselines
  to manufacture a win. Never omit a failing relevant shape from the final
  result.

### Blocker protocol

Record an issue in `BLOCKERS.md` immediately when progress on some requirement
needs an action the agent cannot perform. Include the date, status, affected
goal, exact error or evidence, everything already attempted, the precise human
action requested, and a command/check for resuming afterward. Never include
credentials or secrets.

A compiler error, failed optimization, performance regression, busy GPU, or
hard problem is not an external blocker. Continue investigating those. A real
blocker also does not justify ending the whole task when independent work
remains: log it, pursue the remaining work, and periodically retry safe checks.
When resolved, mark the entry resolved with the resolution and date rather than
deleting it.

## Environment and command discipline

- Work from the repository root and use the checked-in Pixi environment for
  every Python command: `pixi run python ...`, `pixi run pytest ...`, or a Pixi
  task. Do not silently use the system Python or install packages globally.
- Inspect `git status` before edits. Preserve unrelated user changes and do not
  use destructive git commands.
- Generated PTX, NCU reports, plots, CSVs, and experiment notes belong under
  `.benchmarks/` or another ignored temporary path, not in source directories.
- Prefer focused tests during iteration, but regularly run the full suite.
- Capture and expose CUDA driver JIT error/info logs when PTX loading fails;
  generic `CUDA_ERROR_INVALID_PTX` messages are insufficient for this port.

### Useful paths

Use these known paths instead of rediscovering tools or accidentally selecting
system Python:

| Purpose | Path |
| --- | --- |
| Repository root | `/home/arseniivanov/ayla` |
| Pixi executable | `/home/arseniivanov/.pixi/bin/pixi` |
| Pixi environment | `/home/arseniivanov/ayla/.pixi/envs/default` |
| Pixi Python | `/home/arseniivanov/ayla/.pixi/envs/default/bin/python` |
| Pixi pip | `/home/arseniivanov/ayla/.pixi/envs/default/bin/pip` |
| Pixi pytest | `/home/arseniivanov/ayla/.pixi/envs/default/bin/pytest` |
| Pixi ninja | `/home/arseniivanov/ayla/.pixi/envs/default/bin/ninja` |
| CUDA root | `/usr/local/cuda` |
| Nsight Compute CLI | `/usr/local/cuda/bin/ncu` |
| Nsight Compute installation | `/opt/nvidia/nsight-compute/2025.3.1` |
| CUDA compiler | `/usr/local/cuda/bin/nvcc` |
| PTX assembler | `/usr/local/cuda/bin/ptxas` |
| Cubin/Fatbin inspector | `/usr/local/cuda/bin/cuobjdump` |
| SASS disassembler | `/usr/local/cuda/bin/nvdisasm` |
| GPU status tool | `/usr/bin/nvidia-smi` |
| Main 4096 benchmark | `/home/arseniivanov/ayla/benchmarks/bench_gemm_collection.py` |
| Square benchmark | `/home/arseniivanov/ayla/benchmarks/benchmark_bf16_square_curve.py` |
| Benchmark harness | `/home/arseniivanov/ayla/benchmarks/common.py` |
| Main configurable GEMM builder | `/home/arseniivanov/ayla/ayla/gemm.py` |
| PTX lowering | `/home/arseniivanov/ayla/ayla/passes/lower_to_ptx_template.py` |
| PTX code generation | `/home/arseniivanov/ayla/ayla/ptx_codegen.py` |
| CUDA driver runtime | `/home/arseniivanov/ayla/ayla/ptx_runtime.py` |
| Architecture defaults/types | `/home/arseniivanov/ayla/ayla/gpu_dialect.py` |
| Canonical port log | `/home/arseniivanov/ayla/B200_PORT.md` |
| External blockers | `/home/arseniivanov/ayla/BLOCKERS.md` |
| Raw benchmark/profiler artifacts | `/home/arseniivanov/ayla/.benchmarks/` |

The CUDA tools are not necessarily inherited on `PATH`. Invoke the absolute
paths above or prepend `/usr/local/cuda/bin` for the individual command. Keep
Python execution inside Pixi even when NCU is the parent process, for example:

```bash
CUDA_VISIBLE_DEVICES=1 /usr/local/cuda/bin/ncu \
  --target-processes all \
  /home/arseniivanov/ayla/.pixi/envs/default/bin/python \
  benchmarks/bench_gemm_collection.py -k 'candidate-name'
```

### Starting the long-running Codex session

`AGENTS.md` provides persistent repository instructions, but it is not a
substitute for an active task goal. When the CLI supports goals, set one before
starting the weekend run:

```text
/goal Complete the Ayla B200 port and every completion gate in AGENTS.md. Keep working autonomously, maintain B200_PORT.md, and record external blockers in BLOCKERS.md.
```

Then send: `Read AGENTS.md, B200_PORT.md, and BLOCKERS.md and begin the B200 port.`
Use `/goal` to inspect it and `/goal resume` if it was paused. Do not clear or
replace the goal until every completion gate is satisfied.

## Shared GPU etiquette

The eight B200s are shared with other workloads. Immediately before every
benchmark, correctness run that allocates large matrices, or NCU capture:

1. Run `nvidia-smi` and inspect per-GPU utilization, memory use, and active
   compute processes.
2. Select a genuinely idle or low-utilization physical GPU and pin the command
   with `CUDA_VISIBLE_DEVICES=<physical-index>`.
3. Recheck immediately before a long sweep because availability can change
   between experiments.
4. If no GPU is suitable, wait 30-60 seconds and check again. Do not pile work
   onto an occupied GPU, and do not interpret contended timings as performance
   data.
5. Record the physical GPU in `B200_PORT.md` or its linked raw result. Keep
   comparison runs on the same GPU and under comparable clocks/thermal
   conditions.

Treat sustained non-Ayla utilization, an unrelated compute process, or material
memory occupancy as contention. Short correctness smoke tests may use a quiet
GPU, while reported benchmark and profiler results require an uncontended one.

## Required execution order

### 1. Establish a trustworthy B200 baseline

- Run `benchmarks/bench_gemm_collection.py` in the Pixi environment, first with
  the torch-only filter and then as a full collection.
- Record GPU name and compute capability, driver, CUDA toolkit, PyTorch CUDA
  version, PTX ISA/target emitted by Ayla, and the exact failure or baseline.
- Add the official B200 dense BF16 tensor-core peak to the benchmark metadata so
  `% peak` is meaningful. Cite the source in a code comment; do not guess it.
- Keep `torch.mm` in the same NT operand layout, graph-capture mode, stream,
  warmup, and timing harness as Ayla.

### 2. Make architecture support explicit

Remove the assumption that PTX version and target are always Hopper
`sm_90a`. Introduce a small, testable architecture/target configuration used by
module construction and lowering. Preserve the H100 path while adding the B200
path; do not scatter target string checks throughout passes. Branch on B200
compute capability at an appropriate target-selection boundary, then pass an
explicit capability/target description through the compiler and runtime.

First make scalar and basic memory/TMA end-to-end tests load and run on B200.
Then implement the Blackwell tensor-core path and its accumulator/tensor-memory
lifecycle. Expect Hopper `wgmma` lowering and register accumulator layouts to
need a real Blackwell `tcgen05` implementation rather than a target-string
substitution. Use narrowly scoped inline PTX where the IR lacks an operation,
then promote recurring concepts into typed IR operations and passes with tests.

Where cluster launch attributes are required, extend `PtxKernel` and the driver
launch path cleanly (for example with the appropriate extensible CUDA launch
API) instead of hiding launch policy in benchmark scripts.

### 3. Restore correctness before tuning

- Get the complete collection benchmark to run without PTX JIT or launch
  failures. Architecture-incompatible legacy entries must be ported, explicitly
  classified by target, or excluded through honest target-aware registration;
  they must not crash a B200 benchmark presented as supported.
- Add unit tests for every new IR type, verifier rule, transform, PTX form, and
  launch attribute.
- Add small end-to-end tests before relying on 4096-size tests for debugging.
- Validate GEMMs against `torch.mm` for multiple seeds. Use error bounds suitable
  for BF16 accumulation and retain the existing stricter checks wherever they
  pass. Check for NaNs/infinities and test boundary/tail behavior for any newly
  generalized shape.
- After meaningful lowering changes, run the full test suite on B200. Do not
  label expected architecture failures as skips once that feature is claimed
  to be supported.

### 4. Solve and optimize the 4096 x 4096 x 4096 BF16 case

Make 4096 cubed the primary optimization loop. For each candidate:

1. Generate/load it and inspect PTX/JIT diagnostics.
2. Prove correctness.
3. Benchmark it against `torch.mm` with the existing harness.
4. Repeat promising measurements and reject wins within run-to-run noise.
5. Profile only stable, correct candidates with NCU.
6. Form the next hypothesis from measurements, change one coherent dimension,
   and log the result.

Use NCU to distinguish tensor-pipe underutilization, memory/TMA stalls, barrier
latency, insufficient eligible warps, poor occupancy, register pressure/spills,
shared-memory pressure or bank conflicts, launch geometry/wave quantization,
instruction issue limits, and cache behavior. Inspect generated SASS and source
correlation; do not optimize from a single headline utilization number.

Search software-pipeline design space systematically: CTA tile M/N/K, tensor
core instruction shape, CTA/thread count, pipeline depth, operand/output buffer
counts, load/compute/store ordering, output tiles per CTA, accumulator layout,
shared-memory swizzles, TMA transaction shapes, L2 policies, cluster dimensions,
and multicast/multicluster fetching. Automate sweeps where practical, prune
invalid configurations early, and retain reproducible winners in the framework
rather than only in an ad-hoc script.

### 5. Generalize through the square curve

Only after the 4096 case is correct and clearly ahead, run and tune
`benchmarks/benchmark_bf16_square_curve.py` across every configured size from
512 through 16384. Configurations may be shape-dependent, but selection must be
expressed cleanly and reproducibly. Re-run correctness and comparison after
every shape-policy change. Update H100-specific plot titles/metadata as part of
the B200 port.

Use `benchmarks/benchmark_bf16_muse_dynamic_m.py` as an additional regression
and generality check after the required square curve is strong; it does not
replace the 4096 and square acceptance gates.

## Benchmark integrity

- Always compare a correct Ayla result with `torch.mm` on identical operands,
  dtype, layout, stream, graph-capture conditions, and GPU.
- Report median latency and throughput plus dispersion/min/max. Confirm final
  wins in multiple independent benchmark invocations.
- Run competitors in rotating or alternating order to reduce clock and thermal
  bias. Do not compare a cold path with a warmed path.
- Keep compilation/JIT and allocation outside timed regions for both sides.
- Save the exact command and raw CSV/output. A screenshot or one rounded number
  is not evidence.
- If clocks or power states drift, cool down/retry or use paired results; do not
  cherry-pick the fastest sample.

## Completion gates

Do not finish until all of the following are true:

1. Ayla has an explicit, tested B200/Blackwell target path and no claimed B200
   kernel relies on Hopper-only PTX instructions. The existing H100/Hopper path
   remains supported and covered by architecture-specific tests.
2. The full relevant test suite passes on B200, including multi-seed numerical
   correctness for the retained GEMM kernels.
3. `benchmarks/bench_gemm_collection.py` completes end to end on an uncontended
   B200 and reports valid B200 peak percentages.
4. The best Ayla 4096 cubed BF16 kernel beats `torch.mm` throughput by at least
   5% in each of three independent, uncontended runs, with the median advantage
   larger than observed run-to-run noise.
5. `benchmarks/benchmark_bf16_square_curve.py` completes correctly for every
   configured size through 16384. Ayla beats `torch.mm` by at least 5% at 4096,
   8192, and 16384 and has a positive geometric-mean throughput advantage over
   the complete curve in three independent, uncontended runs.
6. NCU evidence explains the final kernel's principal bottleneck and confirms
   that the result is not caused by measurement artifacts or GPU contention.
7. No producer/consumer warp specialization has been introduced anywhere in
   the B200 implementation.
8. Code, tests, benchmark metadata, and concise design notes are coherent enough
   for another engineer to reproduce the result from the repository and Pixi
   environment.

If all gates appear satisfied, perform one final clean verification from a new
process on a freshly rechecked idle GPU. Only then may the task be considered
complete. Otherwise continue working.
