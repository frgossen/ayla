# H100 kernel extrapolation workspace

This workspace gives the Ayla, Triton, and CUTLASS agents isolated source trees, common problem definitions, and one paired benchmark protocol. The original challenge descriptions are in `ABLATIONS.md`.

## Layout

| Path | Purpose |
| --- | --- |
| repository root | Current Ayla implementation used by the Ayla reference |
| `agents/triton/` | Triton submissions and lightweight results |
| `agents/cutlass/` | CUTLASS submissions and lightweight results |
| `agents/*-upstream/` | Optional pinned upstream clones, never committed |
| `references/` | Runnable PyTorch, Ayla, Triton, and CUTLASS reference adapters |
| `benchmark/problems.py` | Canonical shapes, physical layouts, outputs, FLOP counts, and correctness checks |
| `benchmark/run.py` | Paired, alternating CUDA-event benchmark harness |
| `benchmark/profile.py` | Single-candidate NVTX entry point for Nsight Compute |
| `results/` | Baseline JSON measurements and Triton configuration sweep |
| `agent_instructions/` | Library-specific handoff notes |

The Python environment is `/home/arseniivanov/ayla/.pixi/envs/default`. Use its Python explicitly so all agents use the same Torch, CUDA bindings, Triton, and CUTLASS DSL packages.

The tools branch intentionally does not vendor the Triton or CUTLASS source
trees, compiled kernels, profiler reports, or caches.  Materialize the pinned
read-only upstream checkouts when needed:

```bash
GIT_HTTP_PROXY=fwdproxy:8080 extrapolation/bootstrap_dependencies.sh
```

## Canonical tensor layout

For every GEMM, B is physically a contiguous `[N, K]` row-major tensor. This has the same storage as a mathematical `[K, N]` column-major tensor. Candidates compute `A @ B.T`; they must not materialize the transpose.

The four registered benchmark problems are:

- `bf16_gemm`: BF16 `[4096,4096] @ [4096,4096].T`, BF16 output.
- `dual_gemm`: BF16 `[4096,4096] @ [128,4096].T`, then the BF16 intermediate multiplies `[4096,128].T`.
- `softplus`: the 4K GEMM followed by stable FP32 softplus and a BF16 output.
- `dynamic_quant`: the 4K GEMM followed by a row maximum, FP32 scale, and INT8 output.

`benchmark/problems.py` is authoritative when prose and code disagree.

## Candidate interface

A candidate is a Python file with:

```python
NAME = "unique_name"
PROBLEM = "softplus"
METADATA = {"backend": "example", "tile": [128, 256]}

def prepare(inputs):
    # Compile and allocate outputs here. This is outside all timings.
    def run():
        # Launch only the operation being compared.
        return output
    return run
```

Never allocate outputs, compile, autotune, or synchronize inside `run()` unless the problem itself requires that work. A multi-kernel PyTorch baseline intentionally includes every kernel launched by one call to `run()`.

## Fair benchmark protocol

First choose an actually idle physical GPU:

```bash
/home/arseniivanov/ayla/.pixi/envs/default/bin/python benchmark/select_gpu.py
nvidia-smi -i 3
```

Then run from `/home/arseniivanov/extrapolation`:

```bash
/home/arseniivanov/ayla/.pixi/envs/default/bin/python benchmark/run.py \
  --problem bf16_gemm \
  --baseline references/torch_bf16_gemm.py \
  --candidate references/ayla_bf16_gemm.py \
  --candidate references/triton_bf16_gemm.py \
  --candidate references/cutlass_bf16_gemm.py \
  --gpu 3 --warmup 300 --batch 20 --pairs 60 --bootstrap 4000 \
  --output results/my_run.json
```

The harness:

- rejects a GPU above 5% utilization or 1 GiB allocated memory before importing CUDA;
- compiles and checks every arm before timing;
- warms arms round-robin, reversing their order every pass;
- reverses candidate order every paired round to cancel thermal drift;
- alternates baseline-first and candidate-first separately for each candidate;
- reports median runtime, TFLOP/s, paired percentage delta, wins, and a bootstrap 95% interval;
- records physical GPU telemetry, clocks, power limit, software versions, source files, and candidate metadata in JSON.

Use the paired delta to decide a winner. Do not compare two unrelated single-run averages. A useful performance claim should reproduce in a second process and its confidence interval should normally exclude zero.

## Free-clock and fixed-clock runs

Report both when a difference is small. Free clocks represent normal operation, but a different instruction mix can change power and therefore boost clocks. A fixed clock below the power limit exposes architectural runtime more cleanly.

GPU 3 was below its 500 W limit at 1050 MHz during the recorded run:

```bash
sudo nvidia-smi -i 3 -lgc 1050,1050
trap 'sudo nvidia-smi -i 3 -rgc' EXIT
/home/arseniivanov/ayla/.pixi/envs/default/bin/python benchmark/run.py \
  --problem bf16_gemm --baseline references/torch_bf16_gemm.py \
  --candidate path/to/candidate.py --gpu 3 \
  --output path/to/results.json
sudo nvidia-smi -i 3 -rgc
```

Always restore clocks, including after a failed command. Do not alter a power limit for ordinary comparisons.

## Profiling and SASS

The profiling entry point performs correctness and warm-up before placing only measured launches in the `candidate` NVTX range:

```bash
/usr/local/cuda-13.0/bin/ncu --target-processes all --nvtx \
  --nvtx-include 'candidate/' \
  --section LaunchStats --section SpeedOfLight --section SchedulerStats \
  --section WarpStateStats --section MemoryWorkloadAnalysis \
  --export agents/ayla/results/profile \
  /home/arseniivanov/ayla/.pixi/envs/default/bin/python benchmark/profile.py \
  --problem bf16_gemm --candidate references/ayla_bf16_gemm.py --gpu 3
```

Useful installed tools are:

- `/usr/local/cuda-13.0/bin/ncu` for counters and annotated SASS;
- `/usr/local/cuda-13.0/bin/cuobjdump --dump-sass` for cubins and executables;
- `/usr/local/cuda-13.0/bin/nvdisasm` for cubin disassembly and control-flow views;
- `/usr/local/cuda-13.0/bin/ptxas -v` for register, shared-memory, and spill reporting;
- `/home/arseniivanov/ayla/tools/` for the existing PTX/SASS utilities and energy benchmark scripts.

Inspect SASS, not PTX alone. PTXAS may fold address arithmetic, remove dead work, or introduce register pressure that changes the conclusion.

For live clocks and power during a longer run:

```bash
nvidia-smi dmon -i 3 -s pucmt
```

## Validated BF16 references

The references use the retained Ayla 4-tile kernel, Triton's official Hopper TMA-persistent tutorial design, and CUTLASS 4.7's official Hopper persistent CuTe DSL kernel. All four outputs, including cuBLAS, were bitwise identical for seed 42.

The full paired results currently saved are:

| Regime | Ayla vs cuBLAS | Triton vs cuBLAS | CUTLASS vs cuBLAS |
| --- | ---: | ---: | ---: |
| Free clock, 500 W limit | +0.67% | +1.67% | -1.06% |
| Fixed 1050 MHz, run A | +1.08% | +1.50% | -0.36% |
| Fixed 1050 MHz, run B | +0.38% | +0.86% | -0.58% |

Negative is faster. See `results/bf16_gemm_free_clock.json`, `results/bf16_gemm_1050mhz.json`, and `results/bf16_gemm_1050mhz_repeat.json` for raw paired measurements and telemetry. Fixed-clock process-to-process variation is larger than the within-process bootstrap intervals, which is why both fixed runs are retained. The Triton reference's default `128x256x64`, three-stage, eight-warp configuration was the fastest valid point in the official 24-point Hopper search space tested here; invalid shared-memory configurations were excluded.

## Agent isolation

Read `AGENTS.md` before making changes. Each agent owns exactly one tree under `agents/`; shared references, benchmark code, results, and the other two agent trees are read-only during the competition.
