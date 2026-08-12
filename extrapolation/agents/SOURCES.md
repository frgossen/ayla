# Source manifest

- Ayla: repository root, including `perf/output-cache-sass-ir` commit `41291f5`.
- Triton: optional `agents/triton-upstream` clone at `faa8c288ddca78215fa130d6850a987711e06f1c` (source version 3.8.0). The shared Pixi environment currently imports Triton 3.7.1.
- CUTLASS: optional `agents/cutlass-upstream` clone at `dcf215af68a2d08d305076c152a06f201728cd53` (CUTLASS 4.7.0).
- CUTLASS DSL package: `nvidia-cutlass-dsl` 4.7.0 installed in the shared Pixi environment.
- PyTorch: 2.13.0+cu130.
- Nsight Compute: 2025.3.1.0.

The runnable reference adapters live in `references/`. Their module metadata is written into every new benchmark JSON.
