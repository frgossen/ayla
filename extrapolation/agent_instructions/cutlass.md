# CUTLASS agent

Your writable tree is `agents/cutlass`. The pinned, read-only CUTLASS 4.7 checkout is `agents/cutlass-upstream`; create it with `bootstrap_dependencies.sh`. The matching `nvidia-cutlass-dsl` 4.7.0 package is installed in the shared Pixi environment.

The validated starting point is `references/cutlass_bf16_gemm.py`. It wraps NVIDIA's official Hopper persistent CuTe DSL example at `examples/python/CuTeDSL/cute/hopper/kernel/dense_gemm/dense_gemm_persistent.py` with a `128x256` CTA tile, `1x1` cluster, swizzle one, and M raster.

Put candidate adapters, build output, CuTe cache files, and JSON under your tree, for example `agents/cutlass/submissions/`, `agents/cutlass/build-agent/`, and `agents/cutlass/results/`. Compilation belongs in `prepare()`. Preserve B as physical `[N,K]`; do not pay for a transpose outside timing.

Run the common harness from `/home/arseniivanov/extrapolation`; do not modify the shared reference or import another agent's candidate.
