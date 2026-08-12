# Triton agent

Your writable tree is `agents/triton`. The optional pinned, read-only clone is `agents/triton-upstream`; create it with `bootstrap_dependencies.sh`. The installed Pixi package is Triton 3.7.1; the clone identifies as 3.8.0, so use the installed package for common measurements unless you explicitly build the clone and document the change.

The validated starting point is `references/triton_bf16_gemm.py`, adapted from `python/tutorials/09-persistent-matmul.py`. Its selected Hopper configuration is block `128x256x64`, three stages, eight warps, group-M eight. The official 24-point space was smoke-swept; this was the fastest valid configuration for the adapter.

Put candidate files, Triton caches, dumps, and JSON under your tree, for example `agents/triton/submissions/`, `agents/triton/.triton-cache/`, and `agents/triton/results/`. Keep autotuning outside timed calls and save the selected configuration in `METADATA`.

Run the common harness from `/home/arseniivanov/extrapolation`; do not modify the shared reference.
