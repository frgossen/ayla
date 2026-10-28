# Consolidated branches

## Scope

- `feature/consolidated`: MM, ADDMM, BMM, measured schedule lookup, OSS search,
  better-benchmark tooling, and the library performance improvements they need.
  Native-layout FP16/BF16 GEMMs use direct Inductor/JIT routing.
- `feature/gemm-chains`: branched from the narrowed consolidated branch. Adds
  generic FX discovery/import/registration, explicit kernel workspaces,
  standalone register writeback, SwiGLU, and GEMM-to-GEMM development.

Multicast, tiling, output staging, traversal, asynchronous scheduling, matmul
initialization, scale-D, and rank-N TMA remain on consolidated: single-GEMM
schedules use them. The existing single-GEMM example kernels remain available.
Consolidated does not install graph-region or SwiGLU rewrites; Torch handles
other operations around each supported GEMM.

The split preserves published ancestry through ordinary commits. Removed code
remains in historical commits, and is restored on the development branch.
`backup/consolidated-before-split` preserves the original `839ab19` snapshot.
The original dirty worktree was preserved.

PR routing is recorded in `pr-consolidation.json`. #351, #356, and #498 split
across branches: single-GEMM behavior is retained on consolidated, while their
FX implementation is kept on development.

## Heuristic cleanup

Both branches use `ayla.gemm_jit.select_gemm_schedule` and the single measured
`ayla/gemm_schedule_table.py`. The legacy decision tree, minimal JSON, seed
benchmark, mapper, converters, and old search drivers were removed. Archived
search evidence is preserved in `.git/pr-triage/legacy-heuristic-archive/`.
All 1,180 measured shapes and 30 additional queries retain identical selections
and builder configurations. Current checks are in `.git/pr-triage/one-heuristic-*`.

## Push

```bash
git -C /home/arseniivanov/ayla-feature-consolidated push -u origin feature/consolidated
git -C /home/arseniivanov/ayla-gemm-chains push -u origin feature/gemm-chains
```

## PR cleanup

Keep #81, #196, #239, #257, #292, #312, #432, #485, and #487 open. They were
held out of the original integration; this split does not add them. Close
superseded feature PRs only once both branches are safely pushed. Their code
being stored on feature branches does not mean it was merged into main.

## Initial split verification

The narrowed consolidated branch passed all 777 GPU tests, Ruff, and Pyright.
FP16/BF16 MM, ADDMM, and BMM routing and CUDA graph replay are covered.

Current split results and commands are stored in
`/home/arseniivanov/ayla/.git/pr-triage/split-*`. The previous combined snapshot
passed 816 GPU tests, Ruff, and Pyright. Its benchmark evidence remains in
`feature-benchmark-gemm.txt` and `feature-benchmark-swiglu.{txt,json}` there.

## Forking

Fork when independent APIs, releases, maintenance, or push authority require
it. A busy reviewer alone does not require a fork. Either prepared feature
branch can be pushed to a fork while keeping upstream as a remote.
