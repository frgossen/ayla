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

## Verification

Both branches passed Ruff and Pyright on PyTorch 2.13/CUDA 13.0 with H100 GPUs.

| Branch | GPU suite | Benchmark checks |
| --- | --- | --- |
| `feature/consolidated` | 777 passed, no skips | OSS MM/ADDMM/BMM samples passed numerical checks; 4096 BF16 collection comparison completed |
| `feature/gemm-chains` | 816 passed, no skips | SwiGLU M=32 passed numerical/finite-output checks; eager, compiled, and Ayla hot-cache timings completed |

FP16/BF16 MM, ADDMM, and BMM routing and CUDA graph replay are covered on both
branches. The development implementation matches the pre-split snapshot,
except removal of one duplicate package export.

Consolidated sample medians: MM 0.0102 ms, ADDMM 0.0091 ms, BMM 0.0071 ms.
The 4096 BF16 collection sample measured Ayla 0.1973 ms versus Torch 0.2019 ms.
SwiGLU M=32 measured Ayla 0.3848 ms versus compiled Torch 0.4032 ms. These are
smoke measurements, not a rerun of the full OSS or better-benchmark corpus.

Exact results are stored in `/home/arseniivanov/ayla/.git/pr-triage/split-*`.
The previous full integration evidence remains in `feature-*` files there.

## Forking

Fork when independent APIs, releases, maintenance, or push authority require
it. A busy reviewer alone does not require a fork. Either prepared feature
branch can be pushed to a fork while keeping upstream as a remote.
