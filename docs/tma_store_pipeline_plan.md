# Two-tile TMA store pipeline

## Goal

Overlap tile 0's asynchronous SMEM-to-GMEM TMA store with computation of tile
1, using the existing IR transforms rather than a kernel-specific persistent
scheduler.

## Implementation

- Reduce the physical grid from 32x16 to 16x16.
- Assign two contiguous logical M tiles to each CTA with hierarchical tiling:
  M=2048 at `for` level, then M=128 at `block.x` level.
- Keep the K=3 input pipeline and one 64 KiB output SMEM buffer.
- Run LICM before fully unrolling the two-iteration output loop.
- Let the existing async scheduler sink store 0's bulk await across tile 1
  compute, stopping before tile 1 overwrites the output SMEM buffer.

The expected flattened order is:

```text
compute tile 0
WRMEM -> output SMEM
issue TMA store 0
compute tile 1
await TMA store 0
WRMEM -> output SMEM
issue TMA store 1
await TMA store 1
```

## Validation status

- 73 non-GPU tests pass.
- CUDA 13 `ptxas` accepts the generated kernel.
- IR tests cover hierarchical tile mapping and bulk-await scheduling.
- PTX test verifies WGMMA work occurs between store 0 issue and wait, and that
  wait 0 precedes the second shared-memory gather.
- GPU correctness and performance remain to be run on an idle H100.

## Remote validation

```bash
git fetch origin
git switch tma-store-pipeline-2tiles
pytest -q tests/gemm_collection/test_correctness.py -k tma_store_k3_2tiles
python benchmarks/bench_gemm_collection.py --filter 'wgmma256 wg'
```

Before benchmarking, require the selected GPU to report 0% compute and 0%
memory-controller utilization. Use `CUDA_VISIBLE_DEVICES` to isolate it.
