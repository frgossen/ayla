# Benchmark PR stack

The benchmark capture is intentionally separated from production-library
changes. The local stack, from bottom to top, is:

1. `pr67`
2. `benchmarks/common-harness-pr67` (`31b9c20`)
3. `perf/generalize-tma-gemm-on-common` (`df17320`)
4. `perf/four-stage-configs-on-common` (`eaa7814`)
5. `perf/rectangular-tma-gemm-after-padding-cleanup` (`8a1bad8`)
6. `benchmarks/four-stage-capture-after-padding-cleanup`

## Shared benchmark harness

The first PR is the PR81-inspired benchmark refactor. It moves timing,
statistics, comparison columns, and table formatting into `benchmarks/common.py`
and keeps median-derived throughput. The harness additionally captures every
implementation into equivalent CUDA graphs, shares tensors for identical
shapes, uses one explicit PyTorch CUDA stream, and rotates measurement order to
avoid favoring the implementation measured first.

## Production changes outside the benchmark PR

### Generalized TMA GEMM builder

The first library PR replaces the fixed K3-only module with the configurable
builder from `perf/four-stage-split-output`. It supports the four-stage branch's
output tile counts and dimensions, warp-group partitioning, operand stage and K
tile choices, output slicing/buffering, and scalar or STMatrix output stores.
It preserves the two operators already exposed by `pr67`.

The port also adapts the prototype to the current stack: launch dimensions are
inferred instead of hard-coded, blocked layouts use the current representation,
operand buffers are explicitly multi-buffered, and unrolled K-tail loops use
the current labels. This PR passes independently.

### Four-stage configurations

The second library PR only registers the two split-output configurations needed
to complete parity with `perf/four-stage-split-output`, including its four-tile
K3 configuration with the LHS `evict_first` cache hint. Keeping registration
separate makes the performance-affecting endpoint explicit and easy to compare.

This PR also removes `output_tma_inner_padding`. Padding belonged to the
prototype's old physical-layout representation; the split stack replaced that
representation with explicit block width and ordering, leaving the parameter
with no effect.

### Rectangular dimensions

The third library PR adds independent M, N, and K dimensions to the same
builder so the benchmark-only PR can exercise square sizes and Muse projection
shapes. It does not add the later output-cache scheduling, descriptor, JIT, or
prepared-launch optimizations, and it does not change the registered 4K
configuration.

## Deliberately excluded changes

The changes after `perf/four-stage-split-output` on
`perf/output-cache-sass-ir` are not part of this stack. In particular, this
excludes the later output-arena/cache-policy configuration, WGMMA descriptor
precomputation and PTX instruction reductions, CUDA Driver JIT tuning, and the
Ayla-only prepared-launch cache used during the initial investigation. Those
are production optimizations and should be reviewed in their own subsequent
PRs.

The benchmark capture instead puts both Ayla and `torch.mm` in equivalent CUDA
graphs on the same PyTorch CUDA stream, so it does not require an Ayla-specific
launch-path optimization.
