# BF16 4K comparison against `perf/four-stage-split-output`

This capture uses `perf/four-stage-split-output@e8ea520` as the historical
baseline. It does not use `perf/output-cache-sass-ir` as the reference.

## Fairness protocol

`benchmarks/benchmark_bf16_4k.py` applies the same execution protocol to Ayla
and `torch.mm`:

- both multiply the same BF16 input and the same transposed weight;
- both are captured in CUDA graphs with the same number of GEMM launches;
- both graphs are captured, replayed, and timed on the same explicit PyTorch
  current CUDA stream;
- CUDA events are recorded on that same stream;
- Ayla-first and torch-first measurements alternate each round;
- cold measurements perform the same 240 MiB L2 eviction on the shared stream
  before each single-GEMM graph replay; and
- medians, rather than minimum timings, determine reported throughput.

The graph protocol removes Python dispatch, output allocation, and descriptor
setup from both timed paths. No Ayla-only prepared-launch cache is included in
the library stack.

## Cold-cache results

| Revision/configuration | Ayla TFLOP/s | torch.mm TFLOP/s | Ayla vs torch |
|---|---:|---:|---:|
| `perf/four-stage-split-output`, four-tile K3 | 747.86 | 755.89 | -1.06% |
| New stack, matching four-tile K3 | 748.64 | 750.34 | -0.23% |

The matching K3 implementation differs from the historical branch by 0.10%
in absolute Ayla throughput. The difference in the relative percentages comes
from `torch.mm` moving between the two separate process runs, not from lost
Ayla throughput. The CSV records latency and throughput deltas separately to
avoid sign ambiguity: negative latency is better, while positive throughput is
better.
