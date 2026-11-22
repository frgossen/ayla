# Independent compute warpgroups on Hopper

`warpgroup_ctas=2` pairs two independently progressing 128-thread task programs
in one 256-thread CTA. Each computes its own output tile; both compute, convert
and write back. There is no dedicated producer warp.

The operand ring is shared. A stage passes to the peer only after the last MMA
read completes. Output scratch and mbarriers remain private to each program.
When the K-step count is not divisible by the ring depth, both stage pointers
and handoff events rotate across tasks. This allows five stages with 32 K steps.
`QueueAsyncTail` keeps one safe MMA group pending through peeled loop tails.

Warp-owned eight-row fragments convert and issue TMA stores through private
shared slices. Completion tracks each source buffer before reuse. FP8 operand
swizzles use byte widths; their element widths differ from FP16/BF16.

Pairing is available through `TileMatmulPipeline`, the collection GEMM builder
and the measured FP16/BF16 JIT routes. It requires one warpgroup per tile, an
even fixed virtual grid, complete waves and disjoint output storage. Clusters,
Split-K, batches and outer output-loop pipelining are excluded. `PipelineLoop`
can reorder an authored stream; pairing additionally constructs independent
ownership, private completion state and per-slot handoffs.

`ayla.fp8.scaled_mm(a, weight.T, scale_a, scale_b)` accepts E4M3FN/E5M2 inputs,
FP32 scalar CUDA dequantization scales, and BF16 (default), FP16 or FP32 output.
It supports CUDA graph capture and `torch.compile`.

Measured on H100, unlocked clocks, 40 rounds of 150 graph launches:

| Type | M × N × K | Selected latency | Speedup over preceding Ayla schedule | Speedup over cuBLAS |
|---|---|---:|---:|---:|
| FP16 | 2048 × 11008 × 4096 | 274.27 µs | 1.2024× | 1.0827× |
| BF16 | 2048 × 11008 × 4096 | 248.71 µs | 1.5947× | 1.0982× |
| FP16 | 4096 × 4096 × 4096 | 209.61 µs | 1.0808× | 1.0367× |
| BF16 | 4096 × 4096 × 4096 | 199.38 µs | 1.0326× | 1.0430× |

The FP8 wide MLP up schedule uses M128×N168, K128, five rotating operand
stages and one output arena. The preceding four-stage pairing measured
0.992–0.999× cuBLAS across two H100s; five rotating stages improved it by
another 2.4–2.7%. Unlocked clocks make these measurements run-dependent.

Reproduce the generic comparison with:

```bash
CUDA_VISIBLE_DEVICES=0 python -m benchmarks.benchmark_warpgroup_ctas \
  --dtype bfloat16 --rounds 40 --launches 150 --output /tmp/paired-bf16
```

The benchmark compares matched unpaired/paired geometries, selected library
routes and `torch.mm`, validates outputs and reports register/shared/local
memory resources. `--baseline` accepts a frozen preceding schedule to reproduce
the historical Ayla comparison.
