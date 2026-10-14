# Confirmed high-impact BF16 addmm configurations

The six accepted configurations from the high-impact search are merged into
this branch's existing `ayla/oss_gemm_heuristic.json`. BF16 addmm has independent
`addmm/bfloat16/k128` and `addmm/bfloat16/k64` groups. Every pre-existing FP16
operation group and seed is preserved. The scalar fallback is preserved.
The new groups choose each of the six measured signatures explicitly and use
that fallback for other signatures.

Branch: `jit-gemm-heuristics`.

The same six signatures are updated in `ayla/gemm_schedule_table.py`, using five
new configuration rows. Original seeds remain as fallbacks; all unrelated
shape/dtype routes and original configuration rows are preserved. Exact JSON
rows are also added for these six BF16 addmm signatures. JSON and JIT-table
selection preserve all measured builder controls and post-builder passes.

## Source measurements

These are source-study measurements, not a fresh benchmark of this branch.
They were confirmed with the production JIT compiler at `740afb8`, using the
normal inference-v2 Torch/Ayla pair, eight GPU replicas per fixed configuration,
three correctness seeds per replica, alternating incumbent/candidate order,
a two-second adaptive sampling budget, and a 1,000-round cap. Source result
commit: `a1ef578fc1160c07d964b2c7330a05d583a7cd2a`.

The search attempted 18,352 distinct candidate tuples over eight exact shapes.
Six pass the 1% median gain, six-positive-replica, positive-latency-gain,
compiled-kernel-identity and Torch-reference-stability gates. The other two
source-study incumbents were retained. The extra finalist changed-weights graph
check was cancelled at the user's GPU-stop request. No GPU work was restarted
for this integration.

| Addmm M×N×K | Source incumbent % of Torch | Source best % of Torch | Source decision |
| --- | ---: | ---: | --- |
| 4096×4096×4096 | 101.871 | 106.001 | accepted |
| 4096×16384×4096 | 99.142 | 99.142 | retained |
| 4096×4096×16384 | 100.182 | 100.182 | retained |
| 175360×3072×768 | 96.605 | 100.037 | accepted |
| 175360×2304×768 | 96.406 | 99.563 | accepted |
| 175360×768×768 | 88.044 | 97.059 | accepted |
| 175360×768×3072 | 93.390 | 99.897 | accepted |
| 8192×4096×1024 | 92.617 | 103.622 | accepted |
| **Eight-shape geomean** | **95.937** | **100.654** | **+4.917% relative** |

The geomean is equally weighted over those eight source-study shapes. It does
not update a full-corpus or native-backend geomean. Heuristic
`source_percentage_of_torch` values identify the original production-JIT
measurements; benchmark callers must time their own selected configuration.
Compiler, runtime, synchronization and timing code are unchanged by this update.
