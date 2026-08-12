# LLM Agent Generation Specification: Advanced BF16 H100 Kernels

This document provides plain-text specifications for three advanced compute-bound GEMM kernels targeting the NVIDIA H100 (Hopper) architecture. These configurations are designed to stress-test LLM agents on their ability to handle WGMMA pipelining, complex register-level intra-warp reductions, and custom mathematical epilogues in `bfloat16`.

## 1. Tile-Fused Dual GEMM ($A \cdot B_1 \cdot B_2$)

### 1.1 Problem Definition
In standard transformer architectures, down-projections or sequential linear layers often force an intermediate write to High Bandwidth Memory (HBM). This kernel fuses two GEMMs into a single threadblock execution. The intermediate tensor $C_1 = A \cdot B_1$ must remain entirely in Shared Memory (SMEM) or registers before being consumed as the input for the second matrix multiplication with $B_2$.

### 1.2 Tensor Shapes & Datatypes (4K Scale)
*   **Tensor A:** `[M, K1]` $
ightarrow$ `[4096, 4096]`, `bfloat16`, Row-Major
*   **Tensor B1:** `[K1, N1]` $
ightarrow$ `[4096, 128]`, `bfloat16`, Column-Major
*   **Tensor B2:** `[N1, N2]` $
ightarrow$ `[128, 4096]`, `bfloat16`, Column-Major
*   **Output D:** `[M, N2]` $
ightarrow$ `[4096, 4096]`, `bfloat16`, Row-Major

*Note: $N_1 = 128$ is deliberately small so that block-tiles of $C_1$ fit within Hopper's 228 KB SMEM limit per Streaming Multiprocessor.*

### 1.3 Hardware Constraints & Generation Targets
*   **WGMMA Chaining:** The agent must generate `wgmma.mma_async` instructions for $A \cdot B_1$, wait for completion (`wgmma.commit_group.sync.aligned`), and immediately route the accumulators into SMEM.
*   **TMA Integration:** TMA must be configured for double-buffered asynchronous reads of $A$, $B_1$, and $B_2$, but bypassed for the intermediate $C_1$.
*   **Failure Modes:** Standard templates will attempt to write $C_1$ to HBM and launch a second grid. The LLM must override the block scheduler.

---

## 2. GEMM with Stable Softplus Epilogue

### 2.1 Problem Definition
Computes a dense matrix multiplication followed by a numerically stable Softplus activation. Because $\exp(x)$ overflows for large inputs in `bfloat16` and `fp32`, the epilogue must dynamically branch per element.

### 2.2 Tensor Shapes & Datatypes (4K Scale)
*   **Tensor A:** `[M, K]` $
ightarrow$ `[4096, 4096]`, `bfloat16`, Row-Major
*   **Tensor B:** `[K, N]` $
ightarrow$ `[4096, 4096]`, `bfloat16`, Column-Major
*   **Scalar $\tau$:** Temperature parameter, `float32`.
*   **Output C:** `[M, N]` $
ightarrow$ `[4096, 4096]`, `bfloat16`, Row-Major

### 2.3 Mathematical Operation
For each element $x$ in the accumulator (cast to `float32`):
$y = 	ext{Softplus}(x / 	au) = egin{cases} 	au \cdot \log(1 + \exp(x / 	au)) & 	ext{if } x / 	au \le 20 \ x & 	ext{if } x / 	au > 20 \end{cases}$

### 2.4 Hardware Constraints & Generation Targets
*   **Transcendental Masking:** The `tl.where` (Triton) or EVT node (CUTLASS) must avoid evaluating `exp` on the out-of-bounds register variables to prevent infs/NaNs propagating down the hardware pipeline.
*   **Latency Hiding:** Transcendental operations are slow on Hopper. The agent must successfully interleave epilogue math with TMA stores.

---

## 3. GEMM + Channel-wise Dynamic Quantization Epilogue

### 3.1 Problem Definition
A critical operation for LLM inference (e.g., FP8/INT8 weight-only or dynamic activation pipelines). The kernel computes a standard `bfloat16` GEMM, but instead of storing `bfloat16`, it computes the maximum absolute value per row, derives a scale factor, quantizes the row to `int8`, and outputs both the quantized matrix and the scale factors.

### 3.2 Tensor Shapes & Datatypes (4K Scale)
*   **Tensor A:** `[M, K]` $
ightarrow$ `[4096, 4096]`, `bfloat16`, Row-Major
*   **Tensor B:** `[K, N]` $
ightarrow$ `[4096, 4096]`, `bfloat16`, Column-Major
*   **Output Y:** `[M, N]` $
ightarrow$ `[4096, 4096]`, `int8`, Row-Major
*   **Output Scales:** `[M, 1]` $
ightarrow$ `[4096, 1]`, `float32`, Column-Major

### 3.3 Mathematical Operation
1.  Compute accumulator row $i$: $C_i = A_i \cdot B$ (in `float32`).
2.  Compute scale: $S_i = \max(\operatorname{abs}(C_i)) / 127.0$.
3.  Quantize: $Y_i = 	ext{round}(C_i / S_i)$.

### 3.4 Hardware Constraints & Generation Targets
*   **Two-Pass Register Read:** Accumulator registers on H100 are heavily fragmented across the warp. The agent must coordinate a spatial `Redux` (reduction) operation across the warp to find the max absolute value *before* it can scale and cast the individual elements.
*   **Dual Output Pipes:** The kernel must handle storing a 2D matrix ($Y$) and a 1D vector ($S$) simultaneously at the end of the pipeline.
