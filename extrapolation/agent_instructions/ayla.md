# Ayla agent

Your writable tree is `agents/ayla`. The validated BF16 reference imports the current Ayla checkout from the repository root; the historical fused submissions are retained here without duplicating the whole Ayla source tree.

Start from `references/ayla_bf16_gemm.py` and the corresponding builder under the repository's `ayla/gemm_collection/`. Put new candidate adapters and results under your tree, for example `agents/ayla/submissions/` and `agents/ayla/results/`.

Use Ayla's PTX generation and launcher in `prepare()`. Validate generated PTX with PTXAS, inspect the cubin's SASS, and check registers/spills before timing. The shared Pixi Python automatically builds the existing runtime extension once Ninja is on PATH; the harness handles that path.

Run the common harness from `/home/arseniivanov/extrapolation`; you are only allowed to edit the ayla library if the functionality you need does not exist, or if there are inefficiencies in the current implementation.
