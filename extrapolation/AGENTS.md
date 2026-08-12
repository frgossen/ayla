# Agent rules

Read `README.md`, `ABLATIONS.md`, `benchmark/problems.py`, and your matching file in `agent_instructions/` before working.

## Ownership

- The Ayla agent may write only under `agents/ayla/`.
- The Triton agent may write only under `agents/triton/`.
- The CUTLASS agent may write only under `agents/cutlass/`.
- `benchmark/`, `references/`, `results/`, the top-level documents, and the other agents' trees are read-only.

Save candidate modules, build output, notes, and measurements inside your own tree. Do not copy, import, inspect generated artifacts from, modify, stage, stash, clean, reset, or check out files in another agent's tree. Do not switch the pinned upstream revision. Never run a repository-wide Git command from `agents/` or the extrapolation root; set the working directory to your own tree explicitly.

You may read shared benchmark/reference files and upstream source in your own tree. Do not solve a task by calling another library's submitted kernel.

## Required workflow

1. Confirm the selected physical GPU is idle with `benchmark/select_gpu.py` and `nvidia-smi`.
2. Implement the candidate contract documented in `README.md` in your own tree.
3. Run correctness through `benchmark/run.py` before profiling or claiming performance.
4. Compile and autotune in `prepare()`, outside the timed `run()` closure.
5. Use the paired alternating harness with at least 300 warm-ups, 20 launches per sample, and 60 paired samples for a final number.
6. Reproduce small wins in a fresh process. Test both normal clocks and a fixed clock below the power limit.
7. Inspect resource use and actual SASS. Report registers, spills, shared memory, launch geometry, and important NCU counters with runtime.
8. Put all logs and JSON in `agents/<your-library>/results/`.

Do not use `--allow-busy` for a result. Do not change GPU power limits. If you lock clocks, install a shell trap first and restore them explicitly afterward. Do not leave background monitors or profiler processes running.

Do not use Streams, Cuda streams, or any kind of graph replay. This is host-side optimization that can be done regardless of framework and should not be included.

The output must implement the exact semantics in `benchmark/problems.py`. Relaxing correctness, changing the seed or physical layout, materializing a transpose outside the timing, returning a cached answer, or excluding required epilogue launches invalidates a result.
