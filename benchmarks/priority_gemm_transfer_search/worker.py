"""Time normal production JIT seeds, saving each pair before the next trial."""

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402 -- script path setup precedes local imports
from ayla.gemm_jit import (  # noqa: E402 -- script path setup precedes local imports
    build_gemm_module,
    _seed,
    _DEFAULTS,
    select_gemm_schedule,
    gemm_op,
)
from ayla.ptx_codegen import make_ptx_kernel  # noqa: E402 -- script path setup precedes local imports
from ayla.ptx_runtime import make_ptx_launcher  # noqa: E402 -- script path setup precedes local imports
from benchmarks import common  # noqa: E402 -- script path setup precedes local imports
from benchmarks.benchmark_oss_pytorch_gemm_shapes import _baseline_config  # noqa: E402 -- script path setup precedes local imports
from benchmarks.oss_gemm_corpus import benchmark_shapes, validate_shapes, load_shapes  # noqa: E402 -- script path setup precedes local imports

p = argparse.ArgumentParser()
p.add_argument("job", type=Path)
p.add_argument("output", type=Path)
p.add_argument("--seconds", type=float, default=0.5)
p.add_argument("--rounds", type=int, default=200)
p.add_argument("--confirm", action="store_true")
a = p.parse_args()
job = json.loads(a.job.read_text())
shape = next(
    s
    for s in benchmark_shapes(validate_shapes(load_shapes(HERE / "corpus.json")))
    if s.shape_id == job["target"]
)
common._MIN_RUN_TIME_S = a.seconds
common._MAX_ROUNDS = a.rounds
results = []
baseline = _baseline_config(shape)


def save():
    tmp = a.output.with_suffix(".tmp")
    tmp.write_text(json.dumps(results, indent=2, default=str) + "\n")
    tmp.replace(a.output)


for i, entry in enumerate(job["entries"]):
    start = time.time()
    row = dict(
        target=shape.shape_id,
        op=shape.op,
        dtype=shape.dtype,
        m=shape.m,
        n=shape.n,
        k=shape.k,
        entry=entry,
        status="error",
        seconds=a.seconds,
        round_cap=a.rounds,
        gpu_uuid=os.environ.get("CUDA_VISIBLE_DEVICES", "unknown"),
    )
    try:
        captured = {**_DEFAULTS, **entry["configuration"]}
        seed = _seed(
            captured,
            shape.op,
            shape.m,
            shape.n,
            shape.k,
            (1, shape.m, shape.n, shape.k),
            exact=True,
        )
        if entry.get("installed_route"):
            selected = select_gemm_schedule(
                shape.op, shape.dtype, shape.m, shape.n, shape.k
            )
            assert selected is not None and selected.captured_config == captured
            seed = selected
        assert seed is not None and seed.builder_config
        kernel = make_ptx_kernel(
            build_gemm_module(shape.op, shape.dtype, 1, shape.m, shape.n, shape.k, seed)
        )
        launcher = make_ptx_launcher(kernel)
        config_builder = (
            common.addmm_bench_config
            if shape.op == "addmm"
            else common.gemm_bench_config
        )
        wrapper = (
            common.out_parameter_addmm_op
            if shape.op == "addmm"
            else common.out_parameter_op
        )
        if entry.get("installed_route"):
            launcher = gemm_op(shape.op, shape.dtype)
        bench = config_builder(
            name="Ayla " + entry["label"],
            m=shape.m,
            n=shape.n,
            k=shape.k,
            compute=wrapper(launcher),
            dtype=torch.bfloat16,
        )
        check = (
            common.assert_addmm_against_torch
            if shape.op == "addmm"
            else common.assert_against_torch
        )
        seeds = [int(shape.shape_id[-8:], 16)] + (
            [104729, 15485863] if a.confirm else []
        )
        checks = [
            check(bench, seed=s, max_abs_error=32, max_relative_rms_error=0.002)
            for s in seeds
        ]
        assert all(math.isfinite(v) for c in checks for v in c), checks
        normalized = re.sub(r"//[^\n]*", "", kernel.ptx)
        launch = dict(
            grid_dim=kernel.grid_dim,
            block_dim=kernel.block_dim,
            cluster_dim=kernel.cluster_dim,
            shared_mem_bytes=kernel.shared_mem_bytes,
            signature=[repr(t) for t in kernel.signature],
            ptx_optimization_level=kernel.ptx_optimization_level,
        )
        code_hash = hashlib.sha256(
            (normalized + json.dumps(launch, sort_keys=True)).encode()
        ).hexdigest()
        path = HERE / "kernels" / shape.shape_id
        path.mkdir(parents=True, exist_ok=True)
        (path / (code_hash + ".ptx")).write_text(kernel.ptx)
        row.update(
            configuration=captured,
            builder_configuration=seed.builder_config,
            code_fingerprint=code_hash,
            correctness=checks,
            correctness_seeds=seeds,
            **launch,
        )
        if a.confirm:
            from executable_fingerprint import fingerprint

            row.update(
                fingerprint(
                    kernel,
                    HERE
                    / "executables"
                    / shape.shape_id
                    / (a.output.stem + "_" + code_hash + ".cubin"),
                )
            )
        torch.cuda.empty_cache()
        ref, result = common.run_bench([baseline, bench], inference=True)
        row.update(
            status="ok",
            torch_ms=ref.median_ms,
            ayla_ms=result.median_ms,
            torch_pct=100 * ref.median_ms / result.median_ms,
            rounds=result.rounds,
            layers=result.iters,
            stdev_ms=result.stdev_ms,
            mean_ms=result.mean_ms,
        )
        del bench, launcher, kernel, seed
    except Exception as e:
        row.update(
            error=type(e).__name__ + ": " + str(e), traceback=traceback.format_exc()
        )
    row["elapsed_seconds"] = time.time() - start
    results.append(row)
    save()
    print(
        json.dumps(
            {
                k: row.get(k)
                for k in (
                    "target",
                    "status",
                    "torch_pct",
                    "ayla_ms",
                    "elapsed_seconds",
                    "error",
                )
            }
            | {"label": entry["label"]}
        ),
        flush=True,
    )
    gc.collect()
    torch.cuda.empty_cache()
