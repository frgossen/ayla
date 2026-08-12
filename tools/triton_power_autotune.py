#!/usr/bin/env python3
"""Autotune the Hopper TMA-persistent BF16 4096 GEMM from a clean cache.

The search space mirrors Triton's persistent-matmul tutorial.  B is stored as
[N, K], so the operation is A @ B.T.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


def descriptor_hook(args):
    bm = args["BLOCK_M"]
    bn = args["BLOCK_N"]
    bk = args["BLOCK_K"]
    args["a_desc"].block_shape = [bm, bk]
    args["b_desc"].block_shape = [bn, bk]
    args["c_desc"].block_shape = [bm, bn // 2 if args["EPILOGUE_SUBTILE"] else bn]


CONFIGS = [
    triton.Config(
        {
            "BLOCK_M": 128,
            "BLOCK_N": bn,
            "BLOCK_K": bk,
            "GROUP_M": 8,
            "EPILOGUE_SUBTILE": subtile,
        },
        num_stages=stages,
        num_warps=warps,
        pre_hook=descriptor_hook,
    )
    for bn in (128, 256)
    for bk in (64, 128)
    for stages in (2, 3, 4)
    for warps in (4, 8)
    for subtile in (False, True)
]


@triton.jit
def _pid(tile_id, num_pid_in_group, num_pid_m, GROUP_M: tl.constexpr):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + tile_id % group_size_m
    pid_n = tile_id % num_pid_in_group // group_size_m
    return pid_m, pid_n


@triton.autotune(configs=CONFIGS, key=["M", "N", "K"])
@triton.jit
def persistent_bf16(
    a_desc,
    b_desc,
    c_desc,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
):
    start_pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n
    num_pid_in_group = GROUP_M * num_pid_n
    tile_id_c = start_pid - NUM_SMS

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        pid_m, pid_n = _pid(tile_id, num_pid_in_group, num_pid_m, GROUP_M)
        offset_m = pid_m * BLOCK_M
        offset_n = pid_n * BLOCK_N
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, K, BLOCK_K):
            a = a_desc.load([offset_m, k])
            b = b_desc.load([offset_n, k])
            accumulator = tl.dot(a, b.T, accumulator)

        # Use a separate induction variable so the epilogue can be pipelined
        # independently from the next tile's prologue.
        tile_id_c += NUM_SMS
        out_m, out_n = _pid(tile_id_c, num_pid_in_group, num_pid_m, GROUP_M)
        out_m *= BLOCK_M
        out_n *= BLOCK_N
        if EPILOGUE_SUBTILE:
            acc = tl.reshape(accumulator, (BLOCK_M, 2, BLOCK_N // 2))
            acc = tl.permute(acc, (0, 2, 1))
            acc0, acc1 = tl.split(acc)
            c_desc.store([out_m, out_n], acc0.to(tl.bfloat16))
            c_desc.store([out_m, out_n + BLOCK_N // 2], acc1.to(tl.bfloat16))
        else:
            c_desc.store([out_m, out_n], accumulator.to(tl.bfloat16))


def gpu_snapshot(index: int) -> dict[str, str]:
    fields = "name,uuid,power.draw,power.limit,clocks.sm,temperature.gpu"
    values = subprocess.check_output(
        ["nvidia-smi", "-i", str(index), f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
        text=True,
    ).strip().split(", ")
    return dict(zip(fields.split(","), values))


def config_dict(config, timing=None):
    row = dict(config.kwargs)
    row.update(num_stages=config.num_stages, num_warps=config.num_warps)
    if timing is not None:
        row.update(median_ms=timing[0], p20_ms=timing[1], p80_ms=timing[2])
    return row


def event_batch(run, iterations: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        run()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / iterations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--power-watts", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--batch", type=int, default=50)
    args = parser.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu):
        raise RuntimeError("set CUDA_VISIBLE_DEVICES to the same GPU before starting Python")
    cache_dir = os.environ.get("TRITON_CACHE_DIR")
    if not cache_dir:
        raise RuntimeError("TRITON_CACHE_DIR must point at a fresh isolated directory")
    cache_path = Path(cache_dir)
    if any(cache_path.iterdir()):
        raise RuntimeError(f"refusing non-empty Triton cache: {cache_path}")

    torch.manual_seed(42)
    m = n = k = 4096
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
    output = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    a_desc = TensorDescriptor.from_tensor(a, [1, 1])
    b_desc = TensorDescriptor.from_tensor(b, [1, 1])
    c_desc = TensorDescriptor.from_tensor(output, [1, 1])
    sms = torch.cuda.get_device_properties(0).multi_processor_count

    def grid(meta):
        tiles = triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"])
        return (min(sms, tiles),)

    before = gpu_snapshot(args.physical_gpu)
    if round(float(before["power.limit"])) != args.power_watts:
        raise RuntimeError(f"expected {args.power_watts} W cap, got {before['power.limit']} W")

    def run():
        persistent_bf16[grid](a_desc, b_desc, c_desc, m, n, k, sms)

    autotune_start = time.time()
    run()
    torch.cuda.synchronize()
    autotune_seconds = time.time() - autotune_start

    timing_pairs = [
        (config_dict(config, timing), timing[0])
        for config, timing in persistent_bf16.configs_timings.items()
    ]
    timings = [row for row, _ in sorted(timing_pairs, key=lambda item: item[1])]
    winner = config_dict(persistent_bf16.best_config)

    expected = torch.mm(a, b.T)
    torch.cuda.synchronize()
    delta = (output.float() - expected.float()).abs()
    correctness = {
        "max_abs": delta.max().item(),
        "mean_abs": delta.mean().item(),
        "allclose_rtol_1e-2_atol_1e-2": torch.allclose(output, expected, rtol=1e-2, atol=1e-2),
    }

    for _ in range(args.warmup):
        run()
    torch.cuda.synchronize()
    samples_us = [event_batch(run, args.batch) for _ in range(args.samples)]
    runtime_us = statistics.median(samples_us)
    flops = 2 * m * n * k

    result = {
        "schema_version": 1,
        "timestamp_unix": time.time(),
        "physical_gpu": args.physical_gpu,
        "power_watts": args.power_watts,
        "problem": {"M": m, "N": n, "K": k, "dtype": "bfloat16", "operation": "A @ B.T"},
        "search_space_size": len(CONFIGS),
        "cache_dir": str(cache_path.resolve()),
        "autotune_seconds": autotune_seconds,
        "winner": winner,
        "autotuner_timings": timings,
        "validation": {
            "warmup": args.warmup,
            "samples": args.samples,
            "batch": args.batch,
            "median_us": runtime_us,
            "min_us": min(samples_us),
            "max_us": max(samples_us),
            "tflops": flops / (runtime_us * 1e-6) / 1e12,
            "samples_us": samples_us,
        },
        "correctness": correctness,
        "gpu_before": before,
        "gpu_after": gpu_snapshot(args.physical_gpu),
        "torch_version": torch.__version__,
        "triton_version": triton.__version__,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
