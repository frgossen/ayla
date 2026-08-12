#!/usr/bin/env python3
"""Interleave the pinned Triton config and both power-autotune winners."""
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
from triton.tools.tensor_descriptor import TensorDescriptor

from triton_power_autotune import persistent_bf16


CONFIGS = [
    {"name": "pinned_s3_unsplit", "BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64,
     "GROUP_M": 8, "EPILOGUE_SUBTILE": False, "num_stages": 3, "num_warps": 8},
    {"name": "s3_split", "BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64,
     "GROUP_M": 8, "EPILOGUE_SUBTILE": True, "num_stages": 3, "num_warps": 8},
    {"name": "s4_split", "BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64,
     "GROUP_M": 8, "EPILOGUE_SUBTILE": True, "num_stages": 4, "num_warps": 8},
]


def snapshot(index):
    fields = "power.draw,power.limit,clocks.sm,temperature.gpu"
    values = subprocess.check_output(
        ["nvidia-smi", "-i", str(index), f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
        text=True,
    ).strip().split(", ")
    return dict(zip(fields.split(","), values))


def event_batch(run, iterations):
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
    parser.add_argument("--warmup-rounds", type=int, default=1000)
    parser.add_argument("--rounds", type=int, default=80)
    parser.add_argument("--batch", type=int, default=10)
    args = parser.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu):
        raise RuntimeError("CUDA_VISIBLE_DEVICES does not match --physical-gpu")
    before = snapshot(args.physical_gpu)
    if round(float(before["power.limit"])) != args.power_watts:
        raise RuntimeError(f"wrong power cap: {before}")

    torch.manual_seed(42)
    m = n = k = 4096
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
    output = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    a_desc = TensorDescriptor.from_tensor(a, [1, 1])
    b_desc = TensorDescriptor.from_tensor(b, [1, 1])
    c_desc = TensorDescriptor.from_tensor(output, [1, 1])
    sms = torch.cuda.get_device_properties(0).multi_processor_count

    runners = []
    for config in CONFIGS:
        launch = {key: value for key, value in config.items() if key != "name"}

        def run(launch=launch):
            bm, bn, bk = launch["BLOCK_M"], launch["BLOCK_N"], launch["BLOCK_K"]
            a_desc.block_shape = [bm, bk]
            b_desc.block_shape = [bn, bk]
            c_desc.block_shape = [bm, bn // 2 if launch["EPILOGUE_SUBTILE"] else bn]
            grid = (min(sms, triton.cdiv(m, bm) * triton.cdiv(n, bn)),)
            persistent_bf16.fn[grid](a_desc, b_desc, c_desc, m, n, k, sms, **launch)

        runners.append((config["name"], run))

    # Compile first, then heat every variant equally in alternating order.
    for _, run in runners:
        run()
    torch.cuda.synchronize()
    for outer in range(args.warmup_rounds):
        order = runners if outer % 2 == 0 else list(reversed(runners))
        for _, run in order:
            run()
    torch.cuda.synchronize()

    samples = {name: [] for name, _ in runners}
    for outer in range(args.rounds):
        order = runners if outer % 2 == 0 else list(reversed(runners))
        for name, run in order:
            samples[name].append(event_batch(run, args.batch))

    expected = torch.mm(a, b.T)
    correctness = {}
    for name, run in runners:
        run()
        torch.cuda.synchronize()
        correctness[name] = {
            "max_abs": (output.float() - expected.float()).abs().max().item(),
            "allclose": torch.allclose(output, expected, rtol=1e-2, atol=1e-2),
        }

    flops = 2 * m * n * k
    rows = {}
    for config in CONFIGS:
        name = config["name"]
        median_us = statistics.median(samples[name])
        rows[name] = {
            "config": config,
            "median_us": median_us,
            "min_us": min(samples[name]),
            "max_us": max(samples[name]),
            "tflops": flops / (median_us * 1e-6) / 1e12,
            "samples_us": samples[name],
        }
    result = {
        "timestamp_unix": time.time(),
        "physical_gpu": args.physical_gpu,
        "power_watts": args.power_watts,
        "protocol": {"warmup_rounds": args.warmup_rounds, "rounds": args.rounds, "batch": args.batch,
                     "ordering": "forward/reverse round robin"},
        "results": rows,
        "correctness": correctness,
        "gpu_before": before,
        "gpu_after": snapshot(args.physical_gpu),
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({name: {"median_us": row["median_us"], "tflops": row["tflops"]}
                      for name, row in rows.items()}, indent=2))


if __name__ == "__main__":
    main()
