#!/usr/bin/env python3
"""Sweep an idle GPU's power limit across the best 4096 BF16 GEMMs.

The selected physical GPU is exposed as logical CUDA device 0.  Its original
power limit is restored even when measurement fails or is interrupted.  Each
result is appended to CSV immediately so a partial run remains useful.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import socket
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


FLOPS = 2 * 4096**3
AYLA_OP = "auto_matmul_4096_wgmma256_wg_tma_store_k3_4tiles_split2_dbuf2"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXTRAPOLATION_ROOT = REPOSITORY_ROOT / "extrapolation"
REFERENCE_FILES = {
    "Ayla best": EXTRAPOLATION_ROOT / "references/ayla_bf16_gemm.py",
    "Triton TMA persistent": EXTRAPOLATION_ROOT / "references/triton_bf16_gemm.py",
    "CUTLASS CuTe DSL": EXTRAPOLATION_ROOT / "references/cutlass_bf16_gemm.py",
}
TRITON_POWER_POLICIES = {
    "Triton tuned @ 350 W": {
        "BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8,
        "EPILOGUE_SUBTILE": True, "num_stages": 3, "num_warps": 8,
    },
    "Triton tuned @ 500 W": {
        "BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8,
        "EPILOGUE_SUBTILE": True, "num_stages": 4, "num_warps": 8,
    },
    "Triton current (@ 650 W)": {
        "BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8,
        "EPILOGUE_SUBTILE": False, "num_stages": 3, "num_warps": 8,
    },
}
QUERY = (
    "clocks.sm,power.draw,temperature.gpu,"
    "clocks_throttle_reasons.sw_power_cap"
)
FIELDS = [
    "power_limit_w",
    "implementation",
    "median_ms",
    "mean_ms",
    "stdev_ms",
    "tflops",
    "mean_power_w",
    "median_power_w",
    "power_stdev_w",
    "tflops_per_watt",
    "joules_per_gemm",
    "median_sm_clock_mhz",
    "max_temperature_c",
    "power_capped_fraction",
    "timing_batches",
    "power_samples",
]


def _run(command: list[str], *, check: bool = True) -> str:
    result = subprocess.run(command, capture_output=True, text=True)
    if check and result.returncode:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(command)}\n"
            f"{result.stderr.strip()}"
        )
    return result.stdout.strip()


def _gpu_row(gpu: int) -> dict[str, str]:
    fields = [
        "index", "uuid", "name", "utilization.gpu", "utilization.memory",
        "memory.used", "temperature.gpu", "power.draw", "power.limit", "power.min_limit",
        "power.max_limit",
    ]
    values = _run([
        "nvidia-smi", "-i", str(gpu), f"--query-gpu={','.join(fields)}",
        "--format=csv,noheader,nounits",
    ]).split(",")
    return dict(zip(fields, (value.strip() for value in values)))


def _compute_processes(uuid: str) -> list[str]:
    output = _run([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ], check=False)
    return [line for line in output.splitlines() if line.startswith(uuid + ",")]


def _require_idle_gpu(gpu: int) -> dict[str, str]:
    row = _gpu_row(gpu)
    processes = _compute_processes(row["uuid"])
    if processes:
        raise RuntimeError(
            f"GPU {gpu} has active compute processes:\n" + "\n".join(processes)
        )
    utilization = float(row["utilization.gpu"])
    memory_used = float(row["memory.used"])
    if utilization > 5 or memory_used > 256:
        raise RuntimeError(
            f"GPU {gpu} is not idle: utilization={utilization:g}%, "
            f"memory.used={memory_used:g} MiB"
        )
    return row


def _set_power_limit(gpu: int, watts: float) -> None:
    output = _run([
        "sudo", "-n", "nvidia-smi", "-i", str(gpu),
        "--power-limit", f"{watts:g}",
    ])
    print(output, flush=True)


class Sampler(threading.Thread):
    def __init__(self, gpu: int, interval: float) -> None:
        super().__init__(daemon=True)
        self.gpu = gpu
        self.interval = interval
        self.done = threading.Event()
        self.clocks: list[float] = []
        self.powers: list[float] = []
        self.temperatures: list[float] = []
        self.capped: list[bool] = []
        self.error: Exception | None = None

    def run(self) -> None:
        try:
            while not self.done.is_set():
                output = _run([
                    "nvidia-smi", "-i", str(self.gpu),
                    f"--query-gpu={QUERY}", "--format=csv,noheader,nounits",
                ])
                clock, power, temperature, capped = (
                    value.strip() for value in output.split(",")
                )
                self.clocks.append(float(clock))
                self.powers.append(float(power))
                self.temperatures.append(float(temperature))
                self.capped.append(capped == "Active")
                self.done.wait(self.interval)
        except Exception as error:  # surfaced by finish()
            self.error = error

    def finish(self) -> None:
        self.done.set()
        self.join()
        if self.error is not None:
            raise self.error


def _continuous_load(op: Callable, seconds: float, batch: int, torch) -> None:
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        for _ in range(batch):
            op()
        torch.cuda.synchronize()


def _measure(
    name: str,
    op: Callable,
    *,
    gpu: int,
    power_limit: int,
    seconds: float,
    ramp_seconds: float,
    batch: int,
    sample_interval: float,
    torch,
) -> dict[str, float | int | str]:
    _continuous_load(op, ramp_seconds, batch, torch)
    sampler = Sampler(gpu, sample_interval)
    sampler.start()
    batch_times: list[float] = []
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(batch):
            op()
        end.record()
        torch.cuda.synchronize()
        batch_times.append(start.elapsed_time(end) / batch)
    sampler.finish()

    if not sampler.powers or not batch_times:
        raise RuntimeError("measurement produced no timing or power samples")
    median_ms = statistics.median(batch_times)
    mean_power = statistics.mean(sampler.powers)
    result: dict[str, float | int | str] = {
        "power_limit_w": power_limit,
        "implementation": name,
        "median_ms": median_ms,
        "mean_ms": statistics.mean(batch_times),
        "stdev_ms": statistics.stdev(batch_times) if len(batch_times) > 1 else 0.0,
        "tflops": FLOPS / (median_ms / 1e3) / 1e12,
        "mean_power_w": mean_power,
        "median_power_w": statistics.median(sampler.powers),
        "power_stdev_w": (
            statistics.stdev(sampler.powers) if len(sampler.powers) > 1 else 0.0
        ),
        "median_sm_clock_mhz": statistics.median(sampler.clocks),
        "max_temperature_c": max(sampler.temperatures),
        "power_capped_fraction": sum(sampler.capped) / len(sampler.capped),
        "timing_batches": len(batch_times),
        "power_samples": len(sampler.powers),
    }
    result["tflops_per_watt"] = float(result["tflops"]) / mean_power
    result["joules_per_gemm"] = mean_power * median_ms / 1e3
    return result


def _combine_results(repetitions: list[dict]) -> dict:
    """Combine forward/reverse passes into one row per implementation/limit."""
    if not repetitions:
        raise ValueError("cannot combine an empty result list")
    median_ms = statistics.median(float(row["median_ms"]) for row in repetitions)
    mean_power = statistics.mean(float(row["mean_power_w"]) for row in repetitions)
    timing_batches = sum(int(row["timing_batches"]) for row in repetitions)
    power_samples = sum(int(row["power_samples"]) for row in repetitions)
    result = {
        "power_limit_w": repetitions[0]["power_limit_w"],
        "implementation": repetitions[0]["implementation"],
        "median_ms": median_ms,
        "mean_ms": statistics.mean(float(row["mean_ms"]) for row in repetitions),
        "stdev_ms": statistics.mean(float(row["stdev_ms"]) for row in repetitions),
        "tflops": FLOPS / (median_ms / 1e3) / 1e12,
        "mean_power_w": mean_power,
        "median_power_w": statistics.median(
            float(row["median_power_w"]) for row in repetitions
        ),
        "power_stdev_w": statistics.mean(
            float(row["power_stdev_w"]) for row in repetitions
        ),
        "median_sm_clock_mhz": statistics.median(
            float(row["median_sm_clock_mhz"]) for row in repetitions
        ),
        "max_temperature_c": max(
            float(row["max_temperature_c"]) for row in repetitions
        ),
        "power_capped_fraction": sum(
            float(row["power_capped_fraction"]) * int(row["power_samples"])
            for row in repetitions
        ) / power_samples,
        "timing_batches": timing_batches,
        "power_samples": power_samples,
    }
    result["tflops_per_watt"] = float(result["tflops"]) / mean_power
    result["joules_per_gemm"] = mean_power * median_ms / 1e3
    return result


def _load_reference(path: Path, ordinal: int):
    spec = importlib.util.spec_from_file_location(f"power_reference_{ordinal}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load reference {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_triton_power_policies(inputs, torch):
    """Create fixed launchers for the power-specific autotune winners."""
    import triton
    from triton.tools.tensor_descriptor import TensorDescriptor

    tools_dir = str(Path(__file__).resolve().parent)
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    from triton_power_autotune import persistent_bf16

    a, b = inputs["a"], inputs["b"]
    m, k = a.shape
    n = b.shape[0]
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    operations = {}
    metadata = {}
    for name, config in TRITON_POWER_POLICIES.items():
        output = torch.empty((m, n), dtype=a.dtype, device=a.device)
        a_desc = TensorDescriptor.from_tensor(a, [1, 1])
        b_desc = TensorDescriptor.from_tensor(b, [1, 1])
        c_desc = TensorDescriptor.from_tensor(output, [1, 1])
        launch = dict(config)

        def run(
            launch=launch, output=output, a_desc=a_desc, b_desc=b_desc, c_desc=c_desc,
        ):
            bm, bn, bk = launch["BLOCK_M"], launch["BLOCK_N"], launch["BLOCK_K"]
            a_desc.block_shape = [bm, bk]
            b_desc.block_shape = [bn, bk]
            c_desc.block_shape = [bm, bn // 2 if launch["EPILOGUE_SUBTILE"] else bn]
            grid = (min(sms, triton.cdiv(m, bm) * triton.cdiv(n, bn)),)
            persistent_bf16.fn[grid](
                a_desc, b_desc, c_desc, m, n, k, sms, **launch,
            )
            return output

        operations[name] = run
        metadata[name] = {"backend": "Triton TMA persistent", **config}
    return operations, metadata


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _git_revision() -> str:
    return _run(["git", "rev-parse", "HEAD"], check=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True, help="physical GPU index")
    parser.add_argument(
        "--limits", type=int, nargs="+", default=[350, 450, 550, 650],
    )
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--ramp-seconds", type=float, default=1.5)
    parser.add_argument("--cooldown-seconds", type=float, default=2.0)
    parser.add_argument("--batch", type=int, default=50)
    parser.add_argument("--sample-interval", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--passes", type=int, default=2,
        help="passes per limit; alternating passes reverse implementation order",
    )
    parser.add_argument(
        "--reverse-order", action="store_true",
        help="measure Ayla before cuBLAS at the first power limit",
    )
    parser.add_argument(
        "--only", choices=["cublas", "ayla", "triton", "cutlass"],
        help="measure only one implementation (useful for isolated reruns)",
    )
    parser.add_argument(
        "--triton-policy-sweep", action="store_true",
        help="measure the 350 W/500 W autotune winners and current Triton policy only",
    )
    parser.add_argument(
        "--triton-policy", choices=list(TRITON_POWER_POLICIES),
        help="measure one fixed Triton policy in an isolated process",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    initial = _require_idle_gpu(args.gpu)
    limits = list(dict.fromkeys(args.limits))
    minimum = float(initial["power.min_limit"])
    maximum = float(initial["power.max_limit"])
    if any(limit < minimum or limit > maximum for limit in limits):
        raise SystemExit(f"power limits must be within [{minimum:g}, {maximum:g}] W")

    # This must happen before importing torch: physical GPU args.gpu becomes
    # logical CUDA device 0 for the benchmark process.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    python_bin = str(Path(sys.executable).resolve().parent)
    os.environ["PATH"] = os.pathsep.join(
        [python_bin, *os.environ.get("PATH", "").split(os.pathsep)]
    )
    sys.path.insert(0, str(EXTRAPOLATION_ROOT))
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("selected CUDA device is not available")
    torch.manual_seed(args.seed)
    a = torch.randn(4096, 4096, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(4096, 4096, dtype=torch.bfloat16, device="cuda")
    cublas_output = torch.empty_like(a)
    inputs = {"a": a, "b": b}
    if args.triton_policy_sweep or args.triton_policy:
        if args.only is not None:
            raise ValueError("--only cannot be combined with a Triton policy sweep")
        operations, reference_metadata = _make_triton_power_policies(inputs, torch)
        if args.triton_policy:
            operations = {args.triton_policy: operations[args.triton_policy]}
            reference_metadata = {
                args.triton_policy: reference_metadata[args.triton_policy]
            }
    else:
        operations = {
            "cuBLAS torch.mm": lambda: torch.mm(a, b.t(), out=cublas_output),
        }
        reference_metadata = {}
        for ordinal, (name, path) in enumerate(REFERENCE_FILES.items()):
            module = _load_reference(path, ordinal)
            prepared = module.prepare(inputs)
            operations[name] = prepared.run if hasattr(prepared, "run") else prepared
            reference_metadata[name] = {
                "path": str(path),
                **getattr(module, "METADATA", {}),
            }

    reference = torch.mm(a, b.t(), out=cublas_output)
    correctness = {}
    for name, operation in operations.items():
        candidate = operation()
        torch.cuda.synchronize()
        max_error = (reference.float() - candidate.float()).abs().max().item()
        if not torch.allclose(reference, candidate, atol=2.0):
            raise RuntimeError(f"{name} correctness check failed: max error {max_error:g}")
        correctness[name] = max_error
    torch.cuda.synchronize()
    if args.only is not None:
        selected_name = {
            "cublas": "cuBLAS torch.mm",
            "ayla": "Ayla best",
            "triton": "Triton TMA persistent",
            "cutlass": "CUTLASS CuTe DSL",
        }[args.only]
        operations = {selected_name: operations[selected_name]}

    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / "power_efficiency.csv"
    metadata_path = args.output / "metadata.json"
    original_limit = float(initial["power.limit"])
    metadata = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "git_revision": _git_revision(),
        "gpu": initial,
        "physical_gpu_index": args.gpu,
        "original_power_limit_w": original_limit,
        "requested_power_limits_w": limits,
        "seconds_per_measurement": args.seconds,
        "ramp_seconds": args.ramp_seconds,
        "cooldown_seconds": args.cooldown_seconds,
        "sample_interval_seconds": args.sample_interval,
        "launches_per_timing_batch": args.batch,
        "passes_per_power_limit": args.passes,
        "seed": args.seed,
        "shape": [4096, 4096, 4096],
        "dtype": "bfloat16",
        "flops_per_gemm": FLOPS,
        "ayla_op": AYLA_OP,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "driver_version": _run([
            "nvidia-smi", "-i", str(args.gpu), "--query-gpu=driver_version",
            "--format=csv,noheader",
        ]).strip(),
        "correctness_max_abs_error_vs_cublas_bf16": correctness,
        "reference_metadata": reference_metadata,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    rows: list[dict] = []
    try:
        for limit_index, limit in enumerate(limits):
            # Ensure the GPU was not claimed between measurements.
            processes = _compute_processes(initial["uuid"])
            own_pid = str(os.getpid())
            foreign = [line for line in processes if f", {own_pid}," not in line]
            if foreign:
                raise RuntimeError("selected GPU was claimed:\n" + "\n".join(foreign))
            _set_power_limit(args.gpu, limit)
            time.sleep(args.cooldown_seconds)
            repetitions = {name: [] for name in operations}
            base_order = list(operations)
            if bool(limit_index % 2) != args.reverse_order:
                base_order.reverse()
            for pass_index in range(args.passes):
                order = base_order if pass_index % 2 == 0 else list(reversed(base_order))
                for name in order:
                    result = _measure(
                        name, operations[name], gpu=args.gpu,
                        power_limit=limit, seconds=args.seconds,
                        ramp_seconds=args.ramp_seconds, batch=args.batch,
                        sample_interval=args.sample_interval, torch=torch,
                    )
                    repetitions[name].append(result)
                    print(json.dumps({"pass": pass_index, **result}, sort_keys=True), flush=True)
            for name in operations:
                result = _combine_results(repetitions[name])
                rows.append(result)
                _write_csv(csv_path, rows)
                print(json.dumps({"combined": True, **result}, sort_keys=True), flush=True)
    finally:
        _set_power_limit(args.gpu, original_limit)

    metadata["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    metadata["restored_power_limit_w"] = float(_gpu_row(args.gpu)["power.limit"])
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"wrote {csv_path} and {metadata_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
