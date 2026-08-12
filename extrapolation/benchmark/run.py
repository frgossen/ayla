"""Fair CUDA benchmark runner using alternating paired A/B batches."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", required=True)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", action="append", type=Path, default=[])
    parser.add_argument("--gpu", type=int, default=3, help="physical nvidia-smi GPU index")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=300, help="round-robin launches per arm")
    parser.add_argument("--batch", type=int, default=20, help="launches per CUDA-event sample")
    parser.add_argument("--pairs", type=int, default=60, help="paired samples per candidate")
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument("--allow-busy", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def gpu_snapshot(index: int) -> dict:
    fields = "name,uuid,utilization.gpu,memory.used,power.draw,power.limit,clocks.sm,temperature.gpu"
    command = [
        "nvidia-smi", "-i", str(index), f"--query-gpu={fields}",
        "--format=csv,noheader,nounits",
    ]
    values = subprocess.check_output(command, text=True).strip().split(", ")
    return dict(zip(fields.split(","), values))


def load_module(path: Path, ordinal: int):
    path = path.resolve()
    spec = importlib.util.spec_from_file_location(f"submission_{ordinal}_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "prepare"):
        raise AttributeError(f"{path} must export prepare(inputs) -> zero-argument callable")
    return module


def cuda_batch(torch, runner, iterations: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        runner()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / iterations


def bootstrap_ci(values, repetitions):
    rng = random.Random(0)
    count = len(values)
    samples = sorted(
        statistics.median(rng.choices(values, k=count))
        for _ in range(repetitions)
    )
    return samples[int(0.025 * repetitions)], samples[int(0.975 * repetitions)]


def main():
    args = parse_args()

    # Python packages such as torch.utils.cpp_extension discover Ninja through
    # PATH, while ``pixi run python`` does not necessarily expose the sibling
    # executables of the active interpreter.  Make the selected environment
    # self-contained before importing any candidate.
    python_bin = str(Path(sys.executable).resolve().parent)
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    if python_bin not in path_entries:
        os.environ["PATH"] = os.pathsep.join([python_bin, *path_entries])

    before = gpu_snapshot(args.gpu)
    if not args.allow_busy and (int(before["utilization.gpu"]) > 5 or int(before["memory.used"]) > 1024):
        raise RuntimeError(f"GPU {args.gpu} is not idle enough: {before}")

    # This must happen before importing torch or any candidate module.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    sys.path.insert(0, str(ROOT))
    import torch
    from benchmark.problems import PROBLEMS

    problem = PROBLEMS[args.problem]
    modules = [load_module(args.baseline, 0)] + [
        load_module(path, i + 1) for i, path in enumerate(args.candidate)
    ]
    for module in modules:
        declared = getattr(module, "PROBLEM", args.problem)
        if declared != args.problem:
            raise ValueError(f"{module.__file__} declares PROBLEM={declared!r}")

    torch.manual_seed(args.seed)
    inputs = problem.make_inputs("cuda", args.seed)
    expected = problem.reference(inputs)
    torch.cuda.synchronize()

    arms = []
    correctness = {}
    arm_metadata = {}
    for module in modules:
        name = getattr(module, "NAME", Path(module.__file__).stem)
        prepared = module.prepare(inputs)
        runner = prepared.run if hasattr(prepared, "run") else prepared
        if not callable(runner):
            raise TypeError(f"{name}.prepare() did not return a callable")
        actual = runner()
        torch.cuda.synchronize()
        correctness[name] = problem.check(actual, expected)
        arms.append((name, runner))
        arm_metadata[name] = {
            "file": str(Path(module.__file__).resolve()),
            **getattr(module, "METADATA", {}),
        }

    # Compile every arm before timing and warm them in round-robin order so no
    # candidate receives a uniquely cold or uniquely hot GPU.
    for i in range(args.warmup):
        order = arms if i % 2 == 0 else list(reversed(arms))
        for _, runner in order:
            runner()
    torch.cuda.synchronize()

    baseline_name, baseline = arms[0]
    comparisons = {}
    candidates = arms[1:]
    candidate_positions = {name: i for i, (name, _) in enumerate(candidates)}
    for outer in range(args.pairs):
        # Reverse candidate order each round to cancel slow thermal drift.
        ordered = candidates if outer % 2 == 0 else list(reversed(candidates))
        for name, runner in ordered:
            row = comparisons.setdefault(name, {"baseline_us": [], "candidate_us": [], "delta_pct": []})
            # Flip A/B order independently for every candidate. Using its
            # current position in ``ordered`` would fail to alternate when an
            # even number of candidates is reversed.
            baseline_first = (outer + candidate_positions[name]) % 2 == 0
            if baseline_first:
                base_us = cuda_batch(torch, baseline, args.batch)
                candidate_us = cuda_batch(torch, runner, args.batch)
            else:
                candidate_us = cuda_batch(torch, runner, args.batch)
                base_us = cuda_batch(torch, baseline, args.batch)
            row["baseline_us"].append(base_us)
            row["candidate_us"].append(candidate_us)
            row["delta_pct"].append(100.0 * (candidate_us - base_us) / base_us)

    rows = {}
    for name, samples in comparisons.items():
        lo, hi = bootstrap_ci(samples["delta_pct"], args.bootstrap)
        candidate_us = statistics.median(samples["candidate_us"])
        rows[name] = {
            "baseline_median_us": statistics.median(samples["baseline_us"]),
            "baseline_tflops": problem.flops
            / (statistics.median(samples["baseline_us"]) * 1e-6)
            / 1e12,
            "candidate_median_us": candidate_us,
            "candidate_tflops": problem.flops / (candidate_us * 1e-6) / 1e12,
            "paired_median_delta_pct": statistics.median(samples["delta_pct"]),
            "paired_delta_95ci_pct": [lo, hi],
            "candidate_wins": sum(delta < 0 for delta in samples["delta_pct"]),
            "pairs": len(samples["delta_pct"]),
        }

    result = {
        "schema_version": 1,
        "timestamp_unix": time.time(),
        "command": sys.argv,
        "problem": args.problem,
        "seed": args.seed,
        "physical_gpu": args.gpu,
        "gpu_before": before,
        "gpu_after": gpu_snapshot(args.gpu),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "baseline": baseline_name,
        "arms": arm_metadata,
        "correctness": correctness,
        "comparisons": rows,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
