"""Run one prepared candidate inside an NVTX range for Nsight Compute."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark.run import gpu_snapshot, load_module


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--launches", type=int, default=1)
    parser.add_argument("--allow-busy", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    before = gpu_snapshot(args.gpu)
    if not args.allow_busy and (
        int(before["utilization.gpu"]) > 5 or int(before["memory.used"]) > 1024
    ):
        raise RuntimeError(f"GPU {args.gpu} is not idle enough: {before}")

    python_bin = str(Path(sys.executable).resolve().parent)
    os.environ["PATH"] = os.pathsep.join(
        [python_bin, *os.environ.get("PATH", "").split(os.pathsep)]
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    sys.path.insert(0, str(ROOT))

    import torch
    from benchmark.problems import PROBLEMS

    problem = PROBLEMS[args.problem]
    module = load_module(args.candidate, 0)
    torch.manual_seed(args.seed)
    inputs = problem.make_inputs("cuda", args.seed)
    expected = problem.reference(inputs)
    prepared = module.prepare(inputs)
    runner = prepared.run if hasattr(prepared, "run") else prepared

    actual = runner()
    torch.cuda.synchronize()
    problem.check(actual, expected)
    for _ in range(args.warmup):
        runner()
    torch.cuda.synchronize()

    torch.cuda.nvtx.range_push("candidate")
    for _ in range(args.launches):
        runner()
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    print(f"profiled {getattr(module, 'NAME', args.candidate.stem)} on physical GPU {args.gpu}")


if __name__ == "__main__":
    main()
