"""Profile a workload factory returning (compute, example_inputs).

python -m benchmarks.plot_sm_overlap --workload my_module:make_workload
python -m benchmarks.plot_sm_overlap --replot .benchmarks/sm_overlap/sm_overlap.json
"""

import argparse
import importlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--workload", help="module:factory returning (compute, inputs)")
    source.add_argument("--replot", type=Path, help="saved JSON capture; requires no GPU")
    parser.add_argument("--output", type=Path, default=Path(".benchmarks/sm_overlap"))
    parser.add_argument("--reference", choices=("eager", "inductor"), default="inductor")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--replays", type=int, default=100)
    parser.add_argument("--capacity", type=int, default=2048)
    parser.add_argument("--rtol", type=float, default=.015)
    parser.add_argument("--atol", type=float, default=.015)
    args = parser.parse_args()
    from ayla.profiling.sm_overlap import plot_capture, profile_workload

    if args.replot:
        data = json.loads(args.replot.read_text())
    else:
        module, separator, factory = args.workload.partition(":")
        if not separator or not module or not factory:
            parser.error("workload must be module:factory")
        compute, inputs = getattr(importlib.import_module(module), factory)()
        import torch
        reference = torch.compile(compute) if args.reference == "inductor" else compute
        data = profile_workload(
            compute, inputs, reference=reference, output=args.output,
            name=args.workload, samples=args.samples, rounds=args.rounds,
            replays=args.replays, capacity=args.capacity, rtol=args.rtol, atol=args.atol,
        )
    plot_capture(data, args.output)
    print(args.output / "sm_overlap.png")


if __name__ == "__main__":
    main()
