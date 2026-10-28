"""Check installed routing with three independent weight sets and changed inputs."""

import argparse
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402 -- script path setup precedes local imports
from ayla.gemm_jit import gemm_op  # noqa: E402 -- script path setup precedes local imports
from benchmarks.common import (  # noqa: E402 -- script path setup precedes local imports
    out_parameter_addmm_op,
    out_parameter_op,
    torch_addmm_nt,
    torch_mm_nt,
)

p = argparse.ArgumentParser()
p.add_argument("job", type=Path)
p.add_argument("output", type=Path)
a = p.parse_args()
job = json.loads(a.job.read_text())
pr = next(
    p
    for p in json.loads((HERE / "priorities.json").read_text())
    if p["target"] == job["target"]
)
m, n, k = (pr[x] for x in ("m", "n", "k"))
op = pr["op"]
torch.manual_seed(32452843)
activation = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
inputs = []
for _ in range(3):
    bt = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    inputs.append(
        [torch.randn(n, device="cuda", dtype=torch.bfloat16), activation, bt]
        if op == "addmm"
        else [activation, bt]
    )
compute = (out_parameter_addmm_op if op == "addmm" else out_parameter_op)(
    gemm_op(op, "bfloat16")
)
stream = torch.cuda.Stream()
stream.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(stream):
    for _ in range(3):
        for args in inputs:
            compute(*args)
stream.synchronize()
graph = torch.cuda.CUDAGraph()
snapshots = []
with torch.cuda.graph(graph, stream=stream):
    for args in inputs:
        activation.mul_(1)
        snapshots.append(compute(*args).clone())
checks = []
for replay in range(3):
    if replay:
        activation.normal_()
        for args in inputs:
            args[-1].normal_()
            if op == "addmm":
                args[0].normal_()
    expected = [
        (torch_addmm_nt if op == "addmm" else torch_mm_nt)(*args) for args in inputs
    ]
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        graph.replay()
    stream.synchronize()
    for layer, (actual, reference) in enumerate(zip(snapshots, expected, strict=True)):
        diff = actual.float() - reference.float()
        max_abs = float(diff.abs().max())
        rrms = float(
            diff.square().mean().sqrt() / reference.float().square().mean().sqrt()
        )
        assert (
            math.isfinite(max_abs)
            and math.isfinite(rrms)
            and max_abs <= 32
            and rrms <= 0.002
        ), (replay, layer, max_abs, rrms)
        checks.append(
            dict(replay=replay, layer=layer, max_abs=max_abs, relative_rms=rrms)
        )
a.output.write_text(
    json.dumps([dict(target=job["target"], status="ok", checks=checks)], indent=2)
    + "\n"
)
print(job["target"], "graph check passed", flush=True)
