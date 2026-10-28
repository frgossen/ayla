"""NCU diagnostic for a production seed; its duration is never an admission score."""

import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402 -- script path setup precedes local imports
from ayla.gemm_jit import _seed, _DEFAULTS, build_gemm_module  # noqa: E402 -- script path setup precedes local imports
from ayla.ptx_codegen import make_ptx_kernel  # noqa: E402 -- script path setup precedes local imports
from ayla.ptx_runtime import make_ptx_launcher  # noqa: E402 -- script path setup precedes local imports
from benchmarks.common import torch_addmm_nt, torch_mm_nt, _reset_l2  # noqa: E402 -- script path setup precedes local imports

p = argparse.ArgumentParser()
p.add_argument("job", type=Path)
p.add_argument("--entry", type=int, default=0)
p.add_argument("--torch", action="store_true")
a = p.parse_args()
job = json.loads(a.job.read_text())
pr = next(
    p
    for p in json.loads((HERE / "priorities.json").read_text())
    if p["target"] == job["target"]
)
m, n, k = (pr[x] for x in ("m", "n", "k"))
op = pr["op"]
x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
bt = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
bias = torch.randn(n, dtype=torch.bfloat16, device="cuda") if op == "addmm" else None
args = (bias, x, bt) if bias is not None else (x, bt)
if a.torch:
    def compute():
        return (torch_addmm_nt if op == "addmm" else torch_mm_nt)(*args)
else:
    cfg = {**_DEFAULTS, **job["entries"][a.entry]["configuration"]}
    seed = _seed(cfg, op, m, n, k, (1, m, n, k), exact=True)
    kernel = make_ptx_kernel(build_gemm_module(op, "bfloat16", 1, m, n, k, seed))
    launcher = make_ptx_launcher(kernel)
    out = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")

    def compute():
        return launcher(*args, out)


for _ in range(5):
    compute()
torch.cuda.synchronize()
stream = torch.cuda.current_stream()
flush = torch.zeros(
    2 * torch.cuda.get_device_properties(0).L2_cache_size // 4,
    dtype=torch.float32,
    device="cuda",
)
_reset_l2(flush, stream)
x.mul_(1)
torch.cuda.synchronize()
torch.cuda.profiler.start()
compute()
torch.cuda.synchronize()
torch.cuda.profiler.stop()
