"""Five profiles on two GPUs; all diagnostic counters use normal kernels."""

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
out = HERE / "ncu"
out.mkdir(exist_ok=True)
ps = json.loads((HERE / "priorities.json").read_text())
leaders = json.loads((HERE / "leaders.json").read_text())


def worker(gpu, rank):
    pr = next(p for p in ps if p["rank"] == rank)
    leader = next(r for r in leaders if r["target"] == pr["target"])
    job = dict(
        target=pr["target"],
        entries=[
            dict(configuration=pr["configuration"]),
            dict(configuration=leader["configuration"]),
        ],
    )
    path = out / f"r{rank}_job.json"
    path.write_text(json.dumps(job, indent=2) + "\n")
    for kind in (
        ("torch", "incumbent", "leader") if rank == 23 else ("torch", "incumbent")
    ):
        name = f"r{rank}_{kind}"
        report = out / (name + ".ncu-rep")
        if report.exists() and (out / (name + ".csv")).exists():
            continue
        cmd = [
            "/usr/local/cuda-13.0/bin/ncu",
            "--config-file",
            "0",
            "--force-overwrite",
            "--profile-from-start",
            "off",
            "--replay-mode",
            "kernel",
            "--cache-control",
            "all",
            "--clock-control",
            "none",
            "--pipeline-boost-state",
            "dynamic",
            "--launch-count",
            "1",
        ]
        for section in (
            "SpeedOfLight",
            "LaunchStats",
            "Occupancy",
            "SchedulerStats",
            "MemoryWorkloadAnalysis",
            "WarpStateStats",
        ):
            cmd += ["--section", section]
        cmd += [
            "--export",
            str(out / name),
            sys.executable,
            str(HERE / "profile.py"),
            str(path),
            "--entry",
            "1" if kind == "leader" else "0",
        ]
        if kind == "torch":
            cmd.append("--torch")
        env = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES=str(gpu),
            PYTHONPATH=str(ROOT),
            TORCH_NO_COMPILER_WRAPPER="1",
            OMP_NUM_THREADS="1",
        )
        with (out / (name + ".log")).open("w") as log:
            try:
                r = subprocess.run(
                    cmd, cwd=ROOT, env=env, stdout=log, stderr=log, timeout=240
                )
            except subprocess.TimeoutExpired:
                print("NCU", name, "timeout", flush=True)
                continue
        if r.returncode == 0 and report.exists():
            with (out / (name + ".csv")).open("w") as stream:
                subprocess.run(
                    [
                        "/usr/local/cuda-13.0/bin/ncu",
                        "--import",
                        str(report),
                        "--csv",
                        "--page",
                        "raw",
                    ],
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
        print("NCU", name, "exit", r.returncode, flush=True)


with ThreadPoolExecutor(max_workers=2) as pool:
    for f in (pool.submit(worker, 6, 23), pool.submit(worker, 7, 19)):
        f.result()
