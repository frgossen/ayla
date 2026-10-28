"""One sequential worker per free GPU; bounded jobs and resumable pair output."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PYTHON = Path(sys.executable)
p = argparse.ArgumentParser()
p.add_argument("manifest", type=Path)
p.add_argument("--phase", required=True)
p.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
p.add_argument("--seconds", type=float, default=0.5)
p.add_argument("--rounds", type=int, default=200)
p.add_argument("--timeout", type=int, default=900)
p.add_argument("--confirm", action="store_true")
p.add_argument("--graph", action="store_true")
a = p.parse_args()
phase = HERE / a.phase
phase.mkdir(parents=True, exist_ok=True)
jobs = json.loads(a.manifest.read_text())
work = queue.Queue()
assigned = {int(g): queue.Queue() for g in a.gpus.split(",")}
for i, j in enumerate(jobs):
    output = phase / f"{i:04d}.json"
    if output.exists() and len(json.loads(output.read_text())) == len(j["entries"]):
        continue
    job_path = phase / f"{i:04d}_job.json"
    job_path.write_text(json.dumps(j, indent=2) + "\n")
    (assigned[j["replica_gpu"]] if "replica_gpu" in j else work).put(
        (i, j, job_path, output)
    )
gpus = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True
)
uuid_map = {
    int(line.split(",")[0]): line.split(",")[1].strip() for line in gpus.splitlines()
}


def worker(gpu):
    while True:
        q = assigned[gpu] if not assigned[gpu].empty() else work
        try:
            i, j, job_path, output = q.get_nowait()
        except queue.Empty:
            return
        # Never start a new job on a GPU that another application has occupied.
        occupied = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,gpu_uuid",
                "--format=csv,noheader",
            ],
            text=True,
        )
        if any(uuid_map[gpu] in line for line in occupied.splitlines()):
            q.put((i, j, job_path, output))
            print(f"GPU {gpu} acquired by another job; yielding it", flush=True)
            return
        env = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES=uuid_map[gpu],
            PYTHONPATH=str(ROOT),
            TORCH_NO_COMPILER_WRAPPER="1",
            OMP_NUM_THREADS="1",
        )
        command = [
            str(PYTHON),
            str(HERE / "worker.py"),
            str(job_path),
            str(output),
            "--seconds",
            str(a.seconds),
            "--rounds",
            str(a.rounds),
        ]
        if a.confirm:
            command.append("--confirm")
        if a.graph:
            command = [
                str(PYTHON),
                str(HERE / "graph_check.py"),
                str(job_path),
                str(output),
            ]
        start = time.time()
        with (phase / f"{i:04d}.log").open("w") as log:
            try:
                r = subprocess.run(
                    command,
                    env=env,
                    cwd=ROOT,
                    stdout=log,
                    stderr=log,
                    timeout=a.timeout,
                )
                status = f"exit {r.returncode}"
            except subprocess.TimeoutExpired:
                status = "timeout"
        rows = json.loads(output.read_text()) if output.exists() else []
        print(
            json.dumps(
                dict(
                    phase=a.phase,
                    job=i,
                    gpu=gpu,
                    target=j["target"],
                    status=status,
                    complete=len(rows),
                    expected=len(j["entries"]),
                    seconds=round(time.time() - start, 1),
                )
            ),
            flush=True,
        )
        q.task_done()


with ThreadPoolExecutor(max_workers=len(a.gpus.split(","))) as pool:
    for f in as_completed([pool.submit(worker, int(g)) for g in a.gpus.split(",")]):
        f.result()
if not work.empty() or any(not q.empty() for q in assigned.values()):
    raise SystemExit("Available GPUs exhausted; remaining jobs are resumable")
