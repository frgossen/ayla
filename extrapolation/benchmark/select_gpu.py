"""Print idle-GPU telemetry, least-used devices first."""
import csv
import subprocess


fields = ["index", "name", "uuid", "utilization.gpu", "memory.used", "power.draw", "power.limit", "clocks.sm"]
output = subprocess.check_output(
    ["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"],
    text=True,
)
rows = [dict(zip(fields, row)) for row in csv.reader(output.splitlines(), skipinitialspace=True)]
rows.sort(key=lambda row: (int(row["utilization.gpu"]), int(row["memory.used"]), int(row["index"])))
print("index  util%  memory_MiB  watts/limit  sm_MHz  name")
for row in rows:
    print(
        f"{int(row['index']):5d}  {int(row['utilization.gpu']):5d}  "
        f"{int(row['memory.used']):10d}  {float(row['power.draw']):5.0f}/{float(row['power.limit']):.0f}  "
        f"{int(row['clocks.sm']):6d}  {row['name']}"
    )
