"""Unpack frozen benchmark evidence into a fresh, ignored artifact directory."""

import argparse
import gzip
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
p = argparse.ArgumentParser()
p.add_argument(
    "--output",
    type=Path,
    default=ROOT / ".benchmarks/priority_transfer_replay_20261014",
)
a = p.parse_args()
if a.output.exists():
    raise SystemExit(f"{a.output} already exists; choose a fresh --output directory")
evidence = json.loads(gzip.decompress((HERE / "evidence.json.gz").read_bytes()))
a.output.mkdir(parents=True)
for name, value in evidence["artifacts"].items():
    relative = Path(name)
    assert not relative.is_absolute() and ".." not in relative.parts
    path = a.output / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")
for path in HERE.glob("*.py"):
    if path.name != "unpack.py":
        shutil.copy2(path, a.output / path.name)
print(json.dumps(evidence["summary"], indent=2))
print("Extracted to", a.output)
