# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepare isolated inline, batched and WAL-capacity-reuse controls."""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must not exist")
    here = Path(__file__).resolve().parent
    output = args.output.resolve()
    subprocess.run([sys.executable, str(here.parent / "batching/prepare.py"), str(output)], check=True)
    expected = json.loads((here / "patch-inputs.json").read_text())
    for name, hashes in expected.items():
        if digest(output / "batched" / name) != hashes["batched"]:
            parser.error(f"Batched source changed: {name}; use the recorded source revision")
    target = output / "reused"
    shutil.copytree(output / "batched", target)
    patched = subprocess.run(
        ["patch", "--batch", "--fuzz=0", "-p1", "-i", str(here / "reused.patch")],
        cwd=target,
        check=True,
        capture_output=True,
        text=True,
    )
    (output / "reuse-patch.log").write_text(patched.stdout + patched.stderr)
    for name, hashes in expected.items():
        assert digest(target / name) == hashes["reused"], name
    report = json.loads((output / "source-inputs.json").read_text())
    report["reused"] = {
        str(path.relative_to(target)): digest(path)
        for directory in ("src", "include", "tests")
        for path in (target / directory).rglob("*")
        if path.is_file()
    }
    (output / "source-inputs.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
