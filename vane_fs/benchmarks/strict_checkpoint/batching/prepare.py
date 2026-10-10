# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepare isolated inline/batched strict-checkpoint controls."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must not exist")
    here = Path(__file__).resolve().parent
    component = here.parents[2]
    expected = json.loads((here / "patch-inputs.json").read_text())
    for name, hashes in expected.items():
        if digest(component / name) != hashes["baseline"]:
            parser.error(f"Baseline source changed: {name}; use the recorded source revision")
    output = args.output.resolve()
    output.mkdir(parents=True)
    report = {}
    for variant in ("inline", "batched"):
        target = output / variant
        target.mkdir()
        for name in ("src", "include", "python", "tests", "triplets", "LICENSES"):
            shutil.copytree(component / name, target / name, ignore=shutil.ignore_patterns("__pycache__"))
        for name in ("CMakeLists.txt", "pyproject.toml", "vcpkg.json", "README.md", "LICENSE"):
            shutil.copy2(component / name, target / name)
        if variant == "batched":
            patched = subprocess.run(
                ["patch", "--batch", "--fuzz=0", "-p1", "-i", str(here / "batched.patch")],
                cwd=target,
                check=True,
                capture_output=True,
                text=True,
            )
            (output / "patch.log").write_text(patched.stdout + patched.stderr)
        for name, hashes in expected.items():
            assert digest(target / name) == hashes["candidate" if variant == "batched" else "baseline"], name
        report[variant] = {
            str(path.relative_to(target)): digest(path)
            for directory in ("src", "include", "tests")
            for path in (target / directory).rglob("*")
            if path.is_file()
        }
    (output / "source-inputs.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
